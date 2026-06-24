"""WiCER: Compile-Evaluate-Refine loop for improved compilation quality.

Implements the iterative compilation loop described in
arXiv:2605.07068v1: compile (blind) → generate probes → evaluate →
diagnose → refine → repeat.

LLM calls are routed through the shared :class:`knox.llm.LLMClient`
which provides typed errors, failover and observability. When the
client returns ``None`` (all models in the chain failed), the
compiler falls back to deterministic heuristics so the pipeline never
crashes on a single bad provider.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from knox.compilation._two_phase_core import _tokens_to_chars
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.core.models import ChunkTree, CompiledWiki, ResolvedChunk
from knox.core.pipeline_defs import PipelineConfig
from knox.core.template_loader import get_template_loader
from knox.llm.client import JSON_RESPONSE_FORMAT, LLMClient
from knox.llm.format import encode_for_llm

logger = logging.getLogger(__name__)

_loader = get_template_loader()

# Absolute token reserve carved out of the context window for the WiCER probe
# and evaluation prompts (system prompt + framing overhead).  Sized for the
# compact JSON-emitting prompts used by ``probe_questions.md.j2`` and
# ``probe_eval.md.j2``; both are short fixed envelopes.
_WICER_PROMPT_RESERVE_TOKENS: Final[int] = 1024


class WiCERCompiler(BaseCompiler):
    """Compile-Evaluate-Refine loop for improved compilation quality.

    Implements Algorithm 1 from arXiv:2605.07068v1 (CEGAR-style):
    runs up to *max_iterations* rounds of compilation, probe generation,
    LLM-as-judge evaluation, failure diagnosis, and re-compilation with
    a monotonically growing pin set of facts the prior pass dropped.

    The base compiler must be synthesis-capable so the pin set can be
    honored through the generation prompt; a non-synthesis compiler
    (e.g. raw :class:`GraphCompiler`) will silently drop the pin set
    and the loop will not converge.

    Parameters
    ----------
    base_compiler:
        The underlying compiler used for blind and refined compilation.
        Defaults to a fresh :class:`GraphCompiler` (sufficient for tests
        and offline fallback, but the production pipeline wires a
        :class:`~knox.compilation.two_phase.TwoPhaseCompiler` here).
    max_iterations:
        Maximum number of refinement rounds (default 3).
    llm_client:
        Shared LLM client. When ``None``, a default client is built
        from the global default model. Inject a custom client (or one
        with a fallback chain) to control model selection per stage.
    """

    def __init__(
        self,
        config: PipelineConfig,
        llm_client: LLMClient | None = None,
        base_compiler: BaseCompiler | None = None,
    ) -> None:
        self._config = config
        self._compiler = base_compiler or GraphCompiler(
            model=config.llm.default_model,
            min_content_length=config.budget.min_content_length,
        )
        self.max_iterations = config.budget.wicer_max_iterations
        self._client = llm_client  # None is OK — caller handles graceful degradation

        # Output budgets (how many tokens the LLM may emit).
        self._probe_output_budget = config.budget.wicer_probe_output_max
        self._eval_output_budget = config.budget.wicer_eval_output_max

        # Input batch sizes adapt to the model's discovered context window.
        # When no client is available, fall back to a conservative size.
        cw = llm_client.context_window if llm_client else None
        if cw is not None:
            self._probe_input_chars = _tokens_to_chars(
                max(1, cw.max_input - _WICER_PROMPT_RESERVE_TOKENS - self._probe_output_budget),
            )
            self._eval_input_chars = _tokens_to_chars(
                max(1, cw.max_input - _WICER_PROMPT_RESERVE_TOKENS - self._eval_output_budget),
            )
        else:
            self._probe_input_chars = _tokens_to_chars(self._probe_output_budget)
            self._eval_input_chars = _tokens_to_chars(self._eval_output_budget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, chunks: list[ResolvedChunk], *, trees: list[ChunkTree] | None = None, pinned_facts: list[str] | None = None) -> CompiledWiki:
        """Run the WiCER iterative-compilation loop.

        Implements Algorithm 1 from arXiv:2605.07068v1:

        1. ``W_0 = Compile(D)`` — blind compilation via the base compiler
           (synthesis-capable; e.g. :class:`TwoPhaseCompiler`).
        2. For each iteration: probe → evaluate → diagnose failures →
           accumulate critical facts into ``F_cumulative`` →
           ``W_{t+1} = Compile(D, pin=F_cumulative)``.
        3. Stop on convergence (no failures or no new facts) or max iters.

        Returns a :class:`CompiledWiki` — never raises.

        ``pinned_facts`` is an optional caller-supplied seed for
        ``F_cumulative`` (e.g. for chained WiCER passes or external
        constraints). New facts diagnosed by the loop are appended to it.
        """
        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={
                    "chunk_count": 0,
                    "source_count": 0,
                    "source_names": [],
                    "wicer_iterations": 0,
                },
            )

        # F_cumulative — facts that MUST appear in W_{t+1} (CEGAR refinement).
        cumulative_pinned: list[str] = list(dict.fromkeys(f.strip() for f in (pinned_facts or []) if f and f.strip()))

        # Count sources once to drive a determinate progress bar. The base
        # compiler's per-source extraction (the bulk of compile time) reports
        # against this total; WiCER's own steps hold the bar full.
        num_sources = max(len({str(rc.chunk.source_file or "unknown") for rc in chunks}), 1)
        # Propagate the hook to the synthesis base so its extraction counts
        # are visible (otherwise the bar has nothing real to count).
        self._compiler.on_progress = self.on_progress

        wiki = self._blind_compile(chunks, trees=trees, pinned_facts=cumulative_pinned or None)
        self._report(num_sources, num_sources, "blind compilation done")
        iteration_info: dict[str, Any] = {
            "wicer_iterations": 0,
            "wicer_refinements": 0,
            "wicer_probes_generated": 0,
            "wicer_probes_passed": 0,
            "wicer_pinned_facts": len(cumulative_pinned),
        }

        for attempt in range(self.max_iterations):
            self._report(num_sources, num_sources, f"iter {attempt + 1}/{self.max_iterations} — generating probes")
            probes = self._generate_probes(wiki, chunks)
            if not probes:
                logger.info("WiCER: no probes generated, ending loop")
                break
            iteration_info["wicer_probes_generated"] += len(probes)

            self._report(num_sources, num_sources, f"iter {attempt + 1}/{self.max_iterations} — evaluating {len(probes)} probes")
            eval_results = self._evaluate(wiki, probes)
            failures = [r for r in eval_results if not r.get("correct", False)]
            passed = len(eval_results) - len(failures)
            iteration_info["wicer_probes_passed"] += passed

            if not failures:
                logger.info("WiCER: all probes passed at iteration %d", attempt + 1)
                iteration_info["wicer_iterations"] = attempt + 1
                break

            self._report(num_sources, num_sources, f"iter {attempt + 1}/{self.max_iterations} — diagnosing {len(failures)} failures")
            new_facts = self._diagnose(failures)
            if not new_facts:
                logger.info("WiCER: no critical facts extracted, ending loop")
                iteration_info["wicer_iterations"] = attempt + 1
                break

            # F_cumulative ← F_cumulative ∪ D_facts (dedup, preserve order).
            before = len(cumulative_pinned)
            cumulative_pinned = list(dict.fromkeys(cumulative_pinned + [f.strip() for f in new_facts if f and f.strip()]))
            added = len(cumulative_pinned) - before
            if added == 0:
                logger.info("WiCER: all diagnosed facts already pinned, ending loop")
                iteration_info["wicer_iterations"] = attempt + 1
                break

            self._report(num_sources, num_sources, f"iter {attempt + 1}/{self.max_iterations} — recompiling with {len(cumulative_pinned)} pinned fact(s)")
            wiki = self._blind_compile(chunks, trees=trees, pinned_facts=cumulative_pinned)
            iteration_info["wicer_refinements"] += 1
            iteration_info["wicer_iterations"] = attempt + 1
            iteration_info["wicer_pinned_facts"] = len(cumulative_pinned)
        else:
            iteration_info["wicer_iterations"] = self.max_iterations

        wiki.metadata.update(iteration_info)
        return wiki

    # ------------------------------------------------------------------
    # WiCER steps  (each is overridable for testing)
    # ------------------------------------------------------------------

    def _blind_compile(
        self,
        chunks: list[ResolvedChunk],
        *,
        trees: list[ChunkTree] | None = None,
        pinned_facts: list[str] | None = None,
    ) -> CompiledWiki:
        """Step A / D: compile via the base compiler.

        Used both for the initial blind pass (``pinned_facts=None``) and
        for refinement passes that re-run the base compiler with the
        cumulative pin set as a preservation constraint.
        """
        return self._compiler.compile(chunks, trees=trees, pinned_facts=pinned_facts)

    def _generate_probes(self, wiki: CompiledWiki, chunks: list[ResolvedChunk]) -> list[dict]:
        """Step B: generate probe Q&A pairs from source chunks.

        All chunks are processed in batches of ``self._probe_input_chars`` chars
        so no source content is dropped.
        """
        all_probes: list[dict] = []
        seen_questions: set[str] = set()

        # Accumulate chunk texts into budget-sized batches.
        batch_parts: list[str] = []
        batch_len = 0

        for rc in chunks:
            if not rc.chunk.text:
                continue
            if batch_len + len(rc.chunk.text) > self._probe_input_chars and batch_parts:
                probes = self._probe_batch("\n\n".join(batch_parts))
                for p in probes:
                    q = p.get("question", "")
                    if q not in seen_questions:
                        all_probes.append(p)
                        seen_questions.add(q)
                batch_parts = []
                batch_len = 0
            batch_parts.append(rc.chunk.text)
            batch_len += len(rc.chunk.text)

        if batch_parts:
            probes = self._probe_batch("\n\n".join(batch_parts))
            for p in probes:
                q = p.get("question", "")
                if q not in seen_questions:
                    all_probes.append(p)
                    seen_questions.add(q)

        return all_probes

    def _probe_batch(self, combined: str) -> list[dict]:
        """Generate probes from a single batch of combined text."""
        if not combined:
            return []

        if self._client is None:
            logger.debug("WiCER has no LLM client — using fallback probes")
            return self._fallback_probes(combined)

        result = self._client.try_complete(
            messages=[
                {"role": "system", "content": "You are a test-probe generator. Return only valid JSON."},
                {"role": "user", "content": _loader.render("compilation/wicer_probe.md.j2", text=combined)},
            ],
            role="wicer.probe",
            response_format=JSON_RESPONSE_FORMAT,
        )

        probes = self._extract_probe_list(result)
        if probes is not None:
            return probes

        return self._fallback_probes(combined)

    def _evaluate(self, wiki: CompiledWiki, probes: list[dict]) -> list[dict]:
        """Step B: evaluate wiki against probes using LLM-as-judge.

        Wiki content is processed in segments of ``self._eval_input_chars`` chars
        so the full wiki is checked, not just the first few kilobytes.
        A probe is marked correct if it passes in ANY segment.
        """
        wiki_text = "\n\n".join(n.content for n in wiki.nodes if n.content)
        if not wiki_text:
            return [{"question": p.get("question", ""), "expected_answer": p.get("answer", ""), "found_answer": "", "correct": False} for p in probes]

        if self._client is None:
            logger.debug("WiCER has no LLM client — using fallback evaluation")
            return self._fallback_evaluate(wiki_text, probes)

        # Track probe correctness across segments — a probe passing in any
        # segment is considered passed.
        probe_results: dict[int, dict] = {}

        for seg_start in range(0, len(wiki_text), self._eval_input_chars):
            segment = wiki_text[seg_start : seg_start + self._eval_input_chars]

            result = self._client.try_complete(
                messages=[
                    {"role": "system", "content": "You are an impartial judge of factual correctness. Return only valid JSON."},
                    {
                        "role": "user",
                        "content": _loader.render(
                            "compilation/wicer_evaluate.md.j2",
                            wiki_text=segment,
                            probes_json=encode_for_llm(probes, self._config.llm.data_format),
                        ),
                    },
                ],
                role="wicer.judge",
                response_format=JSON_RESPONSE_FORMAT,
            )

            if result is not None and result.json_data is not None:
                results = result.json_data.get("results", [])
                if isinstance(results, list):
                    for i, r in enumerate(results):
                        if i >= len(probes):
                            break
                        if r.get("correct", False) or i not in probe_results:  # ty: ignore
                            probe_results[i] = r  # ty: ignore

        if probe_results:
            return [probe_results.get(i, {"question": p.get("question", ""), "correct": False}) for i, p in enumerate(probes)]

        return self._fallback_evaluate(wiki_text, probes)

    def _diagnose(self, failures: list[dict]) -> list[str]:
        """Step C: extract critical facts from failures using LLM."""
        if not failures:
            return []

        if self._client is None:
            logger.debug("WiCER has no LLM client — skipping diagnosis")
            return [f.get("expected_answer", "") for f in failures if f.get("expected_answer")]

        result = self._client.try_complete(
            messages=[
                {"role": "system", "content": "You are a diagnostic assistant. Return only valid JSON."},
                {
                    "role": "user",
                    "content": _loader.render(
                        "compilation/wicer_diagnose.md.j2",
                        failures_json=encode_for_llm(failures, self._config.llm.data_format),
                    ),
                },
            ],
            role="wicer.refine",
            response_format=JSON_RESPONSE_FORMAT,
        )

        if result is not None and result.json_data is not None:
            facts = result.json_data.get("critical_facts", [])
            if isinstance(facts, list):
                return [str(f) for f in facts if f]

        return [f.get("expected_answer", "") for f in failures if f.get("expected_answer")]

    @staticmethod
    def _extract_probe_list(result: Any) -> list[dict] | None:
        """Extract a list of probes from various JSON shapes the LLM may return.

        Accepts ``[{...}, {...}]`` or ``{"probes": [...]}`` /
        ``{"results": [...]}`` / ``{"items": [...]}``. Falls back to
        the single list-valued key when the object has only one.
        """
        if result is None:
            return None
        # Try direct list shape first (older content path).
        if result.json_data is None:
            try:
                parsed = json.loads(result.content)
            except json.JSONDecodeError:
                logger.warning("LLM returned malformed JSON for probes — using fallback")
                return None
            return parsed if isinstance(parsed, list) else None

        data = result.json_data
        for key in ("probes", "results", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        list_values = [v for v in data.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        return None

    @staticmethod
    def _fallback_probes(text: str) -> list[dict]:
        """Generate simple fallback probes from text sentences."""
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
        probes: list[dict] = []
        for s in sentences[:5]:
            probes.append(
                {
                    "question": f"What does the source say about: {s[:60]}...?",
                    "answer": s[:200],
                }
            )
        return probes

    @staticmethod
    def _fallback_evaluate(wiki_text: str, probes: list[dict]) -> list[dict]:
        """Fallback evaluation using simple substring matching."""
        results: list[dict] = []
        wiki_lower = wiki_text.lower()
        for p in probes:
            answer = p.get("answer", "")
            found = answer.lower()[:50] in wiki_lower if answer else False
            results.append(
                {
                    "question": p.get("question", ""),
                    "expected_answer": answer,
                    "found_answer": answer if found else "",
                    "correct": found,
                }
            )
        return results
