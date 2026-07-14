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

import time
from typing import Any, Final

from knox.agents.deps import WicerDiagnoseDeps, WicerEvaluateDeps, WicerProbeDeps
from knox.agents.wicer import WicerDiagnoseAgent, WicerEvaluateAgent, WicerProbeAgent
from knox.compilation._two_phase_core import _tokens_to_chars
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.core.models import ChunkTree, CompiledWiki, MetadataField, ResolvedChunk, WikiField
from knox.core.pipeline.config import PipelineConfig
from knox.llm.context_window import ContextWindow, discover_context_window
from knox.llm.format import encode_for_llm
from knox.logging import get_logger

logger = get_logger(__name__)

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
        context_window: ContextWindow | None = None,
        base_compiler: BaseCompiler | None = None,
    ) -> None:
        self._config = config
        self._compiler = base_compiler or GraphCompiler(
            model=config.llm.default_model,
            min_content_length=config.budget.min_content_length,
        )
        self.max_iterations = config.budget.wicer_max_iterations
        self._has_llm = bool(config.llm.default_model)

        # Output budgets (how many tokens the LLM may emit).
        self._probe_output_budget = config.budget.wicer_probe_output_max
        self._eval_output_budget = config.budget.wicer_eval_output_max

        # Input batch sizes adapt to the model's discovered context window.
        # When no context window is available, fall back to a conservative size.
        cw = (
            context_window
            or discover_context_window(
                config.llm.default_model,
                fallback_max_input=config.llm.fallback_max_input,
                fallback_max_output=config.llm.fallback_max_output,
                num_ctx_min=config.llm.num_ctx_min,
                num_ctx_max=config.llm.num_ctx_max,
                context_window_cap=config.llm.context_window_cap,
            )
            if config.llm.default_model
            else None
        )
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

    def compile(self, chunks: list[ResolvedChunk], *, trees: list[ChunkTree] | None = None, pinned_facts: list[str] | None = None) -> CompiledWiki:  # noqa: PLR0915
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
                    WikiField.CHUNK_COUNT: 0,
                    MetadataField.SOURCE_COUNT: 0,
                    MetadataField.SOURCE_NAMES: [],
                    "wicer_iterations": 0,
                },
            )

        compile_start = time.monotonic()

        # F_cumulative — facts that MUST appear in W_{t+1} (CEGAR refinement).
        cumulative_pinned: list[str] = list(dict.fromkeys(f.strip() for f in (pinned_facts or []) if f and f.strip()))

        # Count sources once to drive a determinate progress bar. The base
        # compiler's per-source extraction (the bulk of compile time) reports
        # against this total; WiCER's own steps hold the bar full.
        num_sources = max(len({str(rc.chunk.source_file or "unknown") for rc in chunks}), 1)
        # Propagate the hook to the synthesis base so its extraction counts
        # are visible (otherwise the bar has nothing real to count).
        self._compiler.on_progress = self.on_progress

        blind_start = time.monotonic()
        wiki = self._blind_compile(chunks, trees=trees, pinned_facts=cumulative_pinned or None)
        self._report(num_sources, num_sources, "blind compilation done")
        logger.info(
            "WiCER: blind compile done in %.1fs (%d nodes, %d edges)",
            time.monotonic() - blind_start,
            len(wiki.nodes),
            len(wiki.edges),
        )
        iteration_info: dict[str, Any] = {
            "wicer_iterations": 0,
            "wicer_refinements": 0,
            "wicer_probes_generated": 0,
            "wicer_probes_passed": 0,
            "wicer_pinned_facts": len(cumulative_pinned),
        }

        for attempt in range(self.max_iterations):
            iter_start = time.monotonic()

            self._report(num_sources, num_sources, f"iter {attempt + 1}/{self.max_iterations} — generating probes")
            probes = self._generate_probes(wiki, chunks)
            if not probes:
                logger.info("WiCER: no probes generated, ending loop at iteration %d (%.1fs)", attempt + 1, time.monotonic() - iter_start)
                break
            iteration_info["wicer_probes_generated"] += len(probes)

            self._report(num_sources, num_sources, f"iter {attempt + 1}/{self.max_iterations} — evaluating {len(probes)} probes")
            eval_results = self._evaluate(wiki, probes)
            failures = [r for r in eval_results if not r.get("correct", False)]
            passed = len(eval_results) - len(failures)
            iteration_info["wicer_probes_passed"] += passed

            if not failures:
                logger.info(
                    "WiCER: iter %d/%d done in %.1fs — %d probes, all passed → converged",
                    attempt + 1,
                    self.max_iterations,
                    time.monotonic() - iter_start,
                    len(probes),
                )
                iteration_info["wicer_iterations"] = attempt + 1
                break

            self._report(num_sources, num_sources, f"iter {attempt + 1}/{self.max_iterations} — diagnosing {len(failures)} failures")
            new_facts = self._diagnose(failures)
            if not new_facts:
                logger.info(
                    "WiCER: iter %d/%d done in %.1fs — %d probes, %d passed, %d failed, no new facts",
                    attempt + 1,
                    self.max_iterations,
                    time.monotonic() - iter_start,
                    len(probes),
                    passed,
                    len(failures),
                )
                iteration_info["wicer_iterations"] = attempt + 1
                break

            # F_cumulative ← F_cumulative ∪ D_facts (dedup, preserve order).
            before = len(cumulative_pinned)
            cumulative_pinned = list(dict.fromkeys(cumulative_pinned + [f.strip() for f in new_facts if f and f.strip()]))
            added = len(cumulative_pinned) - before
            if added == 0:
                logger.info(
                    "WiCER: iter %d/%d done in %.1fs — all %d fact(s) already pinned, ending loop",
                    attempt + 1,
                    self.max_iterations,
                    time.monotonic() - iter_start,
                    len(cumulative_pinned),
                )
                iteration_info["wicer_iterations"] = attempt + 1
                break

            recompile_start = time.monotonic()
            self._report(num_sources, num_sources, f"iter {attempt + 1}/{self.max_iterations} — recompiling with {len(cumulative_pinned)} pinned fact(s)")
            wiki = self._blind_compile(chunks, trees=trees, pinned_facts=cumulative_pinned)
            recompile_elapsed = time.monotonic() - recompile_start
            iteration_info["wicer_refinements"] += 1
            iteration_info["wicer_iterations"] = attempt + 1
            iteration_info["wicer_pinned_facts"] = len(cumulative_pinned)

            logger.info(
                "WiCER: iter %d/%d done in %.1fs — probes=%d, passed=%d, failed=%d, new_facts=%d, recompile=%.1fs, pinned=%d",
                attempt + 1,
                self.max_iterations,
                time.monotonic() - iter_start,
                len(probes),
                passed,
                len(failures),
                added,
                recompile_elapsed,
                len(cumulative_pinned),
            )
        else:
            iteration_info["wicer_iterations"] = self.max_iterations

        total_elapsed = time.monotonic() - compile_start
        wiki.metadata.update(iteration_info)

        if self._has_llm:
            # Aggregate usage from all agents (may not have been created)
            total_calls = 0
            total_prompt = 0
            total_completion = 0
            for attr in ("_probe_agent", "_eval_agent", "_diagnose_agent"):
                agent = getattr(self, attr, None)
                if agent is not None and agent.usage is not None:
                    total_calls += agent.usage.call_count
                    total_prompt += agent.usage.prompt_tokens
                    total_completion += agent.usage.completion_tokens
            logger.info(
                "WiCER: done — %d iteration(s), %d refinement(s), %.1fs total, %d agent LLM call(s), %d prompt + %d completion tokens",
                iteration_info["wicer_iterations"],
                iteration_info["wicer_refinements"],
                total_elapsed,
                total_calls,
                total_prompt,
                total_completion,
            )
            wiki.metadata["wicer_llm_calls"] = total_calls
            wiki.metadata["wicer_llm_prompt_tokens"] = total_prompt
            wiki.metadata["wicer_llm_completion_tokens"] = total_completion
        else:
            logger.info(
                "WiCER: done — %d iteration(s), %d refinement(s), %.1fs total",
                iteration_info["wicer_iterations"],
                iteration_info["wicer_refinements"],
                total_elapsed,
            )
        wiki.metadata["compile_duration_s"] = round(total_elapsed, 1)
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

        if not self._has_llm:
            logger.debug("WiCER has no LLM client — skipping probe generation")
            return []

        try:
            agent = WicerProbeAgent(
                litellm_model=self._config.llm.default_model,
                max_tokens=self._config.budget.verify_output_max,
                skills=self._config.skills,
                retries=self._config.llm.output_retries,
                reasoning_effort=self._config.llm.reasoning_effort,
            )
            output = agent.run(WicerProbeDeps(text=combined, source_id="wicer-probe"))
            return [p.model_dump() for p in output.probes]
        except Exception as exc:
            logger.warning("WiCER: probe generation failed: %s", exc, exc_info=True)
            return []

    def _evaluate(self, wiki: CompiledWiki, probes: list[dict]) -> list[dict]:
        """Step B: evaluate wiki against probes using LLM-as-judge.

        Wiki content is processed in segments of ``self._eval_input_chars`` chars
        so the full wiki is checked, not just the first few kilobytes.
        A probe is marked correct if it passes in ANY segment.
        """
        wiki_text = "\n\n".join(n.content for n in wiki.nodes if n.content)
        if not wiki_text:
            return [{"question": p.get("question", ""), "expected_answer": p.get("answer", ""), "found_answer": "", "correct": False} for p in probes]

        if not self._has_llm:
            logger.debug("WiCER has no LLM client — skipping evaluation")
            return [{"question": p.get("question", ""), "expected_answer": p.get("answer", ""), "found_answer": "", "correct": False} for p in probes]

        # Track probe correctness across segments — a probe passing in any
        # segment is considered passed.
        probe_results: dict[int, dict] = {}

        for seg_start in range(0, len(wiki_text), self._eval_input_chars):
            segment = wiki_text[seg_start : seg_start + self._eval_input_chars]

            try:
                agent = WicerEvaluateAgent(
                    litellm_model=self._config.llm.default_model,
                    max_tokens=self._config.budget.verify_output_max,
                    skills=self._config.skills,
                    retries=self._config.llm.output_retries,
                    reasoning_effort=self._config.llm.reasoning_effort,
                )
                output = agent.run(
                    WicerEvaluateDeps(
                        wiki_text=segment,
                        probes_json=encode_for_llm(probes, self._config.llm.data_format),
                        source_id="wicer-eval",
                    )
                )
                for i, r in enumerate(output.results):
                    if i >= len(probes):
                        break
                    rd = r.model_dump()
                    if rd.get("correct", False) or i not in probe_results:
                        probe_results[i] = rd
            except Exception as exc:
                logger.warning("WiCER: evaluation failed: %s", exc, exc_info=True)

        if probe_results:
            return [probe_results.get(i, {"question": p.get("question", ""), "correct": False}) for i, p in enumerate(probes)]

        logger.warning("WiCER: evaluation returned no results — marking all probes as unevaluated")
        return [{"question": p.get("question", ""), "correct": False} for p in probes]

    def _diagnose(self, failures: list[dict]) -> list[str]:
        """Step C: extract critical facts from failures using LLM."""
        if not failures:
            return []

        if not self._has_llm:
            logger.debug("WiCER has no LLM client — skipping diagnosis")
            return [f.get("expected_answer", "") for f in failures if f.get("expected_answer")]

        try:
            agent = WicerDiagnoseAgent(
                litellm_model=self._config.llm.default_model,
                max_tokens=self._config.budget.verify_output_max,
                skills=self._config.skills,
                retries=self._config.llm.output_retries,
                reasoning_effort=self._config.llm.reasoning_effort,
            )
            output = agent.run(
                WicerDiagnoseDeps(
                    failures_json=encode_for_llm(failures, self._config.llm.data_format),
                    source_id="wicer-diagnose",
                )
            )
            return output.critical_facts
        except Exception as exc:
            logger.warning("WiCER: diagnosis failed: %s", exc, exc_info=True)
            return [f.get("expected_answer", "") for f in failures if f.get("expected_answer")]


__all__ = ["WiCERCompiler"]
