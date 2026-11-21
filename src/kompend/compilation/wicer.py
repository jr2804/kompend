"""WiCER: Compile-Evaluate-Refine loop for improved compilation quality.

Implements the iterative compilation loop described in
arXiv:2605.07068v1: compile (blind) → generate probes → evaluate →
diagnose → refine → repeat.

LLM calls are routed through the shared :class:`kompend.llm.LLMClient`
which provides typed errors, failover and observability. When the
client returns ``None`` (all models in the chain failed), the
compiler falls back to deterministic heuristics so the pipeline never
crashes on a single bad provider.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import count
from typing import Any, Final

from kompend.agents.deps import WicerDiagnoseDeps, WicerEvaluateDeps, WicerProbeDeps
from kompend.agents.wicer import WicerDiagnoseAgent, WicerEvaluateAgent, WicerProbeAgent
from kompend.compilation.base import BaseCompiler
from kompend.compilation.graph import GraphCompiler
from kompend.compilation.two_phase import tokens_to_chars
from kompend.core.config import ContextManagerPreset
from kompend.core.models import ChunkTree, CompiledWiki, MetadataField, ResolvedChunk, WikiField
from kompend.core.pipeline.config import PipelineConfig
from kompend.llm.context_window import ContextWindow, answer_reserve_tokens, discover_context_window
from kompend.llm.format import encode_payload
from kompend.llm.request_budget import RequestBudget
from kompend.llm.throughput import timeout_for
from kompend.llm.usage import UsageTracker
from kompend.logging import get_logger

# Fraction of the available context window (after reserves) allocated to
# WiCER probe input text.  The chars-per-token heuristic (2.5) is too
# optimistic for models like Gemma (actual ~2.0), causing the batch to
# overshoot the real token limit.  A utilization factor leaves headroom
# for tokenizer density variation across model families.
_WICER_CONTEXT_UTILIZATION: Final[float] = 0.50

# Absolute token reserve carved out of the context window for the WiCER probe
# and evaluation prompts (system prompt + framing overhead).  Sized for the
# compact JSON-emitting prompts used by ``probe_questions.md.j2`` and
# ``probe_eval.md.j2``; both are short fixed envelopes.
_WICER_PROMPT_RESERVE_TOKENS: Final[int] = 1024

# Requests one refinement work unit costs: each probe batch and each eval
# segment is covered by exactly one single-shot agent run (the diagnosis is
# one more run on top). The count shapes the derived pass budget; the clock
# comes from the measured throughput fit, never from this number.
_WICER_REQUESTS_PER_UNIT: Final[int] = 1

# Headroom multiplier on the derived refinement-pass budget. The per-turn
# price already carries the throughput fit's own safety factor, so this only
# covers pass-level spread (batches are not all average-sized, and the fit
# is still settling in early iterations). A fraction that shapes the
# adaptive bound, never an absolute number.
_WICER_PASS_HEADROOM: Final[float] = 1.5


logger = get_logger(__name__)


@dataclass(frozen=True)
class WicerPassBudget:
    """Refinement budget for one loop pass, derived — never configured.

    ``pass_cap_s`` is the pass's own price: the planned agent requests (one
    per probe batch and eval segment, plus the diagnosis) times the
    fit-priced seconds per request, with headroom for pass-level spread,
    plus the measured blind-compile time as the recompile allowance (a
    refinement re-runs the same base compiler over the same corpus).
    ``effective_s`` applies the configured ``wicer_max_seconds`` ceiling
    (``0`` disables the ceiling, leaving the derived budget unbounded).
    """

    probe_batches: int
    eval_segments: int
    requests_per_unit: int
    planned_requests: int
    turn_s: float
    priced_by: str
    fit_samples: int
    blind_s: float
    pass_cap_s: float
    configured_s: float
    effective_s: float

    def report(self) -> str:
        """One-line derivation of the budget, for log lines."""
        return (
            f"{self.probe_batches} probe batches + {self.eval_segments} eval segments + 1 diagnose "
            f"= {self.planned_requests} requests x {self.turn_s:.1f}s/req "
            f"({self.priced_by}, {self.fit_samples} fit samples) x headroom {_WICER_PASS_HEADROOM} "
            f"+ blind {self.blind_s:.1f}s recompile -> derived {self.pass_cap_s:.0f}s, "
            f"configured cap {self.configured_s:.0f}s, effective {self.effective_s:.0f}s"
        )


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
        :class:`~kompend.compilation.two_phase.TwoPhaseCompiler` here).
    max_iterations:
        Maximum number of refinement rounds. ``0`` (the default) means
        "until the loop converges" — the loop stops on its own when it
        finds no new facts, when all probes pass, or when the wall-clock
        budget (``wicer_max_seconds``) is spent. A positive value is an
        explicit user-set limit.
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
            model=config.llm.default_model or "",
            min_content_length=config.budget.min_content_length,
        )
        self.max_iterations = config.budget.wicer_max_iterations
        self._max_seconds = config.budget.wicer_max_seconds
        self._has_llm = bool(config.llm.default_model)
        # One tracker for the three WiCER agents: every agent of the loop
        # records into it, so the done line reports the provider's numbers
        # instead of a sum of per-agent counters.
        self._usage_tracker = UsageTracker(throughput_window_tokens=config.llm.throughput_window_tokens)

        # Input batch sizes adapt to the model's discovered context window.
        # The room left for the answer comes from the window itself
        # (``ContextWindow.answer_reserve``) — kompend sends no ``max_tokens``.
        cw = (
            context_window
            or discover_context_window(
                config.llm.default_model,
                llm_config=config.llm,
            )
            if config.llm.default_model
            else None
        )
        cpt = config.llm.default_chars_per_token
        if cw is not None:
            self._context_window: ContextWindow | None = cw
            budget = RequestBudget(
                window=cw,
                reserved={"prompt": _WICER_PROMPT_RESERVE_TOKENS},
                utilization=_WICER_CONTEXT_UTILIZATION,
            )
            self._probe_input_chars = budget.content_chars(cpt)
            self._eval_input_chars = self._probe_input_chars
            logger.info("WiCER: probe budget (%s) -> %d chars per probe", budget.report(), self._probe_input_chars)
        else:
            self._context_window = None
            fallback_tokens = answer_reserve_tokens(config.llm.fallback_max_input)
            self._probe_input_chars = tokens_to_chars(fallback_tokens, chars_per_token=cpt)
            self._eval_input_chars = self._probe_input_chars
            logger.warning(
                "WiCER: no context window discovered — probe budget falls back to %d chars (answer reserve only)",
                self._probe_input_chars,
            )

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
        blind_elapsed = time.monotonic() - blind_start
        self._report(num_sources, num_sources, "blind compilation done")
        logger.info(
            "WiCER: blind compile done in %.1fs (%d nodes, %d edges)",
            blind_elapsed,
            len(wiki.nodes),
            len(wiki.edges),
        )

        # The refinement clock starts here, not at compile() entry. The blind
        # compile is the base compiler's own work; charging it to the
        # refinement budget made iteration 1 unreachable for every model
        # slower than the budget (run bc35f97e: 20199.5s of blind compile
        # against an 1800s budget, so the loop stopped before its first
        # probe).
        refine_start = time.monotonic()
        iteration_info: dict[str, Any] = {
            "wicer_iterations": 0,
            "wicer_refinements": 0,
            "wicer_probes_generated": 0,
            "wicer_probes_passed": 0,
            "wicer_pinned_facts": len(cumulative_pinned),
            "wicer_pass_cap_s": 0.0,
            "wicer_pass_derived_s": 0.0,
            "wicer_pass_units": 0,
            "wicer_pass_turn_s": 0.0,
        }

        iterations: Iterable[int] = range(1, self.max_iterations + 1) if self.max_iterations > 0 else count(1)
        for attempt in iterations:
            iter_start = time.monotonic()

            budget = self._derive_pass_budget(chunks, wiki, blind_s=blind_elapsed)
            logger.info("WiCER: pass budget — %s", budget.report())
            iteration_info["wicer_pass_cap_s"] = round(budget.effective_s, 1)
            iteration_info["wicer_pass_derived_s"] = round(budget.pass_cap_s, 1)
            iteration_info["wicer_pass_units"] = budget.probe_batches + budget.eval_segments
            iteration_info["wicer_pass_turn_s"] = round(budget.turn_s, 2)

            if (time.monotonic() - refine_start) > budget.effective_s:
                logger.warning(
                    "WiCER: refinement budget spent after %d iteration(s) — stopping the loop (%s; the blind compile is not charged to this budget)",
                    attempt - 1,
                    budget.report(),
                )
                iteration_info["wicer_iterations"] = attempt - 1
                break

            label = f"{attempt}/{self.max_iterations}" if self.max_iterations > 0 else str(attempt)

            self._report(num_sources, num_sources, f"iter {label} — generating probes")
            probes = self._generate_probes(wiki, chunks)
            if not probes:
                logger.info("WiCER: no probes generated, ending loop at iteration %d (%.1fs)", attempt, time.monotonic() - iter_start)
                break
            iteration_info["wicer_probes_generated"] += len(probes)

            self._report(num_sources, num_sources, f"iter {label} — evaluating {len(probes)} probes")
            eval_results = self._evaluate(wiki, probes)
            failures = [r for r in eval_results if not r.get("correct", False)]
            passed = len(eval_results) - len(failures)
            iteration_info["wicer_probes_passed"] += passed

            if not failures:
                logger.info(
                    "WiCER: iter %s done in %.1fs — %d probes, all passed → converged",
                    label,
                    time.monotonic() - iter_start,
                    len(probes),
                )
                iteration_info["wicer_iterations"] = attempt
                break

            self._report(num_sources, num_sources, f"iter {label} — diagnosing {len(failures)} failures")
            new_facts = self._diagnose(failures)
            if not new_facts:
                logger.info(
                    "WiCER: iter %s done in %.1fs — %d probes, %d passed, %d failed, no new facts",
                    label,
                    time.monotonic() - iter_start,
                    len(probes),
                    passed,
                    len(failures),
                )
                iteration_info["wicer_iterations"] = attempt
                break

            # F_cumulative ← F_cumulative ∪ D_facts (dedup, preserve order).
            before = len(cumulative_pinned)
            cumulative_pinned = list(dict.fromkeys(cumulative_pinned + [f.strip() for f in new_facts if f and f.strip()]))
            added = len(cumulative_pinned) - before
            if added == 0:
                logger.info(
                    "WiCER: iter %s done in %.1fs — all %d fact(s) already pinned, ending loop",
                    label,
                    time.monotonic() - iter_start,
                    len(cumulative_pinned),
                )
                iteration_info["wicer_iterations"] = attempt
                break

            recompile_start = time.monotonic()
            self._report(num_sources, num_sources, f"iter {label} — recompiling with {len(cumulative_pinned)} pinned fact(s)")
            wiki = self._blind_compile(chunks, trees=trees, pinned_facts=cumulative_pinned)
            recompile_elapsed = time.monotonic() - recompile_start
            iteration_info["wicer_refinements"] += 1
            iteration_info["wicer_iterations"] = attempt
            iteration_info["wicer_pinned_facts"] = len(cumulative_pinned)

            logger.info(
                "WiCER: iter %s done in %.1fs — probes=%d, passed=%d, failed=%d, new_facts=%d, recompile=%.1fs, pinned=%d",
                label,
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
            # One tracker for the three WiCER agents: the totals are what the
            # provider reported, not a sum of per-agent counters.
            tracker = self._usage_tracker
            logger.info(
                "WiCER: done — %d iteration(s), %d refinement(s), %.1fs total, %d agent LLM call(s), %d prompt + %d completion tokens",
                iteration_info["wicer_iterations"],
                iteration_info["wicer_refinements"],
                total_elapsed,
                tracker.call_count,
                tracker.prompt_tokens,
                tracker.completion_tokens,
            )
            wiki.metadata["wicer_llm_calls"] = tracker.call_count
            wiki.metadata["wicer_llm_prompt_tokens"] = tracker.prompt_tokens
            wiki.metadata["wicer_llm_completion_tokens"] = tracker.completion_tokens
        else:
            logger.info(
                "WiCER: done — %d iteration(s), %d refinement(s), %.1fs total",
                iteration_info["wicer_iterations"],
                iteration_info["wicer_refinements"],
                total_elapsed,
            )
        wiki.metadata["compile_duration_s"] = round(total_elapsed, 1)
        return wiki

    def _derive_pass_budget(self, chunks: list[ResolvedChunk], wiki: CompiledWiki, *, blind_s: float) -> WicerPassBudget:
        """Price one refinement pass from the corpus and the measured fit.

        The corpus units are the batches and segments the pass will
        actually cover (same helpers the steps use, so the plan and the
        work count identically); the per-unit allowance is one agent run
        per unit plus the diagnosis; the clock per request is the
        throughput fit once the shared tracker holds enough samples,
        else the request ceiling — the first calls are how the fit is
        learned, so they must not be starved by it. The measured
        blind-compile time is the recompile allowance: a refinement
        re-runs the same base compiler over the same corpus.
        """
        probe_batches = len(self._probe_batches(chunks))
        eval_segments = len(self._eval_segments(wiki))
        planned = (probe_batches + eval_segments) * _WICER_REQUESTS_PER_UNIT + 1
        ceiling = self._config.llm.request_ceiling_s
        throughput = self._usage_tracker.throughput()
        if throughput is None:
            turn_s = ceiling
            priced_by = "ceiling"
            fit_samples = 0
        else:
            turn_s = timeout_for(
                throughput,
                self._usage_tracker.average_prompt_tokens(),
                self._usage_tracker.average_completion_tokens(),
                ceiling_s=ceiling,
            )
            priced_by = "fit"
            fit_samples = throughput.samples
        pass_cap_s = planned * turn_s * _WICER_PASS_HEADROOM + blind_s
        configured = self._max_seconds
        effective = min(pass_cap_s, configured) if configured > 0 else pass_cap_s
        return WicerPassBudget(
            probe_batches=probe_batches,
            eval_segments=eval_segments,
            requests_per_unit=_WICER_REQUESTS_PER_UNIT,
            planned_requests=planned,
            turn_s=turn_s,
            priced_by=priced_by,
            fit_samples=fit_samples,
            blind_s=blind_s,
            pass_cap_s=pass_cap_s,
            configured_s=configured,
            effective_s=effective,
        )

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

    def _probe_batches(self, chunks: list[ResolvedChunk]) -> list[str]:
        """Group chunk texts into budget-sized probe batches.

        Single definition of the probe batching: the budget derivation
        and probe generation count the same batches, so the plan and the
        work cannot drift apart. Every non-empty chunk lands in exactly
        one batch — no content is dropped.
        """
        batches: list[str] = []
        batch_parts: list[str] = []
        batch_len = 0
        for rc in chunks:
            if not rc.chunk.text:
                continue
            if batch_len + len(rc.chunk.text) > self._probe_input_chars and batch_parts:
                batches.append("\n\n".join(batch_parts))
                batch_parts = []
                batch_len = 0
            batch_parts.append(rc.chunk.text)
            batch_len += len(rc.chunk.text)
        if batch_parts:
            batches.append("\n\n".join(batch_parts))
        return batches

    def _generate_probes(self, wiki: CompiledWiki, chunks: list[ResolvedChunk]) -> list[dict]:
        """Step B: generate probe Q&A pairs from source chunks.

        All chunks are processed in batches of ``self._probe_input_chars`` chars
        so no source content is dropped.
        """
        all_probes: list[dict] = []
        seen_questions: set[str] = set()
        for batch in self._probe_batches(chunks):
            probes = self._probe_batch(batch)
            for p in probes:
                q = p.get("question", "")
                if q not in seen_questions:
                    all_probes.append(p)
                    seen_questions.add(q)
        return all_probes

    def _loop_context_manager(self) -> ContextManagerPreset | bool:
        """Context manager for the probe loop.

        The loop keeps earlier probes in context, so the "aggressive"
        preset (compress at 70%, keep 5 messages) is the safe default.
        Every other configured value (``compile.llm.context_manager``) is
        honored as-is.
        """
        configured = self._config.llm.context_manager
        if configured == ContextManagerPreset.DEFAULT:
            return ContextManagerPreset.AGGRESSIVE
        return configured

    def _probe_batch(self, combined: str) -> list[dict]:
        """Generate probes from a single batch of combined text."""
        if not combined:
            return []

        if not self._has_llm:
            logger.debug("WiCER has no LLM client — skipping probe generation")
            return []

        try:
            agent = WicerProbeAgent(
                model_string=self._config.llm.default_model or "",
                skills=self._config.skills,
                usage_tracker=self._usage_tracker,
                timeout=self._config.llm.request_ceiling_s,
                reasoning_effort=self._config.llm.reasoning_effort,
                # The probe loop keeps earlier probes in context, so the
                # "aggressive" preset (compress at 70%, keep 5 messages) is
                # the safe default; every other configured value is used
                # as-is.
                context_manager=self._loop_context_manager(),
                context_window=self._context_window.max_input if self._context_window else None,
                rlm=self._config.llm.rlm,
                llm=self._config.llm,
            )
            output = agent.run(WicerProbeDeps(text=combined, source_id="wicer-probe"))
            return [p.model_dump() for p in output.probes]
        except Exception as exc:
            logger.warning("WiCER: probe generation failed: %s", exc, exc_info=True)
            return []

    def _eval_segments(self, wiki: CompiledWiki) -> list[str]:
        """Split wiki text into budget-sized evaluation segments.

        Single definition of the eval segmentation: the budget derivation
        and evaluation cover the same segments. An empty wiki yields no
        segments — the caller marks every probe unevaluated.
        """
        wiki_text = "\n\n".join(n.content for n in wiki.nodes if n.content)
        if not wiki_text:
            return []
        return [wiki_text[seg_start : seg_start + self._eval_input_chars] for seg_start in range(0, len(wiki_text), self._eval_input_chars)]

    def _evaluate(self, wiki: CompiledWiki, probes: list[dict]) -> list[dict]:
        """Step B: evaluate wiki against probes using LLM-as-judge.

        Wiki content is processed in segments of ``self._eval_input_chars`` chars
        so the full wiki is checked, not just the first few kilobytes.
        A probe is marked correct if it passes in ANY segment.
        """
        segments = self._eval_segments(wiki)
        if not segments:
            return [{"question": p.get("question", ""), "expected_answer": p.get("answer", ""), "found_answer": "", "correct": False} for p in probes]

        if not self._has_llm:
            logger.debug("WiCER has no LLM client — skipping evaluation")
            return [{"question": p.get("question", ""), "expected_answer": p.get("answer", ""), "found_answer": "", "correct": False} for p in probes]

        # Track probe correctness across segments — a probe passing in any
        # segment is considered passed.
        probe_results: dict[int, dict] = {}

        for segment in segments:
            try:
                agent = WicerEvaluateAgent(
                    model_string=self._config.llm.default_model or "",
                    skills=self._config.skills,
                    usage_tracker=self._usage_tracker,
                    timeout=self._config.llm.request_ceiling_s,
                    reasoning_effort=self._config.llm.reasoning_effort,
                    context_manager=self._config.llm.context_manager,
                    context_window=self._context_window.max_input if self._context_window else None,
                    llm=self._config.llm,
                )
                output = agent.run(
                    WicerEvaluateDeps(
                        wiki_text=segment,
                        probes_json=encode_payload(probes, self._config.llm.data_format),
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
                model_string=self._config.llm.default_model or "",
                skills=self._config.skills,
                usage_tracker=self._usage_tracker,
                timeout=self._config.llm.request_ceiling_s,
                reasoning_effort=self._config.llm.reasoning_effort,
                context_manager=self._config.llm.context_manager,
                context_window=self._context_window.max_input if self._context_window else None,
                llm=self._config.llm,
            )
            output = agent.run(
                WicerDiagnoseDeps(
                    failures_json=encode_payload(failures, self._config.llm.data_format),
                    source_id="wicer-diagnose",
                )
            )
            return output.critical_facts
        except Exception as exc:
            logger.warning("WiCER: diagnosis failed: %s", exc, exc_info=True)
            return [f.get("expected_answer", "") for f in failures if f.get("expected_answer")]


__all__ = ["WiCERCompiler"]
