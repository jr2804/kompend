"""Segment planning and adaptation for the generation pass.

The plan phase sizes sources against the context window and decides how a
failed unit shrinks; the pass loop in :mod:`.flow` consumes these units.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Final, NamedTuple

from kompend.agents.decision import decide_error_action, decide_unit_adaptation
from kompend.agents.failure import AgentFailureKind
from kompend.compilation.adaptation import AdaptationSignal, ShrinkOutcome, plan_adaptation, split_unit
from kompend.compilation.concept_broker import ConceptBroker
from kompend.compilation.two_phase import TwoPhaseRuntime, segment_sources, tokens_to_chars
from kompend.core.models import ResolvedChunk
from kompend.core.models.enums import ExtractionField
from kompend.core.paths import safe_stem
from kompend.core.text.matching import best_match, cosine
from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.decision import ErrorAction, ErrorDecision
from kompend.llm.request_budget import RequestBudget
from kompend.logging import get_logger
from kompend.rag.embedding_pydantic import make_provider

logger = get_logger(__name__)


#: rapidfuzz-scale (0–100) threshold for "does a produced page title cover
#: this concept" (T2 string tier).
_COVER_SIMILARITY_THRESHOLD = 85.0


#: Failure kinds that end one pass but not the segment. An empty answer (or
#: a request the provider refused for size) says nothing about the next
#: pass: that pass carries different material — the broker's remaining
#: concepts and a fresh prompt — so the segment keeps asking instead of
#: reporting one empty pass as a failed segment.
_PASS_CONTINUATION_KINDS: Final[frozenset[AgentFailureKind]] = frozenset({AgentFailureKind.EMPTY_ANSWER, AgentFailureKind.CONTEXT_OVERFLOW})


#: Cosine threshold for the T3 semantic tier — catches paraphrases and
#: punctuation variants the string tier misses (e.g. "ITU-T P.10/G.100" vs
#: "P.10 G.100 Recommendation"). Conservative: a false *credit* silently
#: drops a concept's page, so the bar sits above lint's topic-vs-chunk 0.55.
_COVER_SEMANTIC_THRESHOLD = 0.65


#: Failure kinds whose remedy is a judgment call: Loop 2 may be consulted
#: when the deterministic policy declines them. Transport failures and
#: provider overflows are arithmetic — the model is never asked about those.
_AMBIGUOUS_KINDS: Final[frozenset[AgentFailureKind]] = frozenset({AgentFailureKind.SCHEMA_MISMATCH, AgentFailureKind.OTHER, AgentFailureKind.EMPTY_ANSWER})


class _CoverageCredit:
    """Decides whether produced page titles cover fetched concepts.

    T2: rapidfuzz similarity (:func:`kompend.core.text.matching.best_match`).
    T3: embedding cosine via the configured embedding model, embedded
    incrementally and cached per string so repeated passes stay cheap.
    When no embedding provider is available, T2 alone bounds the work —
    callers degrade gracefully.
    """

    def __init__(self, embedding_model: str) -> None:
        self._provider = None
        if embedding_model and embedding_model != "hash":
            try:
                self._provider = make_provider(embedding_model)
            except Exception as exc:
                logger.debug("Coverage semantic tier disabled (no provider): %s", exc)
        self._vectors: dict[str, list[float]] = {}

    def covered(self, concept_name: str, titles: set[str]) -> bool:
        """True when some title in *titles* covers *concept_name*."""
        if not titles:
            return False
        if best_match(concept_name, titles, threshold=_COVER_SIMILARITY_THRESHOLD) is not None:
            return True
        if self._provider is None:
            return False
        cvec = self._vec(concept_name)
        if cvec is None:
            return False
        for title in titles:
            tvec = self._vec(title)
            if tvec is not None and cosine(cvec, tvec) >= _COVER_SEMANTIC_THRESHOLD:
                return True
        return False

    def _vec(self, text: str) -> list[float] | None:
        if self._provider is None:
            return None
        cached = self._vectors.get(text)
        if cached is not None:
            return cached
        try:
            vec = self._provider.embed(text)
        except Exception as exc:
            logger.debug("Coverage embed failed for %r: %s", text, exc)
            self._provider = None  # one failure disables the tier
            return None
        self._vectors[text] = vec
        return vec


class _SegmentUnit(NamedTuple):
    """One generation work unit: Loop 1 splits a segment into smaller ones."""

    label: str
    seg_id: str
    segment: list[tuple[str, str]]
    run_prefix: str | None
    seg_idx: int
    depth: int


def prepare_source_segments(
    compiler: TwoPhaseRuntime,
    chunks: list[ResolvedChunk],
    *,
    budget: RequestBudget | None = None,
) -> list[list[tuple[str, str]]]:
    """Build source segments sized to the context window.

    *budget* is the fixed per-request conversation cost that is NOT the
    segment content: system prompt + rendered user prompt (template + the
    known_titles head + pinned_facts) + tool-call framing + the
    fetch_concepts result + the safety margin. The agentic flow passes the measured budget
    so the segment budget = window - reserved - answer reserve, guaranteeing
    the full request (segment + output) fits.
    Content is preserved: oversized sources are split across more
    segments by :func:`segment_sources` — nothing is cropped.
    """
    by_source: dict[str, str] = defaultdict(str)
    for rc in chunks:
        label = safe_stem(rc.chunk.source_file or "unknown")
        if rc.chunk.text:
            by_source[label] += rc.chunk.text + "\n\n"
    sources: list[tuple[str, str]] = list(by_source.items())

    cw = compiler._context_window
    # The rendered content block is ``"\n\n".join(f"### {label}\n{content}")``
    # — each source adds ``### {label}\n`` plus the ``\n\n`` separator. The
    # request-fit guard measures the *rendered* block, so the sizing must
    # reserve this label/separator overhead or the guard fires on segments
    # that are exactly at the raw-content budget (observed: 27-token
    # overshoot on a 39K-token segment). Conservative: reserve the total
    # across all sources (segments share it, so this over-reserves slightly).
    cpt = compiler._config.llm.default_chars_per_token
    label_overhead_tokens = int(sum(len(label) + 7 for label, _ in sources) / cpt)
    # Sizing-feasibility invariant, checked at planning time (before any
    # LLM call): the smallest segment the sizer can emit is
    # min_segment_tokens (the floor in tokens_to_chars below), and its
    # rendered block adds label_overhead on top. If even that cannot fit
    # beside the reservation + answer reserve, every segment would trip
    # the per-segment request-fit guard and the run would abort after all
    # upstream work — fail here instead, naming the components.
    min_floor_tokens = compiler._config.llm.min_segment_tokens
    if budget is None:
        # Legacy template path: only the system template is pre-known; the
        # content block IS the prompt, so no other reservation applies.
        budget = RequestBudget(
            window=cw,
            reserved={"system": compiler._gen_system_tokens},
        )
    if budget.available_tokens - label_overhead_tokens < min_floor_tokens:
        raise LLMOutputError(
            f"Segment sizing impossible: {budget.report()} + source labels ({label_overhead_tokens})"
            + f" leaves {budget.available_tokens - label_overhead_tokens} tokens, below the minimum segment "
            + f"({min_floor_tokens}). Reduce the reserved payloads (known_titles is served in windows; check "
            + "pinned_facts/categories) or raise compile.llm.context_window."
        )
    # The rendered block carries the label overhead on top of the segment
    # content, so it is reserved before the content budget is derived.
    content_budget = max(min_floor_tokens, budget.content_tokens - label_overhead_tokens)
    segment_char_size = tokens_to_chars(
        content_budget,
        chars_per_token=cpt,
    )
    logger.info(
        "TwoPhase: segment_char_size=%d chars (%s, labels=%d, answer_reserve=%d)",
        segment_char_size,
        budget.report(),
        label_overhead_tokens,
        cw.answer_reserve,
    )
    source_segments = segment_sources(sources, segment_char_size)
    logger.info(
        "TwoPhase: %d source(s) split into %d segment(s) (total content %d chars)",
        len(sources),
        len(source_segments),
        sum(len(c) for _, c in sources),
    )
    return source_segments


def _shrink_and_requeue(
    broker: ConceptBroker,
    coverage: _CoverageCredit,
    compiler: TwoPhaseRuntime,
    pending: deque[_SegmentUnit],
    unit: _SegmentUnit,
    *,
    start_remaining: int,
    seg_titles: set[str],
    max_splits: int,
    kind: AgentFailureKind,
    err: str,
) -> ShrinkOutcome:
    """Shrink a failed unit: Loop 1 decides, Loop 2 gets the ambiguous rest.

    SPLIT means the halves were enqueued — the caller leaves its pass
    loop. DECLINED keeps today's path. SKIP/ABORT carry the decision
    model's verdict for a failure the deterministic policy declined.
    The parent pool is drained on a split, so the tail sees no uncovered
    remainder; the halves carry their own broker pools and transcript
    prefixes.
    """
    # Fetched-but-unpaged concepts are stranded in the parent pool once it
    # drains — they move to the halves with the unfetched remainder, or the
    # split silently drops work.
    unfinished = [c for c in broker.consumed(unit.seg_id) if not coverage.covered(str(c.get(ExtractionField.NAME, "")), seg_titles)]
    unfinished.extend(broker.remaining_concepts(unit.seg_id))
    tracker = compiler._usage_tracker
    signal = AdaptationSignal(
        seg_id=unit.seg_id,
        kind=kind,
        adaptations_used=unit.depth,
        concepts_assigned=start_remaining,
        concepts_remaining=broker.remaining(unit.seg_id),
        fetches=len(broker.consumed(unit.seg_id)),
        context_tokens=tracker.context_tokens or None,
        context_budget=tracker.context_budget or None,
        last_error=err,
    )
    reason = ""
    plan = plan_adaptation(signal, max_adaptations=max_splits)
    if plan.split:
        reason = plan.reason
    else:
        # Loop 2: the deterministic policy declined an ambiguous failure.
        # The decision model may still earn it one more halving — or stop
        # the retries, or fail loudly. Anything else keeps today's path.
        decision = _consult_loop2(compiler, unit, signal, max_splits)
        if decision is None:
            return ShrinkOutcome.DECLINED
        if decision.action is ErrorAction.ABORT:
            return ShrinkOutcome.ABORT
        if decision.action is ErrorAction.SKIP:
            return ShrinkOutcome.SKIP
        if not decision.shrink:
            return ShrinkOutcome.DECLINED
        reason = f"{unit.seg_id}: decision model overrode the decline — halving"
    halves = split_unit(unit.segment, unfinished)
    if halves is None:
        return ShrinkOutcome.DECLINED
    (sources_a, concepts_a), (sources_b, concepts_b) = halves
    broker.drain_caller(unit.seg_id)
    for sub, sub_sources, sub_concepts in (("a", sources_a, concepts_a), ("b", sources_b, concepts_b)):
        sub_id = f"{unit.seg_id}/s{sub}"
        broker.adopt_caller(sub_id, sub_concepts)
        pending.append(
            _SegmentUnit(
                label=f"{unit.label} (split {unit.depth + 1}{sub})",
                seg_id=sub_id,
                segment=sub_sources,
                run_prefix=f"{unit.run_prefix}/sh{unit.depth + 1}-{sub}" if unit.run_prefix else None,
                seg_idx=unit.seg_idx,
                depth=unit.depth + 1,
            )
        )
    logger.warning(
        "TwoPhase: agentic %s exceeds the window (%s) — %s; re-queued as 2 sub-units (%d/%d concepts).",
        unit.label,
        err,
        reason,
        len(concepts_a),
        len(concepts_b),
    )
    return ShrinkOutcome.SPLIT


def _consult_loop2(
    compiler: TwoPhaseRuntime,
    unit: _SegmentUnit,
    signal: AdaptationSignal,
    max_splits: int,
) -> ErrorDecision | None:
    """Ask the decision model about an ambiguous failure Loop 1 declined.

    ``None`` keeps today's path: the kind is not ambiguous, the budget is
    spent, no decision model is configured, or the sidecar call failed.
    """
    if signal.kind not in _AMBIGUOUS_KINDS or signal.adaptations_used >= max_splits:
        return None
    return decide_unit_adaptation(
        compiler._config.decision,
        signal=signal,
        stage=f"agentic-generation:{unit.seg_id}",
        max_adaptations=max_splits,
    )


def _format_uncovered_concepts(
    broker: ConceptBroker,  # noqa: F821 — forward ref
    seg_id: str,
    *,
    limit: int = 10,
) -> str:
    """Build a traceable uncovered-concepts summary for warnings/exceptions.

    Each entry is rendered as ``"<name> [sources: <src1>, <src2>...]"`` so the
    user can grep for the source PDF/section that produced the concept and
    cross-check against the wiki's source tree and the step-store transcripts.

    A trailing ellipsis marks truncation when more than *limit* concepts
    remain. Empty ``sources`` lists fall back to ``"(no-source)"`` so the
    field never silently disappears (memory #149).
    """
    items = broker.remaining_concepts(seg_id)
    if not items:
        return "(none)"
    rendered: list[str] = []
    for concept in items[:limit]:
        name = str(concept.get(ExtractionField.NAME, "?")).strip() or "?"
        sources_raw = concept.get("sources") or []
        if isinstance(sources_raw, str):
            sources: list[str] = [sources_raw]
        else:
            sources = [str(s) for s in sources_raw if s]
        sources_str = ", ".join(sources) if sources else "(no-source)"
        rendered.append(f"{name} [sources: {sources_str}]")
    suffix = "…" if len(items) > limit else ""
    return ", ".join(rendered) + suffix


def _failure_action(
    compiler: TwoPhaseRuntime,
    exc: BaseException,
    *,
    stage: str,
    attempt: int,
    max_attempts: int,
    transient: bool,
) -> ErrorAction:
    """Decide what to do with a failed generation call.

    The optional decision model answers this when it is configured. Without it
    the fallback is deterministic: a transport failure is worth repeating, a
    semantic failure is not - repeating it would produce the same output.
    """
    decided = decide_error_action(
        compiler._config.decision,
        error=str(exc),
        stage=stage,
        attempt=attempt,
        max_attempts=max_attempts,
        context=f"model={compiler._model or '?'}",
    )
    if decided is not None:
        return decided
    return ErrorAction.RETRY if transient else ErrorAction.SKIP
