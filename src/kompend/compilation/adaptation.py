"""Adaptive segment sizing for agentic generation (Loop 1: deterministic shrink).

Static tuning cannot cover every vault/model combination: a segment that
passes the entry-size guard can still overflow mid-conversation when fetch
rounds accumulate. Loop 1 answers that failure deterministically — halve
the unit that overflowed and re-queue the halves — instead of adding
another reserve constant. Each split strictly reduces the unit, so the
loop terminates by construction, bounded by the per-segment adaptation
budget (`budget.segment_shrink_attempts`).

Loop 2 (decision model) consumes the same :class:`AdaptationSignal` for
failures where the remedy is a judgment call. The signal therefore carries
only compact evidence (counts, the utilization snapshot, the error text) —
never request content, which by definition does not fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kompend.agents.failure import AgentFailureKind


class ShrinkOutcome(StrEnum):
    """What the shrink loop decided for a failed unit."""

    SPLIT = "split"  #: halves enqueued; the caller leaves its pass loop.
    DECLINED = "declined"  #: no split and no model verdict; the caller's logic stands.
    SKIP = "skip"  #: the decision model says stop retrying this unit.
    ABORT = "abort"  #: the decision model says fail loudly.


@dataclass(frozen=True)
class AdaptationSignal:
    """Compact evidence about one failed generation unit.

    The contract Loop 2's decision model will consume. Raw numbers only —
    the policy, not the caller, decides what they mean.
    """

    seg_id: str
    kind: AgentFailureKind
    adaptations_used: int
    concepts_assigned: int
    concepts_remaining: int
    fetches: int
    context_tokens: int | None
    context_budget: int | None
    answer_tokens: int | None = None  #: measured answer size (model average), if a snapshot exists.
    answer_reserve: int | None = None  #: reserved answer room the answer must fit (effective window's reserve).
    last_error: str = ""

    @property
    def input_share(self) -> float | None:
        """Measured context share (tokens/budget), or ``None`` with no snapshot."""
        if not self.context_tokens or not self.context_budget:
            return None
        return self.context_tokens / self.context_budget

    @property
    def output_pressure(self) -> bool:
        """Whether the measured answer does not fit its measured reserve.

        Unknown (no snapshot) never counts as pressure: without evidence the
        policy cannot claim the answer was the ceiling.
        """
        if not self.answer_tokens or not self.answer_reserve:
            return False
        return self.answer_tokens >= self.answer_reserve

    def describe(self) -> str:
        """Render the compact failure record for the Loop-2 decision model.

        Counts and kinds only — never request content, which by definition
        does not fit the window it failed in.
        """
        window = f"{self.context_tokens}/{self.context_budget} tokens" if self.context_tokens is not None and self.context_budget else "no usage snapshot"
        answer = f"{self.answer_tokens}/{self.answer_reserve} tokens" if self.answer_tokens is not None and self.answer_reserve else "no answer snapshot"
        return (
            f"unit={self.seg_id} kind={self.kind.value} "
            f"adaptations={self.adaptations_used} "
            f"concepts_assigned={self.concepts_assigned} concepts_remaining={self.concepts_remaining} "
            f"fetches={self.fetches} window={window} answer={answer} last_error={self.last_error}"
        )


@dataclass(frozen=True)
class Adaptation:
    """What the failed unit should become."""

    split: bool
    reason: str


def plan_adaptation(signal: AdaptationSignal, *, max_adaptations: int, pressure_ratio: float) -> Adaptation:
    """Decide whether *signal* earns another, smaller attempt.

    Overflow is arithmetic, not judgment: a provider rejection, or a failure
    whose measured utilization shows the window was the ceiling, re-runs as
    two halves. *pressure_ratio* is the pipeline's own sizing ratio
    (``compile.llm.request_utilization``) — the line lives in config, not in
    a constant. Anything else (transport, genuine schema problems, an answer
    that arrived under no pressure) keeps today's path.
    """
    if signal.adaptations_used >= max_adaptations:
        return Adaptation(
            split=False,
            reason=f"{signal.seg_id}: adaptation budget spent ({signal.adaptations_used}/{max_adaptations})",
        )
    if signal.kind is AgentFailureKind.TRANSIENT:
        return Adaptation(split=False, reason=f"{signal.seg_id}: transport failure is worth repeating unchanged")
    share = signal.input_share
    pressured = (share is not None and share >= pressure_ratio) or signal.output_pressure
    if signal.kind is AgentFailureKind.CONTEXT_OVERFLOW or (
        signal.kind in (AgentFailureKind.SCHEMA_MISMATCH, AgentFailureKind.OTHER, AgentFailureKind.EMPTY_ANSWER) and pressured
    ):
        return Adaptation(
            split=True,
            reason=f"{signal.seg_id}: {signal.kind.value} at {signal.context_tokens}/{signal.context_budget} tokens — halving",
        )
    return Adaptation(
        split=False,
        reason=f"{signal.seg_id}: {signal.kind.value} is not a size failure",
    )


def split_unit(
    segment: list[tuple[str, str]],
    concepts: list[dict[str, Any]],
) -> tuple[tuple[list[tuple[str, str]], list[dict[str, Any]]], tuple[list[tuple[str, str]], list[dict[str, Any]]]] | None:
    """Split one failed unit into two strictly smaller halves, or ``None``.

    Halves always partition the text. Concepts split contiguously; a lone
    concept is grounded against both text halves (the page upsert dedupes
    the double write). ``None`` means the unit cannot shrink — the caller
    keeps today's loud failure instead of re-sending the identical payload.
    """
    if not concepts:
        return None
    sources_a, sources_b = split_sources(segment)
    if not sources_a or not sources_b:
        return None
    concepts_a, concepts_b = split_concepts(concepts)
    if not concepts_a:
        return None
    if not concepts_b:
        concepts_b = list(concepts)
    return (sources_a, concepts_a), (sources_b, concepts_b)


def split_sources(segment: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Halve a source list by characters, keeping ``(label, content)`` pairs intact.

    A single source splits its characters with the label on both halves, so
    provenance survives. Both halves are non-empty whenever the input holds
    at least two characters; anything smaller cannot usefully shrink.
    """
    total = sum(len(content) for _, content in segment)
    if total < 2:
        return list(segment), []
    if len(segment) == 1:
        label, content = segment[0]
        cut = max(1, len(content) // 2)
        if cut >= len(content):
            return list(segment), []
        return [(label, content[:cut])], [(label, content[cut:])]
    half = total / 2
    running = 0
    for index, (_, content) in enumerate(segment):
        running += len(content)
        if running >= half and index < len(segment) - 1:
            return list(segment[: index + 1]), list(segment[index + 1 :])
    return list(segment[:-1]), [segment[-1]]


def split_concepts(concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Halve a concept list contiguously, mirroring the broker's fair split.

    Concepts are opaque to the shrink loop — partitioned, never inspected —
    so the value type stays ``Any``, matching the broker's concept maps.
    """
    if not concepts:
        return [], []
    cut = max(1, len(concepts) // 2)
    if cut >= len(concepts):
        return list(concepts), []
    return list(concepts[:cut]), list(concepts[cut:])
