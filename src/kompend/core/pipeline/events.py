"""Typed pipeline events — one emit, every observer (knox-ty8f.7).

``WikiPipeline.emit(phase, current, total, detail)`` stays the single
structured channel on the machine path. It now publishes a
:class:`PipelineEvent` to a :class:`PipelineEventBus` before handing the same
values to the legacy progress callback, so the rich bar, a JSONL recorder, a
test collector or an OTel bridge all see one consistent sequence.

Contract (epic knox-ty8f): stdlib only, no parallel logging framework. The
vocabulary is a :class:`StrEnum`, the payload is a frozen dataclass, and a
sink is any object with one ``on_event`` method. A broken sink is best-effort
enrichment: it is logged at WARNING and never aborts the pipeline.

Documented ingest sequence::

    parse:started      current=0, total=N, detail="starting"
    parse:item_started one per source before dispatch, source=<source name>
    parse:progress     one per source, detail=<source name>
    parse:completed    current=N, total=N, detail="done"
    summary:started    current=0, total=M, detail="starting"   (when enabled)
    summary:progress   one per tree
    summary:completed  current=M, total=M, detail="done"

The full pipeline adds the same started/progress/completed triple for
``index``, ``resolve``, ``compile`` and ``output``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol

from kompend.core.reporting.handle import current_run_id
from kompend.core.verbosity import is_verbose
from kompend.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "PipelineEvent",
    "PipelineEventBus",
    "PipelineEventField",
    "PipelineEventKind",
    "PipelineEventSink",
    "PipelinePhase",
]

_START_DETAIL: Final[str] = "starting"
_DONE_DETAIL: Final[str] = "done"


class PipelineEventField(StrEnum):
    """Keys of :meth:`PipelineEvent.to_dict` — the JSONL record contract."""

    PHASE = "phase"
    KIND = "kind"
    CURRENT = "current"
    TOTAL = "total"
    DETAIL = "detail"
    SOURCE = "source"
    ATTRIBUTES = "attributes"
    RUN_ID = "run_id"
    AT = "at"


class PipelineEventKind(StrEnum):
    """Where an event sits in its phase's lifetime."""

    STARTED = "started"
    ITEM_STARTED = "item_started"
    PROGRESS = "progress"
    COMPLETED = "completed"


class PipelinePhase(StrEnum):
    """Curated phases the pipeline emits progress for.

    Values are the strings already passed to ``WikiPipeline.emit`` and are a
    public contract (a JSONL consumer keys on them): add freely, never rename.
    ``OTHER`` is the fallback for an emit whose phase is outside the
    vocabulary — the typo is logged and the event is still published, so a
    progress label can never abort a run.
    """

    INDEX = "index"
    PARSE = "parse"
    SUMMARY = "summary"
    RESOLVE = "resolve"
    COMPILE = "compile"
    OUTPUT = "output"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """One structured progress event from the pipeline."""

    phase: PipelinePhase
    kind: PipelineEventKind
    current: int
    total: int
    detail: str = ""
    source: str | None = None
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    run_id: str | None = None
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready record; keys are :class:`PipelineEventField` values."""
        return {
            PipelineEventField.PHASE.value: self.phase.value,
            PipelineEventField.KIND.value: self.kind.value,
            PipelineEventField.CURRENT.value: self.current,
            PipelineEventField.TOTAL.value: self.total,
            PipelineEventField.DETAIL.value: self.detail,
            PipelineEventField.SOURCE.value: self.source,
            PipelineEventField.ATTRIBUTES.value: dict(self.attributes),
            PipelineEventField.RUN_ID.value: self.run_id,
            PipelineEventField.AT.value: self.at,
        }

    @classmethod
    def from_emit(
        cls,
        phase: str,
        current: int,
        total: int,
        detail: str = "",
        *,
        source: str | None = None,
        attributes: Mapping[str, str | int | float | bool] | None = None,
    ) -> PipelineEvent:
        """Build the typed event for one ``emit`` call.

        Unknown phase strings become :attr:`PipelinePhase.OTHER` with a
        warning instead of raising: progress reporting is best-effort and
        must never take the run down.
        """
        return cls(
            phase=_phase_for(phase),
            kind=_kind_for(current=current, total=total, detail=detail),
            current=current,
            total=total,
            detail=detail,
            source=source,
            attributes=dict(attributes or {}),
            run_id=current_run_id(),
        )

    @classmethod
    def item_started(
        cls,
        phase: str,
        current: int,
        total: int,
        source: str,
        *,
        attributes: Mapping[str, str | int | float | bool] | None = None,
    ) -> PipelineEvent:
        """Build the event that opens one item's span before dispatch.

        The rich bar never sees this kind: :meth:`WikiPipeline.emit_item_start`
        publishes it on the bus only, so adding per-item spans changes no
        progress output. ``current`` is the item's zero-based index.
        """
        return cls(
            phase=_phase_for(phase),
            kind=PipelineEventKind.ITEM_STARTED,
            current=current,
            total=total,
            detail=source,
            source=source,
            attributes=dict(attributes or {}),
            run_id=current_run_id(),
        )


class PipelineEventSink(Protocol):
    """Consumer of pipeline events.

    Implementations must be cheap and must not raise; the bus catches and
    logs anyway, because a diagnostic sink never gets to abort a run.
    """

    def on_event(self, event: PipelineEvent) -> None: ...


class PipelineEventBus:
    """Fan-out of typed pipeline events to registered sinks.

    Thread-safe: worker threads publish while callers subscribe or
    unsubscribe. ``publish`` iterates a snapshot, so a sink may unsubscribe
    itself from inside ``on_event``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sinks: tuple[PipelineEventSink, ...] = ()

    def subscribe(self, sink: PipelineEventSink) -> Callable[[], None]:
        """Register *sink*; returns an idempotent unsubscribe callable."""
        with self._lock:
            self._sinks = (*self._sinks, sink)

        def unsubscribe() -> None:
            self.unsubscribe(sink)

        return unsubscribe

    def unsubscribe(self, sink: PipelineEventSink) -> None:
        """Remove *sink* if present (idempotent)."""
        with self._lock:
            self._sinks = tuple(candidate for candidate in self._sinks if candidate is not sink)

    def publish(self, event: PipelineEvent) -> None:
        """Deliver *event* to every sink; a failing sink is logged and skipped."""
        with self._lock:
            sinks = self._sinks
        for sink in sinks:
            try:
                sink.on_event(event)
            except Exception as exc:  # best-effort: a sink never aborts the pipeline
                logger.warning(
                    "Pipeline event sink %s failed in phase %s — continuing: %s",
                    type(sink).__name__,
                    event.phase.value,
                    exc,
                    exc_info=is_verbose(),
                )


def _phase_for(phase: str) -> PipelinePhase:
    """Resolve a phase string; an unknown one degrades to ``OTHER``."""
    try:
        return PipelinePhase(phase)
    except ValueError:
        logger.warning(
            "Unknown pipeline phase %r — publishing as %s",
            phase,
            PipelinePhase.OTHER.value,
        )
        return PipelinePhase.OTHER


def _kind_for(*, current: int, total: int, detail: str) -> PipelineEventKind:
    """Classify one emit from the detail strings the call sites already use."""
    if detail == _START_DETAIL and current == 0:
        return PipelineEventKind.STARTED
    if detail == _DONE_DETAIL:
        return PipelineEventKind.COMPLETED
    return PipelineEventKind.PROGRESS
