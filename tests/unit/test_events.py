"""Unit tests for the typed pipeline event bus (knox-ty8f.7)."""

from __future__ import annotations

from typing import cast

from kompend.core.pipeline.events import (
    PipelineEvent,
    PipelineEventBus,
    PipelineEventField,
    PipelineEventKind,
    PipelineEventSink,
    PipelinePhase,
)
from kompend.core.pipeline.stages.pipeline import WikiPipeline


class _ListSink:
    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def on_event(self, event: PipelineEvent) -> None:
        self.events.append(event)


class _BrokenSink:
    def on_event(self, event: PipelineEvent) -> None:
        raise RuntimeError("sink exploded")


def test_from_emit_classifies_the_phase_lifecycle() -> None:
    assert _event(detail="starting", current=0).kind is PipelineEventKind.STARTED
    assert _event(current=1, detail="a.md").kind is PipelineEventKind.PROGRESS
    assert _event(current=3, detail="done").kind is PipelineEventKind.COMPLETED


def test_from_emit_keeps_an_unknown_phase_on_the_bus() -> None:
    event = PipelineEvent.from_emit("not-a-phase", 0, 1, "starting")

    assert event.phase is PipelinePhase.OTHER


def test_to_dict_uses_the_field_vocabulary() -> None:
    record = _event().to_dict()

    assert set(record) == {field.value for field in PipelineEventField}
    assert record[PipelineEventField.PHASE.value] == "parse"
    assert record[PipelineEventField.KIND.value] == "started"


def test_bus_fans_out_and_isolates_a_broken_sink() -> None:
    bus = PipelineEventBus()
    first, second = _ListSink(), _ListSink()
    bus.subscribe(first)
    bus.subscribe(second)
    bus.subscribe(_BrokenSink())

    bus.publish(_event())

    assert len(first.events) == 1
    assert len(second.events) == 1


def test_bus_unsubscribe_is_idempotent() -> None:
    bus = PipelineEventBus()
    sink = _ListSink()
    unsubscribe = bus.subscribe(sink)

    bus.publish(_event())
    unsubscribe()
    unsubscribe()
    bus.publish(_event())

    assert len(sink.events) == 1


def _event(
    phase: PipelinePhase = PipelinePhase.PARSE,
    current: int = 0,
    total: int = 3,
    detail: str = "starting",
) -> PipelineEvent:
    return PipelineEvent.from_emit(phase.value, current, total, detail)


def test_emit_publishes_the_documented_ingest_sequence() -> None:
    class _Host:
        cancel = None
        _progress_callback = None
        event_bus = PipelineEventBus()

        def emit(self, phase: str, current: int, total: int, detail: str = "") -> None:
            WikiPipeline.emit(cast("WikiPipeline", self), phase, current, total, detail)

        def subscribe_events(self, sink: PipelineEventSink) -> None:
            self.event_bus.subscribe(sink)

    host = _Host()
    sink = _ListSink()
    host.subscribe_events(sink)

    for current, total, detail in [(0, 2, "starting"), (1, 2, "a.md"), (2, 2, "done")]:
        host.emit("parse", current, total, detail)

    assert [event.kind for event in sink.events] == [
        PipelineEventKind.STARTED,
        PipelineEventKind.PROGRESS,
        PipelineEventKind.COMPLETED,
    ]
    assert [event.phase for event in sink.events] == [PipelinePhase.PARSE] * 3
