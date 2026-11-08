"""Unit tests for the built-in pipeline event sinks (knox-ty8f.7)."""

from __future__ import annotations

import io
import json
from pathlib import Path

from kompend.core.pipeline.event_sinks import CallbackSink, CollectingSink, JsonlSink
from kompend.core.pipeline.events import PipelineEvent, PipelineEventField, PipelinePhase


def _event(*, current: int = 0, total: int = 3, detail: str = "starting") -> PipelineEvent:
    return PipelineEvent.from_emit(PipelinePhase.PARSE.value, current, total, detail)


def test_callback_sink_passes_the_legacy_arguments() -> None:
    seen: list[tuple[str, int, int, str]] = []
    sink = CallbackSink(lambda phase, current, total, detail: seen.append((phase, current, total, detail)))

    sink.on_event(_event(current=1, detail="a.md"))

    assert seen == [("parse", 1, 3, "a.md")]


def test_collecting_sink_keeps_order_and_clears() -> None:
    sink = CollectingSink()
    first = _event()
    second = _event(current=3, detail="done")

    sink.on_event(first)
    sink.on_event(second)

    assert sink.events == (first, second)
    sink.clear()
    assert sink.events == ()


def test_jsonl_sink_writes_one_record_per_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JsonlSink(path)
    try:
        sink.on_event(_event())
        sink.on_event(_event(current=3, detail="done"))
    finally:
        sink.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 2
    assert set(rows[0]) == {field.value for field in PipelineEventField}
    assert rows[1][PipelineEventField.DETAIL.value] == "done"


def test_jsonl_sink_leaves_a_caller_stream_open() -> None:
    stream = io.StringIO()
    sink = JsonlSink(stream)

    sink.on_event(_event())
    sink.close()

    assert not stream.closed
    assert json.loads(stream.getvalue())[PipelineEventField.PHASE.value] == "parse"
