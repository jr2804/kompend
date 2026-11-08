"""Unit tests for the pipeline OTel sink (knox-ty8f.9).

The sink is exercised through a fake tracer: no OpenTelemetry SDK, no
collector, no export. The contract under test is the span lifecycle and the
attribute names, not the exporter.
"""

from __future__ import annotations

from typing import Any

from kompend.core.pipeline.events import PipelineEvent, PipelineEventKind
from kompend.core.pipeline.otel_sink import OtelEventSink


class FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.ended = False
        self.status: Any = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any, description: str | None = None) -> None:
        self.status = (status, description)

    def end(self) -> None:
        self.ended = True


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_span(
        self,
        name: str,
        context: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> FakeSpan:
        span = FakeSpan(name)
        span.attributes.update(attributes or {})
        self.spans.append(span)
        return span


class _StubTrace:
    """Stand-in for ``opentelemetry.trace`` so the error-status branch runs."""

    class StatusCode:
        ERROR = "error"

    @staticmethod
    def Status(code: object, description: str | None = None) -> tuple[object, str | None]:
        return (code, description)

    @staticmethod
    def set_span_in_context(span: object) -> object:
        return span


def test_phase_lifecycle_opens_and_closes_a_span() -> None:
    sink, tracer = _sink()

    sink.on_event(PipelineEvent.from_emit("parse", 0, 2, "starting"))
    sink.on_event(PipelineEvent.from_emit("parse", 2, 2, "done"))
    sink.close()

    root, phase = tracer.spans
    assert root.name == "pipeline.run"
    assert phase.name == "pipeline.parse"
    assert phase.ended
    assert phase.attributes["kompend.phase"] == "parse"
    assert phase.attributes["kompend.current"] == 2
    assert phase.attributes["kompend.total"] == 2
    assert phase.attributes["kompend.duration_ms"] >= 0
    assert root.ended


def test_item_spans_open_at_dispatch_and_close_at_completion() -> None:
    sink, tracer = _sink()

    sink.on_event(PipelineEvent.from_emit("parse", 0, 2, "starting"))
    sink.on_event(PipelineEvent.item_started("parse", 0, 2, "a.md"))
    sink.on_event(PipelineEvent.from_emit("parse", 1, 2, "a.md", source="a.md"))
    sink.on_event(PipelineEvent.item_started("parse", 1, 2, "b.md"))
    sink.on_event(PipelineEvent.from_emit("parse", 2, 2, "b.md", source="b.md"))
    sink.on_event(PipelineEvent.from_emit("parse", 2, 2, "done"))
    sink.close()

    items = [span for span in tracer.spans if span.name == "pipeline.parse.source"]
    assert [span.attributes["kompend.source"] for span in items] == ["a.md", "b.md"]
    assert all(span.ended for span in items)
    assert all("kompend.incomplete" not in span.attributes for span in items)


def test_close_marks_still_open_spans_incomplete() -> None:
    sink, tracer = _sink()

    sink.on_event(PipelineEvent.from_emit("parse", 0, 1, "starting"))
    sink.on_event(PipelineEvent.item_started("parse", 0, 1, "stalled.pdf"))
    sink.close()

    item = tracer.spans[-1]
    phase = tracer.spans[1]
    assert item.ended
    assert item.attributes["kompend.incomplete"] is True
    assert item.status is not None
    assert phase.ended
    assert phase.attributes["kompend.incomplete"] is True
    assert sink._spans == {}
    assert sink._items == {}


def test_emit_attributes_are_copied_onto_the_span() -> None:
    sink, tracer = _sink()

    sink.on_event(
        PipelineEvent.from_emit(
            "parse",
            1,
            3,
            "a.md",
            source="a.md",
            attributes={"parser": "docling", "pages": 3, "cache_hit": False},
        )
    )
    sink.close()

    phase = tracer.spans[1]
    assert phase.attributes["kompend.parser"] == "docling"
    assert phase.attributes["kompend.pages"] == 3
    assert phase.attributes["kompend.cache_hit"] is False


def _sink() -> tuple[OtelEventSink, FakeTracer]:
    tracer = FakeTracer()
    sink = OtelEventSink(tracer=tracer)
    sink._otel_trace = _StubTrace()
    return sink, tracer


def test_item_started_event_is_typed_and_never_feeds_the_bar() -> None:
    event = PipelineEvent.item_started("parse", 4, 9, "source.md")

    assert event.kind is PipelineEventKind.ITEM_STARTED
    assert event.source == "source.md"
    assert event.to_dict()["source"] == "source.md"
    assert event.to_dict()["kind"] == "item_started"


def test_disabled_sink_does_nothing(monkeypatch: Any) -> None:
    monkeypatch.delenv("KOMP_OTEL_ENABLED", raising=False)
    sink = OtelEventSink()

    assert not sink.enabled
    sink.on_event(PipelineEvent.from_emit("parse", 0, 1, "starting"))
    sink.close()
