"""OpenTelemetry span sink for pipeline events (knox-ty8f.9).

Subscribes to the :class:`~kompend.core.pipeline.events.PipelineEventBus` the
machine path already emits on (knox-ty8f.7) and turns the phase lifecycle into
spans::

    pipeline.run
      pipeline.parse
        pipeline.parse.source   (one per source, opened at item start)
      pipeline.summary
      pipeline.resolve
      pipeline.compile
      pipeline.output

Attributes carry the run correlation and the values the bus knows:
``kompend.run_id``, ``kompend.phase``, ``kompend.current``, ``kompend.total``,
``kompend.detail``, ``kompend.source``. An emit may pass a typed ``attributes``
mapping (parser, backend, pages, bytes, cache_hit); the sink copies those onto
the span under the ``kompend.`` prefix, so a stage can add detail without
changing this module.

The module is optional-dependency safe: the OpenTelemetry SDK lives behind the
``otel`` extra, and nothing here imports it until ``KOMP_OTEL_ENABLED`` is
truthy. Disabled means zero work beyond the environment check. When enabled the
sink reuses the provider :mod:`kompend.agents.instrument` configures for the
agent spans, so pipeline and agent traces land in the same backend with the
same service name.

A span that never sees its ``completed`` event (a stalled parse, a cancelled
run) is ended at :meth:`OtelEventSink.close` with
``kompend.incomplete=true`` and an error status — the trace still says the
stage was entered, and a missing duration is visible instead of a silently
absent span.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Final

from kompend.core.env import EnvVar
from kompend.core.pipeline.events import PipelineEvent, PipelineEventKind, PipelinePhase
from kompend.logging import get_logger

logger = get_logger(__name__)

__all__ = ["OtelEventSink"]

_TRACER_NAME: Final[str] = "kompend.pipeline"
_RUN_SPAN_NAME: Final[str] = "pipeline.run"
_ATTR_PREFIX: Final[str] = "kompend."
_ATTR_RUN_ID: Final[str] = "kompend.run_id"
_ATTR_PHASE: Final[str] = "kompend.phase"
_ATTR_CURRENT: Final[str] = "kompend.current"
_ATTR_TOTAL: Final[str] = "kompend.total"
_ATTR_DETAIL: Final[str] = "kompend.detail"
_ATTR_SOURCE: Final[str] = "kompend.source"
_ATTR_INDEX: Final[str] = "kompend.index"
_ATTR_DURATION_MS: Final[str] = "kompend.duration_ms"
_ATTR_INCOMPLETE: Final[str] = "kompend.incomplete"
_TRUE_VALUES: Final[tuple[str, ...]] = ("1", "true", "yes")


class OtelEventSink:
    """Pipeline event sink that projects the phase lifecycle onto OTel spans."""

    def __init__(self, tracer: Any | None = None) -> None:
        self._tracer: Any = tracer
        self._otel_trace: Any | None = None
        self._root: Any | None = None
        self._root_context: Any | None = None
        self._spans: dict[PipelinePhase, Any] = {}
        self._started_at: dict[PipelinePhase, float] = {}
        self._items: dict[tuple[PipelinePhase, str], Any] = {}
        self.enabled = tracer is not None or _enabled()
        if not self.enabled or tracer is not None:
            return
        try:
            otel_trace = import_module("opentelemetry.trace")
            instrument = import_module("kompend.agents.instrument")
        except ImportError as exc:
            logger.debug("OTel extra is not installed — pipeline spans disabled: %s", exc)
            self.enabled = False
            self._tracer = None
            return
        instrument.configure_observability()
        self._otel_trace = otel_trace
        self._tracer = otel_trace.get_tracer(_TRACER_NAME)

    def on_event(self, event: PipelineEvent) -> None:
        """Open, annotate, or close the span that matches *event*."""
        if not self.enabled or self._tracer is None:
            return
        if event.kind is PipelineEventKind.ITEM_STARTED:
            self._start_item(event)
            return
        if self._root is None:
            self._start_root(event)
        if event.kind is PipelineEventKind.STARTED:
            self._start_phase(event)
        elif event.kind is PipelineEventKind.COMPLETED:
            self._complete_phase(event)
        else:
            self._record_progress(event)

    def close(self) -> None:
        """End every span still open; safe to call more than once.

        Called by :class:`~kompend.core.pipeline.stages.pipeline.WikiPipeline`
        at the end of a run. A direct stage call can call it too; without it the
        spans stay in memory and are never exported by the batch processor.
        """
        if not self.enabled:
            return
        for phase, source in tuple(self._items):
            self._finish_item(phase, source, incomplete=True)
        for _phase, span in tuple(self._spans.items()):
            self._finish(span, incomplete=True)
        self._spans.clear()
        self._started_at.clear()
        if self._root is not None:
            self._root.end()
            self._root = None
            self._root_context = None

    # ── Span lifecycle ─────────────────────────────────────────────────

    def _start_root(self, event: PipelineEvent) -> None:
        self._root = self._tracer.start_span(
            _RUN_SPAN_NAME,
            attributes={_ATTR_RUN_ID: event.run_id or ""},
        )
        self._root_context = self._context(self._root)

    def _start_phase(self, event: PipelineEvent) -> None:
        stale = self._spans.pop(event.phase, None)
        if stale is not None:
            self._finish(stale, incomplete=True)
        span = self._tracer.start_span(
            f"pipeline.{event.phase.value}",
            context=self._root_context,
            attributes=self._base_attributes(event),
        )
        self._spans[event.phase] = span
        self._started_at[event.phase] = event.at

    def _record_progress(self, event: PipelineEvent) -> None:
        span = self._spans.get(event.phase)
        if span is None:
            self._start_phase(event)
            span = self._spans[event.phase]
        self._apply(span, event)
        span.add_event("progress", attributes=self._base_attributes(event))
        if event.source:
            self._finish_item(event.phase, event.source, incomplete=False)

    def _complete_phase(self, event: PipelineEvent) -> None:
        span = self._spans.get(event.phase)
        if span is None:
            self._start_phase(event)
            span = self._spans[event.phase]
        self._apply(span, event)
        started_at = self._started_at.pop(event.phase, event.at)
        span.set_attribute(_ATTR_DURATION_MS, max(0.0, (event.at - started_at) * 1000.0))
        self._finish_phase_items(event.phase)
        self._finish(span, incomplete=False)
        self._spans.pop(event.phase, None)

    def _start_item(self, event: PipelineEvent) -> None:
        phase_span = self._spans.get(event.phase)
        if phase_span is None:
            self._start_phase(event)
            phase_span = self._spans[event.phase]
        source = event.source or event.detail
        stale = self._items.pop((event.phase, source), None)
        if stale is not None:
            self._finish(stale, incomplete=True)
        item = self._tracer.start_span(
            f"pipeline.{event.phase.value}.source",
            context=self._context(phase_span),
            attributes={
                **self._base_attributes(event),
                _ATTR_SOURCE: source,
                _ATTR_INDEX: event.current,
            },
        )
        self._items[(event.phase, source)] = item

    def _finish_item(self, phase: PipelinePhase, source: str, *, incomplete: bool) -> None:
        item = self._items.pop((phase, source), None)
        if item is not None:
            self._finish(item, incomplete=incomplete)

    def _finish_phase_items(self, phase: PipelinePhase) -> None:
        for key in tuple(self._items):
            if key[0] is phase:
                self._finish_item(key[0], key[1], incomplete=True)

    def _finish(self, span: Any, *, incomplete: bool) -> None:
        if incomplete:
            span.set_attribute(_ATTR_INCOMPLETE, True)
            if self._otel_trace is not None:
                error_status = self._otel_trace.Status(
                    self._otel_trace.StatusCode.ERROR,
                    "span ended without a completed event",
                )
                span.set_status(error_status)
        span.end()

    # ── Attributes ─────────────────────────────────────────────────────

    def _apply(self, span: Any, event: PipelineEvent) -> None:
        for key, value in self._base_attributes(event).items():
            span.set_attribute(key, value)

    def _context(self, span: Any) -> Any | None:
        if self._otel_trace is None:
            return None
        return self._otel_trace.set_span_in_context(span)

    @staticmethod
    def _base_attributes(event: PipelineEvent) -> dict[str, str | int | float | bool]:
        attributes: dict[str, str | int | float | bool] = {
            _ATTR_RUN_ID: event.run_id or "",
            _ATTR_PHASE: event.phase.value,
            _ATTR_CURRENT: event.current,
            _ATTR_TOTAL: event.total,
        }
        if event.detail:
            attributes[_ATTR_DETAIL] = event.detail
        for key, value in event.attributes.items():
            attributes[_qualify(key)] = value
        return attributes


def _enabled() -> bool:
    """Whether ``KOMP_OTEL_ENABLED`` asks for export (same rule as the agents)."""
    return os.environ.get(EnvVar.OTEL_ENABLED, "").strip().lower() in _TRUE_VALUES


def _qualify(key: str) -> str:
    """Namespace a caller attribute; an already-qualified key is left alone."""
    return key if key.startswith(_ATTR_PREFIX) else f"{_ATTR_PREFIX}{key}"
