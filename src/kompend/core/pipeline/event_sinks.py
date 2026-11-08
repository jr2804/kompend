"""Built-in sinks for :mod:`kompend.core.pipeline.events` (knox-ty8f.7).

Three small adapters cover the in-tree consumers:

- :class:`CallbackSink` — the existing rich progress callback, unchanged;
- :class:`JsonlSink` — one JSON object per event, for a durable sidecar;
- :class:`CollectingSink` — an in-memory list for tests and local probes.

The ledger writer and the OTel bridge (knox-ty8f.9) plug in by implementing
:class:`~kompend.core.pipeline.events.PipelineEventSink`; they are not
special-cased here, because the bus is the seam and every consumer sees the
same payload.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from kompend.core.pipeline.events import PipelineEvent

__all__ = ["CallbackSink", "CollectingSink", "JsonlSink"]

ProgressCallback = Callable[[str, int, int, str], None]


class CallbackSink:
    """Adapt the legacy ``(phase, current, total, detail)`` callback."""

    def __init__(self, callback: ProgressCallback) -> None:
        self._callback = callback

    def on_event(self, event: PipelineEvent) -> None:
        self._callback(event.phase.value, event.current, event.total, event.detail)


class JsonlSink:
    """Append one JSON object per event to a file or text stream.

    The sink owns a file it opens by path and closes it with :meth:`close`;
    a caller-supplied stream stays open. Writes are serialized, so worker
    threads can publish through the same sink.
    """

    def __init__(self, target: Path | str | TextIO) -> None:
        self._lock = threading.Lock()
        if isinstance(target, (str, Path)):
            # Long-lived append handle owned by the sink and closed in close().
            self._stream = Path(target).open("a", encoding="utf-8", newline="\n")  # noqa: SIM115
            self._owns_stream = True
        else:
            self._stream = target
            self._owns_stream = False

    def on_event(self, event: PipelineEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock:
            self._stream.write(line)
            self._stream.write("\n")
            self._stream.flush()

    def close(self) -> None:
        """Close the owned file handle (no-op for a caller-supplied stream)."""
        if self._owns_stream:
            self._stream.close()


class CollectingSink:
    """Keep every event in order — the test collector for an ingest run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[PipelineEvent] = []

    def on_event(self, event: PipelineEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[PipelineEvent, ...]:
        """Snapshot of the collected events, oldest first."""
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        """Drop the collected events."""
        with self._lock:
            self._events.clear()
