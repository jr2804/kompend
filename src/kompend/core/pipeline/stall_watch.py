"""Stall watchdog: one warning per item whose heartbeat goes silent (knox-ty8f.8).

A hung backend looks identical to a slow one from the outside — the progress
bar pins on one file and nothing distinguishes a wedged parser from a big
document. The watchdog closes that gap:

- ``watch(source)`` registers an item entering a watched region.
- ``beat(source, stage=...)`` records a phase transition (the heartbeat).
- A daemon observer thread scans the registry and warns **once per item**
  when no beat has arrived within ``stall_warn_s``, naming the item, the
  stage it is stuck in, and the elapsed time. It never kills anything —
  enforcement stays with the wall-clock budget (``ingest.parse_timeout_s``,
  knox-0mrn); the watchdog only attributes blame.
- ``last_stage(source)`` lets the deadline's ledger detail report the stage
  the file died in.

Contract (epic knox-ty8f): stdlib logging only — no parallel logging
framework. Warnings go through the normal logger; ``WatchStage`` is the
curated stage vocabulary (ledger stage names stay as they are).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Final

from kompend.logging import get_logger

#: Default seconds without a beat before an item is flagged. Matches the
#: scale users tolerate for a large PDF before wondering whether it hung.
DEFAULT_STALL_WARN_S: Final[float] = 30.0

#: Bounds for the observer tick interval — fast enough that small budgets
#: still fire roughly on time, slow enough to be invisible at defaults.
_MIN_TICK_S: Final[float] = 0.05
_MAX_TICK_S: Final[float] = 5.0

logger = get_logger(__name__)


class WatchStage(StrEnum):
    """Curated phases a watched item moves through.

    Values surface in stall warnings and ledger detail strings, so they are
    public vocabulary: add freely, never rename.
    """

    PARSE = "parse"
    POSTPROCESS = "postprocess"
    CHUNK = "chunk"
    ENRICH = "enrich"


class _Item:
    """One watched source: last beat time, phase, and warned flag."""

    __slots__ = ("last_beat", "stage", "warned")

    def __init__(self, *, now: float, stage: WatchStage) -> None:
        self.last_beat = now
        self.stage = stage
        self.warned = False


class StallWatchdog:
    """Warn once per item whose heartbeat has been silent too long.

    Thread-safe by design: worker threads call ``watch``/``beat``/``unwatch``
    while the observer thread scans. The scan is also callable directly
    (``_scan``) so tests can drive it with a fake clock instead of sleeping.
    """

    def __init__(self, stall_warn_s: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._stall_warn_s = stall_warn_s
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, _Item] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def stall_warn_s(self) -> float:
        """Seconds without a beat before an item is flagged."""
        return self._stall_warn_s

    def start(self) -> None:
        """Start the observer thread (idempotent; no-op when disabled).

        A non-positive ``stall_warn_s`` disables the watchdog entirely —
        the same zero-disables convention as ``ingest.parse_timeout_s``.
        """
        if self._stall_warn_s <= 0 or self._thread is not None:
            return
        tick = min(_MAX_TICK_S, max(_MIN_TICK_S, self._stall_warn_s / 10.0))
        self._thread = threading.Thread(
            target=self._monitor,
            args=(tick,),
            name="kompend-stall-watchdog",
            daemon=True,
        )
        self._thread.start()

    def watch(self, source: str, *, stage: WatchStage) -> None:
        """Register (or re-register) *source* entering a watched region."""
        if self._stall_warn_s <= 0:
            return
        with self._lock:
            self._items[source] = _Item(now=self._clock(), stage=stage)

    def beat(self, source: str, *, stage: WatchStage | None = None) -> None:
        """Record a phase transition for *source*; unknown sources are ignored."""
        if self._stall_warn_s <= 0:
            return
        with self._lock:
            item = self._items.get(source)
            if item is None:
                return
            item.last_beat = self._clock()
            if stage is not None:
                item.stage = stage

    def unwatch(self, source: str) -> None:
        """Remove *source* — finished or failed items never warn."""
        with self._lock:
            self._items.pop(source, None)

    def last_stage(self, source: str) -> WatchStage | None:
        """The stage *source* was last seen in (for deadline attribution)."""
        with self._lock:
            item = self._items.get(source)
            return item.stage if item is not None else None

    def close(self) -> None:
        """Stop the observer and drop every registered item."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_MAX_TICK_S)
        self._thread = None
        with self._lock:
            self._items.clear()

    def _monitor(self, tick: float) -> None:
        """Observer loop: scan, warn, sleep. Never raises into its thread."""
        while not self._stop.wait(tick):
            try:
                self._scan()
            except Exception as exc:  # a broken observer must never take down the run it watches
                # Same posture as ``_load_dedup_state``: the watchdog only ever
                # reports; its own failure is logged and the loop continues.
                logger.debug("Stall watchdog scan failed: %s", exc)

    def _scan(self) -> list[tuple[str, WatchStage, float]]:
        """Flag silent items; return what was flagged (for tests).

        Called with no arguments by the observer thread; tests call it
        directly after advancing an injected clock.
        """
        now = self._clock()
        flagged: list[tuple[str, WatchStage, float]] = []
        with self._lock:
            for source, item in self._items.items():
                elapsed = now - item.last_beat
                if not item.warned and elapsed >= self._stall_warn_s:
                    item.warned = True
                    flagged.append((source, item.stage, elapsed))
        for source, stage, elapsed in flagged:
            logger.warning(
                "no progress for %.1f s in stage %s: %s",
                elapsed,
                stage.value,
                source,
            )
        return flagged
