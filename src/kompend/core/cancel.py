"""Cooperative cancellation for long-running work.

A :class:`CancelState` is a thread-safe token shared between the CLI's
ESC-key watcher (who *requests*) and the pipeline/ingest loops (who *honour*
it at unit boundaries). Cancellation is cooperative: the current unit of
work always finishes, caches and tier state for completed units are
persisted, and the CLI exits with :data:`CANCEL_EXIT_CODE` — no traceback,
no half-written artifacts beyond the unit boundary.
"""

from __future__ import annotations

import threading

#: Two ESC presses within this window (seconds) request a graceful cancel.
CANCEL_WINDOW_SECONDS = 1.0

#: Exit code for a gracefully cancelled run (128 + SIGINT convention).
CANCEL_EXIT_CODE = 130

_DEFAULT_REASON = "cancelled by user"


class KompendCancelled(BaseException):
    """Raised at a unit boundary after a graceful cancel was requested.

    Inherits from ``BaseException`` (like ``KeyboardInterrupt``) so the many
    best-effort ``except Exception`` sites in parsers, agents, and progress
    callbacks cannot swallow a deliberate user abort.
    """

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason or _DEFAULT_REASON)
        self.reason = reason or _DEFAULT_REASON


class CancelState:
    """Thread-safe cancellation token.

    The key watcher thread calls :meth:`request`; worker loops call
    :meth:`raise_if_requested` (or check :meth:`requested`) at unit
    boundaries. Idempotent — the first reason wins.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""

    def request(self, reason: str = _DEFAULT_REASON) -> None:
        """Request cancellation (idempotent; the first reason wins)."""
        if self._event.is_set():
            return
        self._reason = reason or _DEFAULT_REASON
        self._event.set()

    def requested(self) -> bool:
        """True once a cancel has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """Why the cancel was requested ("" when never requested)."""
        return self._reason

    def raise_if_requested(self) -> None:
        """Raise :class:`KompendCancelled` when requested; no-op otherwise."""
        if self._event.is_set():
            raise KompendCancelled(self._reason)
