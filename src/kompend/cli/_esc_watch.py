"""Watch stdin for a double-ESC keypress and request a graceful cancel.

Two ESC presses within :data:`~kompend.core.cancel.CANCEL_WINDOW_SECONDS`
call ``CancelState.request("double-ESC")``; the pipeline then stops at its
next unit boundary, saves tier state, and the CLI exits cleanly (see
:mod:`kompend.core.cancel`).

Best-effort by design: when stdin is not a TTY, or the terminal cannot be
polled (no ``msvcrt`` / ``termios``), :meth:`EscDoublePressWatcher.start`
returns ``False`` and Ctrl+C remains the only interrupt. Known edge: on some
terminals Alt+<key> delivers ESC as a prefix, so two quick Alt-combos can
register as a double-ESC.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from collections.abc import Callable

from kompend.core.cancel import CANCEL_WINDOW_SECONDS, CancelState
from kompend.logging import get_logger

_POLL_INTERVAL_SECONDS = 0.02

#: Reason recorded on the CancelState when the double-ESC is detected.
DOUBLE_ESC_REASON = "double-ESC"

logger = get_logger(__name__)

#: Reads one key from stdin, or None when none is buffered. Non-blocking.
KeyReader = Callable[[], "str | None"]
#: Restores terminal state after watching.
Cleanup = Callable[[], None]


class EscDoublePressWatcher:
    """Daemon thread that turns two ESC presses within the window into a cancel request."""

    def __init__(
        self,
        state: CancelState,
        *,
        window: float = CANCEL_WINDOW_SECONDS,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
        read_key: KeyReader | None = None,
        is_tty: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_cancel: Callable[[str], None] | None = None,
    ) -> None:
        self.state = state
        self.window = window
        self.poll_interval = poll_interval
        self._read_key = read_key
        self._is_tty = is_tty if is_tty is not None else sys.stdin.isatty
        self._clock = clock
        self._on_cancel = on_cancel
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cleanup: Cleanup | None = None

    def start(self) -> bool:
        """Start watching. Returns ``False`` (and does nothing) when stdin is not interactive."""
        if self._thread is not None:
            return True
        if not self._is_tty():
            logger.debug("ESC cancel not available — stdin is not a TTY")
            return False
        if self._read_key is None:
            reader = _make_platform_reader()
            if reader is None:
                return False
            self._read_key, self._cleanup = reader
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="kompend-esc-cancel",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop watching and restore terminal state (idempotent)."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=1.0)

    def _run(self) -> None:
        last_esc: float | None = None
        try:
            while not self._stop.is_set():
                key = self._read_key() if self._read_key is not None else None
                if key is not None:
                    now = self._clock()
                    if key == "\x1b":
                        if last_esc is not None and now - last_esc <= self.window:
                            self.state.request(DOUBLE_ESC_REASON)
                            self._notify()
                            return
                        last_esc = now
                    else:
                        last_esc = None
                self._stop.wait(self.poll_interval)
        finally:
            if self._cleanup is not None:
                self._cleanup()

    def _notify(self) -> None:
        if self._on_cancel is None:
            return
        try:
            self._on_cancel(self.state.reason)
        except Exception:  # noqa: BLE001 — notice hook must never break cancelling
            logger.debug("cancel notice hook failed", exc_info=True)


def _make_platform_reader() -> tuple[KeyReader, Cleanup] | None:
    """Build the platform key reader + terminal cleanup.

    Returns ``None`` when no interactive terminal is available. POSIX puts
    the terminal in cbreak mode (ISIG stays on, so Ctrl+C keeps working) and
    the cleanup restores the saved attributes.
    """
    if sys.platform == "win32":
        return _windows_read_key, lambda: None
    try:
        select = importlib.import_module("select")
        termios = importlib.import_module("termios")

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        mode = termios.tcgetattr(fd)
        mode.lflag &= ~(termios.ICANON | termios.ECHO)
        mode.ccc[termios.VMIN] = 0
        mode.ccc[termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, mode)

        def _read() -> str | None:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                return None
            return os.read(fd, 1).decode("utf-8", errors="ignore") or None

        def _restore() -> None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except termios.error:
                logger.debug("termios restore failed — terminal may need reset")

        return _read, _restore
    except Exception as exc:  # noqa: BLE001 — feature probe, not a pipeline error path
        logger.debug("ESC cancel unavailable on this terminal: %s", exc)
        return None


def _windows_read_key() -> str | None:
    """Poll the Windows console buffer for one key event."""
    msvcrt = importlib.import_module("msvcrt")

    if not msvcrt.kbhit():
        return None
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        # Arrow/function-key prefix — drain the second half so it never
        # counts as a (non-ESC) keypress.
        msvcrt.getwch()
        return None
    return ch
