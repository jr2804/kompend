"""The console boundary: output must not be able to kill the run.

Two independent defects live here because both are about the terminal, and both
are fixed where the process touches it rather than in every writer:

- a console handle that a native library invalidates mid-run: every later write
  raises ``OSError``, rich's refresh thread dies, and the run ends as CPython's
  silent exit status 120 (``Py_FinalizeEx`` could not flush) — see
  :class:`HardenedStream`;
- an OEM code page that decodes UTF-8 bytes, so glyphs no OEM page holds print
  as ``Ôá╝Ôöü`` — see :func:`ensure_utf8_code_page`.
"""

from __future__ import annotations

import ctypes
import errno
import io
import os
import sys
import threading
from typing import Any, Final, Protocol, cast

from kompend.cli.runtime.faults import append_fault

#: How often one stream may be rebound; a handle that keeps dying is a dead run.
REPAIR_LIMIT: Final[int] = 8

#: The console code page that decodes UTF-8 (``chcp 65001``).
CP_UTF8: Final[int] = 65001


class ConsoleCodePageApi(Protocol):
    """The two ``kernel32`` calls the code-page switch needs."""

    def GetConsoleOutputCP(self) -> int:
        """Return the attached console's output code page, 0 when there is none."""

    def SetConsoleOutputCP(self, code_page: int) -> int:
        """Set the console's output code page; returns non-zero on success."""


class HardenedStream:
    """A console stream proxy that survives its own handle dying.

    Console I/O is a trust boundary: a native library may take the handle down
    at any moment, and the run's durable output (vault, cache, databases) must
    not be discarded because a progress bar could not be drawn. Only a dead
    handle is absorbed — every other ``OSError`` (a full disk, a broken pipe a
    consumer actually reads) still propagates.
    """

    def __init__(self, raw: Any, *, name: str) -> None:
        self._raw = raw
        self._name = name
        self._noted = False
        self._repairs = 0
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        """Write, repairing a rebound handle or absorbing a dead one."""
        try:
            return self._raw.write(data)
        except OSError as exc:
            if not is_dead_handle(exc):
                raise
            with self._lock:
                if self._repair():
                    try:
                        return self._raw.write(data)
                    except OSError as retry_exc:
                        if not is_dead_handle(retry_exc):
                            raise
                self._note_death(exc)
        return len(data)

    def flush(self) -> None:
        """Flush, repairing a rebound handle or absorbing a dead one.

        Absorbing here is what prevents exit status 120: ``Py_FinalizeEx``
        flushes the Python-level streams on the way out.
        """
        try:
            self._raw.flush()
        except OSError as exc:
            if not is_dead_handle(exc):
                raise
            with self._lock:
                if self._repair():
                    try:
                        self._raw.flush()
                        return
                    except OSError as retry_exc:
                        if not is_dead_handle(retry_exc):
                            raise
                self._note_death(exc)

    def __getattr__(self, attribute: str) -> Any:
        """Everything else (``isatty``, ``fileno``, ``encoding``, …) is the raw stream."""
        return getattr(self._raw, attribute)

    def _repair(self) -> bool:
        """Rebind to a fresh stream over the stream's *current* descriptor.

        Returns ``False`` when the descriptor itself is unusable or the repair
        budget is spent, in which case the caller notes the death and absorbs.
        """
        if self._repairs >= REPAIR_LIMIT:
            return False
        try:
            descriptor = self._raw.fileno()
        except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
            return False
        encoding = getattr(self._raw, "encoding", None) or "utf-8"
        errors = getattr(self._raw, "errors", None) or "replace"
        try:
            fresh = os.fdopen(
                os.dup(descriptor),
                "w",
                encoding=encoding,
                errors=errors,
                buffering=1,
                closefd=True,
            )
        except OSError:
            return False
        self._repairs += 1
        self._raw = fresh
        return True

    def _note_death(self, exc: BaseException) -> None:
        """Record the death once: fault log first, then one try on the sibling."""
        if self._noted:
            return
        self._noted = True
        append_fault(f"console {self._name} died: {exc!r}")
        sibling = sys.stderr if self._name == "stdout" else sys.stdout
        try:
            sibling.write(f"\nkompend: the console {self._name} is no longer writable ({exc!r}); continuing without it.\n")
            sibling.flush()
        except OSError:
            pass


def harden_console_streams() -> None:
    """Wrap stdout/stderr so a dying console handle cannot kill the run.

    On Windows the first heavy parse of a run can invalidate the process's
    console handles: every later console write raises ``OSError`` with
    ``errno 9`` / ``winerror 6``. Unwrapped, that kills rich's refresh thread,
    escapes through the logger into the pipeline, and finally makes
    ``Py_FinalizeEx``'s flush fail — exit status 120, thrown away without a
    traceback.

    The cause is not a lost console but a *rebound* one: a backend that captures
    native engine output re-points file descriptor 2 with ``os.dup2`` for the
    duration of a conversion, and the C-level stream bound to the old handle
    value never recovers. Python's buffered stream caches its handle at startup,
    so the proxy reopens a fresh stream over the *current* descriptor — the
    console comes back instead of going quiet for the rest of the run. If the
    descriptor is really gone the death is absorbed and noted once; the vault,
    not the terminal, is the run's real output.
    """
    if isinstance(sys.stdout, HardenedStream):
        return
    sys.stdout = HardenedStream(sys.stdout, name="stdout")
    sys.stderr = HardenedStream(sys.stderr, name="stderr")


def is_dead_handle(exc: BaseException) -> bool:
    """True for the two ``OSError`` shapes of an invalidated console handle."""
    return isinstance(exc, OSError) and (
        exc.errno == errno.EBADF or getattr(exc, "winerror", None) == 6  # ERROR_INVALID_HANDLE
    )


def ensure_utf8_code_page(api: ConsoleCodePageApi | None = None) -> None:
    """Make an attached Windows console decode output as UTF-8.

    Rich draws the progress bar with glyphs no OEM code page holds (``⠼``,
    ``━``), and outside a real console handle the stream carries them as UTF-8
    bytes. The console then decodes those bytes with its active code page and
    the bar reads ``Ôá╝Ôöü`` instead of ``⠼ ━``. Switching the output code page
    to UTF-8 leaves the Unicode console API untouched and makes the byte path
    agree with it; a host without a console (code page 0) is left alone.
    """
    if api is None:
        if sys.platform != "win32":
            return
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:
            return
        api = cast("ConsoleCodePageApi", win_dll("kernel32", use_last_error=True))
    try:
        if api.GetConsoleOutputCP() in (0, CP_UTF8):
            return
        api.SetConsoleOutputCP(CP_UTF8)
    except OSError:
        return
