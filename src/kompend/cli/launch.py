"""Console entry point: answer a bare ``--version`` before the command tree loads.

``kompend.cli.app`` builds the whole command tree at import time — Cyclopts,
Rich, the flag types — and none of it is needed to print one string. The
launcher handles the version request first, bounds the numeric thread pools, and
delegates every other invocation unchanged. Both paths read
``kompend.__version__``, so the fast path cannot drift from the flag the meta app
declares.
"""

from __future__ import annotations

import errno
import faulthandler
import importlib
import io
import logging
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, cast

import kompend
from kompend.cli._registry import OnDemandModule
from kompend.core.env import EnvVar
from kompend.logging import get_logger

#: The flags the meta app's eager version option answers on their own.
_VERSION_FLAGS = frozenset({"--version", "-v"})

#: Thread-pool variables that numeric libraries read at import time. OpenBLAS
#: multiplies every large allocation by its thread count, so an unbounded pool is
#: a memory bomb: on a 16-core host a single docling parse of an ITU-T PDF reached
#: the commit limit and OpenBLAS killed the process outright ("Memory allocation
#: still failed after 10 retries, giving up") instead of raising `MemoryError` —
#: which a caller could have handled. Capping the pool converts that abort into
#: slower arithmetic.
_THREAD_POOL_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Default cap. Eight threads keep a 16-core box responsive while a parse runs and
#: halve the per-allocation thread buffers; raise it with ``KOMP_BLAS_THREADS``.
_DEFAULT_THREAD_CAP = 8


#: First failure per (thread name, exception) pair is reported in full; see
#: :func:`_log_thread_failure` for why repeats are demoted.
_THREAD_FAILURE_SEEN: dict[tuple[str, str], int] = {}

#: Kept open for the process lifetime: ``faulthandler`` writes to the object it
#: was given, so closing it behind its back would silence the dump.
#: (:func:`_fault_log_stream` owns it — a cached getter, not a module variable.)

#: Where the stream guard and the fault handler both record console deaths.
_FAULT_LOG_NAME = "kompend-faults.log"


#: How often one stream may be rebound; a handle that keeps dying is a dead run.
_REPAIR_LIMIT = 8


#: What ``logging`` calls an exc_info triple: (type, value, traceback).
_ExcInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]


class _Unraisable(Protocol):
    """The parts of ``sys.UnraisableHookArgs`` this hook reads."""

    exc_type: type[BaseException] | None
    exc_value: BaseException | None
    exc_traceback: TracebackType | None
    object: object


class _HardenedStream:
    """A console stream proxy that survives its own handle dying; see
    :func:`_harden_console_streams`.

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
            if not _is_dead_handle(exc):
                raise
            with self._lock:
                if self._repair():
                    try:
                        return self._raw.write(data)
                    except OSError as retry_exc:
                        if not _is_dead_handle(retry_exc):
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
            if not _is_dead_handle(exc):
                raise
            with self._lock:
                if self._repair():
                    try:
                        self._raw.flush()
                        return
                    except OSError as retry_exc:
                        if not _is_dead_handle(retry_exc):
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
        if self._repairs >= _REPAIR_LIMIT:
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
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            path = Path(tempfile.gettempdir()) / _FAULT_LOG_NAME
            with path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(f"{stamp} pid={os.getpid()} console {self._name} died: {exc!r}\n")
        except OSError:
            pass
        sibling = sys.stderr if self._name == "stdout" else sys.stdout
        try:
            sibling.write(f"\nkompend: the console {self._name} is no longer writable ({exc!r}); continuing without it.\n")
            sibling.flush()
        except OSError:
            pass


def main() -> None:
    """Run the CLI; a lone version flag never imports the command tree."""
    _cap_thread_pools()
    if len(sys.argv) == 2 and sys.argv[1] in _VERSION_FLAGS:
        print(kompend.__version__)
        return
    _harden_console_streams()
    _install_failure_hooks()
    _enable_fault_handler()
    importlib.import_module(OnDemandModule.APP.value).main()


@cache
def _enable_fault_handler() -> None:
    """Give a native abort somewhere to speak before the process disappears.

    A Rust or OpenBLAS abort inside a parser exits the process with no
    ``Exception`` and no message, so a long ingest just stops. ``faulthandler``
    prints every thread's Python stack when the fatal signal arrives, which names
    the frame that called into native code. It writes to a file rather than to
    stderr because the same failure mode can take the console handle with it.

    Cached, hence idempotent: the handler is enabled once per process.
    """
    faulthandler.enable(file=_fault_log_stream(), all_threads=True)


@cache
def _fault_log_stream() -> Any:
    """The stream ``faulthandler`` writes to, opened once per process.

    Cached so the handle stays referenced for the life of the process (a closed
    or collected stream would silence the dump). ``_fault_log_stream.cache_clear()``
    is the reset path tests use.
    """
    path = Path(tempfile.gettempdir()) / _FAULT_LOG_NAME
    try:
        handle = path.open("a", encoding="utf-8", errors="replace")
        handle.write(f"\n=== kompend pid={os.getpid()} argv={sys.argv[1:]} ===\n")
        handle.flush()
    except OSError:
        return sys.stderr
    return handle


def _harden_console_streams() -> None:
    """Wrap stdout/stderr so a dying console handle cannot kill the run.

    On Windows the first heavy parse of a run can invalidate the process's
    console handles: every later console write raises ``OSError`` with
    ``errno 9`` / ``winerror 6``. Unwrapped, that kills rich's refresh thread,
    escapes through the logger into the pipeline, and finally makes
    ``Py_FinalizeEx``'s flush fail — CPython's exit status 120, thrown away
    without a traceback.

    The cause is not a lost console but a *rebound* one: a backend that captures
    native engine output re-points file descriptor 2 with ``os.dup2`` for the
    duration of a conversion, and the C-level stream bound to the old handle
    value never recovers. Python's buffered stream caches its handle at startup,
    so the proxy reopens a fresh stream over the *current* descriptor — the
    console comes back instead of going quiet for the rest of the run. If the
    descriptor is really gone the death is absorbed and noted once; the vault,
    not the terminal, is the run's real output.
    """
    if isinstance(sys.stdout, _HardenedStream):
        return
    sys.stdout = _HardenedStream(sys.stdout, name="stdout")
    sys.stderr = _HardenedStream(sys.stderr, name="stderr")


def _is_dead_handle(exc: BaseException) -> bool:
    """True for the two ``OSError`` shapes of an invalidated console handle."""
    return isinstance(exc, OSError) and (
        exc.errno == errno.EBADF or getattr(exc, "winerror", None) == 6  # ERROR_INVALID_HANDLE
    )


def _cap_thread_pools() -> None:
    """Bound the numeric thread pools before numpy, torch or OpenBLAS load.

    Values already present in the environment win — a user who pinned a pool size
    made a decision this default must not overwrite.
    """
    requested = os.environ.get(EnvVar.BLAS_THREADS, "")
    cap = requested.strip() or str(min(_DEFAULT_THREAD_CAP, os.cpu_count() or 1))
    for name in _THREAD_POOL_ENV_VARS:
        os.environ.setdefault(name, cap)


def _install_failure_hooks() -> None:
    """Report a dying thread or an ignored exception through the log.

    CPython's own reporters print a header and then format the traceback. When the
    failure that killed the thread is an allocation failure, formatting allocates
    too, so the report never appears: the interpreter falls through to its
    backstops and prints ``Exception in threading.excepthook:``,
    ``Exception ignored in thread started by …`` and
    ``Exception ignored in sys.unraisablehook`` — a spray of headers with no
    traceback and no thread name. Routing both hooks through the logger makes the
    next occurrence readable, and costs nothing until something dies.
    """
    threading.excepthook = _log_thread_failure
    sys.unraisablehook = cast("Any", _log_unraisable)


def _log_thread_failure(args: threading.ExceptHookArgs) -> None:
    """Log an uncaught thread failure; never replace it with a reporting failure.

    The same thread can die the same way over and over — rich's progress
    refresh thread loses the console write surface (OSError 9 on ConPTY) and
    every retry prints an identical full traceback. The first occurrence is
    reported at ERROR with its traceback; repeats are summarised at DEBUG,
    because a dead daemon thread cannot recover and the repeats carry no new
    information.
    """
    name = args.thread.name if args.thread is not None else "unknown"
    exc = args.exc_value
    key = (name, f"{type(exc).__name__}: {exc}")
    seen = _THREAD_FAILURE_SEEN.get(key, 0)
    _THREAD_FAILURE_SEEN[key] = seen + 1
    if seen:
        _failure_logger().debug("thread %s died again (identical failure #%d): %r", name, seen + 1, exc)
        return
    _report(
        f"thread {name} died: {exc!r}",
        (args.exc_type, exc, args.exc_traceback),
        fallback=lambda: threading.__excepthook__(args),
    )


def _log_unraisable(unraisable: _Unraisable) -> None:
    """Log an exception nobody could catch (a ``__del__`` or a finalizer)."""
    _report(
        f"unraisable exception in {unraisable.object!r}: {unraisable.exc_value!r}",
        (unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback),
        fallback=lambda: sys.__unraisablehook__(cast("Any", unraisable)),
    )


def _report(message: str, info: _ExcInfo, *, fallback: Callable[[], object]) -> None:
    """Write one failure line with its traceback, or hand it to the interpreter.

    The fallback exists for the reporter itself: a log handler that cannot write
    (a closed stream, a starved allocator) must not swallow the original failure,
    so the interpreter's own reporter gets a second chance.
    """
    try:
        _failure_logger().error("%s", message, exc_info=cast("Any", info))
    except BaseException:  # noqa: BLE001 - reporting must not replace the report
        fallback()


def _failure_logger() -> logging.Logger:
    """The logger, imported here so a bare ``--version`` never pulls in Rich."""
    return get_logger(__name__)
