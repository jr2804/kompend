"""Console entry point: answer a bare ``--version`` before the command tree loads.

``kompend.cli.app`` builds the whole command tree at import time — Cyclopts,
Rich, the flag types — and none of it is needed to print one string. The
launcher handles the version request first, bounds the numeric thread pools, and
delegates every other invocation unchanged. Both paths read
``kompend.__version__``, so the fast path cannot drift from the flag the meta app
declares.
"""

from __future__ import annotations

import faulthandler
import importlib
import logging
import os
import sys
import tempfile
import threading
from collections.abc import Callable
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
_FAULT_LOG: Any = None


#: What ``logging`` calls an exc_info triple: (type, value, traceback).
_ExcInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]


class _Unraisable(Protocol):
    """The parts of ``sys.UnraisableHookArgs`` this hook reads."""

    exc_type: type[BaseException] | None
    exc_value: BaseException | None
    exc_traceback: TracebackType | None
    object: object


def main() -> None:
    """Run the CLI; a lone version flag never imports the command tree."""
    _cap_thread_pools()
    if len(sys.argv) == 2 and sys.argv[1] in _VERSION_FLAGS:
        print(kompend.__version__)
        return
    _install_failure_hooks()
    _enable_fault_handler()
    importlib.import_module(OnDemandModule.APP.value).main()


def _enable_fault_handler() -> None:
    """Give a native abort somewhere to speak before the process disappears.

    A Rust or OpenBLAS abort inside a parser exits the process with no
    ``Exception`` and no message, so a long ingest just stops. ``faulthandler``
    prints every thread's Python stack when the fatal signal arrives, which names
    the frame that called into native code. It writes to a file rather than to
    stderr because the same failure mode can take the console handle with it.
    """
    global _FAULT_LOG  # noqa: PLW0603 - the handle must outlive this call
    if _FAULT_LOG is not None:
        return
    path = Path(tempfile.gettempdir()) / "kompend-faults.log"
    try:
        _FAULT_LOG = path.open("a", encoding="utf-8", errors="replace")
        _FAULT_LOG.write(f"\n=== kompend pid={os.getpid()} argv={sys.argv[1:]} ===\n")
        _FAULT_LOG.flush()
    except OSError:
        _FAULT_LOG = sys.stderr
    faulthandler.enable(file=_FAULT_LOG, all_threads=True)


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
