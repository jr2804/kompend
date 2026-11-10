"""A dying thread, a finalizer exception, or a native abort must leave a trace.

Three failure modes vanish without this module:

- an uncaught exception in a thread: CPython prints a header and *then* formats
  the traceback, so an allocation failure kills the report too — what survives is
  a spray of ``Exception in thread …`` lines with nothing under them;
- an exception nobody could catch (``__del__``, a finalizer);
- a fatal signal raised inside native code, which exits the process with no
  ``Exception`` at all, so a long run simply stops.

The first two are routed through the log with their thread or object named; the
third is ``faulthandler``'s job, writing to the file :func:`record_fault` uses.
That file is the one place a report can land when the console itself is gone.
"""

from __future__ import annotations

import faulthandler
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
from typing import Any, Final, Protocol, cast

from kompend.logging import get_logger

#: Where the fault handler and a dying console both record what happened.
FAULT_LOG_NAME: Final[str] = "kompend-faults.log"

#: First failure per (thread name, exception) pair is reported in full; see
#: :func:`log_thread_failure` for why repeats are demoted.
_THREAD_FAILURE_SEEN: dict[tuple[str, str], int] = {}

#: What ``logging`` calls an exc_info triple: (type, value, traceback).
ExcInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]


class Unraisable(Protocol):
    """The parts of ``sys.UnraisableHookArgs`` this hook reads."""

    exc_type: type[BaseException] | None
    exc_value: BaseException | None
    exc_traceback: TracebackType | None
    object: object


@cache
def enable_fault_handler() -> None:
    """Give a native abort somewhere to speak before the process disappears.

    A Rust or OpenBLAS abort inside a parser exits the process with no
    ``Exception`` and no message, so a long ingest just stops. ``faulthandler``
    prints every thread's Python stack when the fatal signal arrives, which names
    the frame that called into native code. It writes to a file rather than to
    stderr because the same failure mode can take the console handle with it.

    Cached, hence idempotent: the handler is enabled once per process.
    """
    faulthandler.enable(file=fault_log_stream(), all_threads=True)


def install_failure_hooks() -> None:
    """Report a dying thread or an ignored exception through the log."""
    threading.excepthook = log_thread_failure
    sys.unraisablehook = cast("Any", log_unraisable)


def log_thread_failure(args: threading.ExceptHookArgs) -> None:
    """Log an uncaught thread failure; never replace it with a reporting failure.

    The same thread can die the same way over and over — rich's progress
    refresh thread loses the console write surface and every retry prints an
    identical full traceback. The first occurrence is reported at ERROR with its
    traceback; repeats are summarised at DEBUG, because a dead daemon thread
    cannot recover and the repeats carry no new information.
    """
    name = args.thread.name if args.thread is not None else "unknown"
    exc = args.exc_value
    key = (name, f"{type(exc).__name__}: {exc}")
    seen = _THREAD_FAILURE_SEEN.get(key, 0)
    _THREAD_FAILURE_SEEN[key] = seen + 1
    if seen:
        _failure_logger().debug("thread %s died again (identical failure #%d): %r", name, seen + 1, exc)
        return
    report_failure(
        f"thread {name} died: {exc!r}",
        (args.exc_type, exc, args.exc_traceback),
        fallback=lambda: threading.__excepthook__(args),
    )


def log_unraisable(unraisable: Unraisable) -> None:
    """Log an exception nobody could catch (a ``__del__`` or a finalizer)."""
    report_failure(
        f"unraisable exception in {unraisable.object!r}: {unraisable.exc_value!r}",
        (unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback),
        fallback=lambda: sys.__unraisablehook__(cast("Any", unraisable)),
    )


def report_failure(message: str, info: ExcInfo, *, fallback: Callable[[], object]) -> None:
    """Write one failure line with its traceback, or hand it to the interpreter.

    The fallback exists for the reporter itself: a log handler that cannot write
    (a closed stream, a starved allocator) must not swallow the original failure,
    so the interpreter's own reporter gets a second chance.
    """
    try:
        _failure_logger().error("%s", message, exc_info=cast("Any", info))
    except Exception:  # reporting must not replace the report
        fallback()


def append_fault(line: str) -> None:
    """Append one line to the fault log, best effort.

    Called from paths that must not raise: a dying console, a fatal signal's
    neighbourhood. A log that cannot be written is skipped silently — the
    alternative is turning a reporting failure into the failure.
    """
    try:
        with fault_log_path().open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"{_stamp()} pid={os.getpid()} {line}\n")
    except OSError:
        return


@cache
def fault_log_stream() -> Any:
    """The stream ``faulthandler`` writes to, opened once per process.

    Cached so the handle stays referenced for the life of the process (a closed
    or collected stream would silence the dump). ``fault_log_stream.cache_clear()``
    is the reset path tests use.
    """
    try:
        handle = fault_log_path().open("a", encoding="utf-8", errors="replace")
        handle.write(f"\n=== kompend pid={os.getpid()} argv={sys.argv[1:]} ===\n")
        handle.flush()
    except OSError:
        return sys.stderr
    return handle


def fault_log_path() -> Path:
    """The fault log's location: the OS temp directory."""
    return Path(tempfile.gettempdir()) / FAULT_LOG_NAME


def _failure_logger() -> logging.Logger:
    """The logger the failure hooks report through."""
    return get_logger(__name__)


def _stamp() -> str:
    """A timestamp for one fault-log line."""
    return time.strftime("%Y-%m-%d %H:%M:%S")
