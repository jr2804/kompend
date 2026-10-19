"""Interpreter parallelism facts behind ``compile.workers``.

kompend parallelises Tier 1 with a :class:`~concurrent.futures.ThreadPoolExecutor`.
Whether that actually runs Python code in parallel depends on the
interpreter, not on kompend:

- **GIL build** (CPython 3.13/3.14 without the ``t`` ABI) — threads
  interleave, they do not run Python bytecode at the same time. Parsing and
  OCR are CPU-bound, so ``workers > 1`` buys overlap for I/O and backends that
  release the GIL (ONNX, native PDF code), not true parallelism.
- **Free-threaded build** (CPython 3.14t+) — ``gil_enabled()`` is false and the
  same thread pool is genuinely parallel.

The check exists so the pipeline can say which of the two it is running on
instead of leaving a silent performance trap behind ``--workers``.
"""

from __future__ import annotations

import sys

#: Message logged once when a thread pool runs on a GIL build.
GIL_BUILD_WARNING: str = (
    "Thread pool of %d workers on a GIL-enabled interpreter: CPU-bound parsing "
    "does not scale across workers. A free-threaded CPython build (3.14t+) "
    "parses documents in parallel with the same setting."
)

#: Message logged once when a thread pool runs on a free-threaded build.
FREE_THREADED_NOTE: str = "Thread pool of %d workers on a free-threaded interpreter: parsing runs in parallel."


def describe_parallelism(workers: int) -> str:
    """Return the one-line parallelism report for *workers* threads."""
    template = FREE_THREADED_NOTE if not gil_enabled() else GIL_BUILD_WARNING
    return template % workers


def gil_enabled() -> bool:
    """Whether the running interpreter holds the global interpreter lock.

    ``sys._is_gil_enabled`` exists from CPython 3.13 onwards (the oldest
    version kompend supports). A missing attribute means a GIL build — the
    conservative answer, since only a free-threaded build omits the lock.
    """
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_enabled is None:
        return True
    return bool(is_enabled())
