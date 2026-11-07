"""Run handle: what :func:`log_run_start` hands to :func:`log_run_end`.

Kept out of ``runlog`` so the CLI can type its holder without importing the
run-log module, which reaches the config engine and the vault store. The
handle itself is four plain fields plus an optional truncation limit.

``error_chars`` is ``None`` for handles built outside ``log_run_start``;
``log_run_end`` resolves it from the display config then.

This module also owns the *current run* context: a ``ContextVar`` set by
``log_run_start`` so observability writers (the per-item ledger) can
correlate their rows with the CLI run without threading the handle through
the pipeline. ``copy_context()`` (used by the parse thread pool) carries it
into worker threads.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

_CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("kompend_current_run_id", default=None)


@dataclass(slots=True)
class RunHandle:
    """Handle returned by :func:`log_run_start`, passed to :func:`log_run_end`."""

    run_id: str
    cmd: str
    db_path: Path
    start_iso: str
    error_chars: int | None = None


def set_current_run_id(run_id: str | None) -> None:
    """Record the run that owns this task's writes (or clear it with ``None``)."""
    _CURRENT_RUN_ID.set(run_id)


def current_run_id() -> str | None:
    """The run :func:`log_run_start` opened in this context, if any."""
    return _CURRENT_RUN_ID.get()
