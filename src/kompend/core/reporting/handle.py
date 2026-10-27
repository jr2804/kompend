"""Run handle: what :func:`log_run_start` hands to :func:`log_run_end`.

Kept out of ``runlog`` so the CLI can type its holder without importing the
run-log module, which reaches the config engine and the vault store. The
handle itself is four plain fields plus an optional truncation limit.

``error_chars`` is ``None`` for handles built outside ``log_run_start``;
``log_run_end`` resolves it from the display config then.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunHandle:
    """Handle returned by :func:`log_run_start`, passed to :func:`log_run_end`."""

    run_id: str
    cmd: str
    db_path: Path
    start_iso: str
    error_chars: int | None = None
