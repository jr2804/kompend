"""Append-only audit trail of every CLI command run (the ``runs`` table).

Each command inserts a ``start`` row on entry and an ``end`` row on exit,
linked by a short ``run_id``. Querying ``runs`` answers "did the build run
through, and if not, where did it stop?":

- an ``end`` row with ``status="ok"``  → command finished cleanly
- an ``end`` row with ``status="error"`` → command crashed (error captured)
- a ``start`` row with no matching ``end`` → process was killed / hung

Rows are tiny; the table is pruned to a soft cap so the store stays bounded.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict

from kompend.core.config import DEFAULTS, resolve_config
from kompend.core.config.keys import DISPLAY_CLI_ARG_PREVIEW, DISPLAY_LOG_ERROR_CHARS, PATHS_STATE_DIR, PATHS_WIKI_DIR
from kompend.core.paths import CONFIG_FILE
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import RUNS_TABLE
from kompend.core.utils.time import iso_local, now_local
from kompend.logging import get_logger

_SOFT_CAP_ROWS = 1000
_KEEP_ROWS = 500

logger = get_logger(__name__)


@dataclass
class RunHandle:
    """Handle returned by :func:`log_run_start`, passed to :func:`log_run_end`."""

    run_id: str
    cmd: str
    db_path: Path
    start_iso: str


class RunEvent(TypedDict):
    """Schema for one row in the ``runs`` table (start/end run records)."""

    event: str  # "start" | "end"
    run_id: str
    cmd: str
    ts: str
    status: NotRequired[str]
    duration_s: NotRequired[float]
    exit: NotRequired[int]
    error: NotRequired[str]


def log_run_start(wiki_dir: str | None = None, *, cmd_tokens: list[str] | None = None) -> RunHandle | None:
    """Insert a ``start`` row for the current command. No-op without a vault.

    ``cmd_tokens`` are the already-parsed command tokens supplied by the CLI
    layer (subcommand + its raw args); the run log never reads ``sys.argv``.
    """
    state_dir = resolve_state_dir(wiki_dir=wiki_dir)
    if state_dir is None:
        return None
    db_path = vault_db_path(state_dir)
    run_id = uuid.uuid4().hex[:8]
    start_iso = iso_local()
    cmd = _command_label(cmd_tokens or [])
    try:
        with open_store(db_path) as conn:
            conn.execute(
                f"INSERT INTO {RUNS_TABLE} (event, run_id, cmd, ts) VALUES (?, ?, ?, ?)",
                ("start", run_id, cmd, start_iso),
            )
    except sqlite3.Error as exc:  # never let logging crash the command
        logger.debug("run-log start write failed: %s", exc)
        return None
    _maybe_prune(db_path)
    return RunHandle(run_id=run_id, cmd=cmd, db_path=db_path, start_iso=start_iso)


def resolve_state_dir(wiki_dir: str | None = None) -> Path | None:
    """Resolve the vault ``.kompend`` dir for the current invocation.

    Parameters
    ----------
    wiki_dir:
        Explicit wiki directory from ``--wiki-dir`` / ``-w``. The
        ``KOMP_WIKI_DIR`` env var is **not** checked here — that is
        handled by Typer's ``WikiDir`` type in the CLI layer.
    """
    cli_overrides = {PATHS_WIKI_DIR: wiki_dir} if wiki_dir else {}
    cfg = resolve_config(cli_overrides=cli_overrides)
    raw = cfg.get(PATHS_STATE_DIR)
    if raw is None:
        return None
    state_dir = Path(str(raw))
    if state_dir.is_dir() and (state_dir / CONFIG_FILE).is_file():
        return state_dir
    return None


def _command_label(cmd_tokens: list[str]) -> str:
    """Render the first few command tokens as the run label."""
    arg_preview_limit = DEFAULTS.get(DISPLAY_CLI_ARG_PREVIEW, 4)
    return " ".join(cmd_tokens[:arg_preview_limit]) or "(none)"


def log_run_end(
    handle: RunHandle | None,
    *,
    status: str,
    exit_code: int | None = None,
    error: str | None = None,
) -> None:
    """Insert the matching ``end`` row for a run started by :func:`log_run_start`."""
    if handle is None:
        return
    duration = (now_local() - _parse_iso(handle.start_iso)).total_seconds()
    capped_error: str | None = None
    if error:
        log_error_limit = DEFAULTS.get(DISPLAY_LOG_ERROR_CHARS, 500)
        capped_error = error[:log_error_limit]
    try:
        with open_store(handle.db_path) as conn:
            conn.execute(
                f"INSERT INTO {RUNS_TABLE} (event, run_id, cmd, ts, status, duration_s, exit, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("end", handle.run_id, handle.cmd, iso_local(), status, round(duration, 2), exit_code, capped_error),
            )
    except sqlite3.Error as exc:
        logger.debug("run-log end write failed: %s", exc)


def _parse_iso(iso: str) -> datetime:
    """Parse an :func:`iso_local` timestamp back to an aware datetime."""
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return now_local()


def _maybe_prune(db_path: Path) -> None:
    """Trim the run log to the last ``_KEEP_ROWS`` once it exceeds the cap."""
    try:
        with open_store(db_path) as conn:
            count = conn.execute(f"SELECT COUNT(*) FROM {RUNS_TABLE}").fetchone()[0]
            if count <= _SOFT_CAP_ROWS:
                return
            conn.execute(
                f"DELETE FROM {RUNS_TABLE} WHERE id NOT IN (SELECT id FROM {RUNS_TABLE} ORDER BY id DESC LIMIT ?)",
                (_KEEP_ROWS,),
            )
    except sqlite3.Error as exc:
        logger.debug("run-log prune failed: %s", exc)
