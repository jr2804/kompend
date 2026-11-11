"""Append-only audit trail of every CLI command run (the ``runs`` table).

Each command inserts a ``start`` row on entry and an ``end`` row on exit,
linked by a short ``run_id``. Querying ``runs`` answers "did the build run
through, and if not, where did it stop?":

- an ``end`` row with ``status="ok"``  → command finished cleanly
- an ``end`` row with ``status="error"`` → command crashed (error captured)
- a ``start`` row with no matching ``end`` → process was killed / hung

Rows are tiny; the table is pruned to a soft cap so the store stays bounded.

The three statements are SQLAlchemy Core statements against the shared ``runs``
table object, so the column names live in one place.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict

from sqlalchemy import delete, func, insert, select

from kompend.core.config import ResolvedConfig, resolve_config
from kompend.core.config._configs import DisplayConfig
from kompend.core.keys import (
    DISPLAY_CLI_ARG_PREVIEW,
    DISPLAY_LOG_ERROR_CHARS,
    PATHS_STATE_DIR,
    PATHS_WIKI_DIR,
)
from kompend.core.paths import CONFIG_FILE, LOGS_SUBDIR
from kompend.core.reporting.handle import RunHandle, set_current_run_id
from kompend.core.store import core_connection, open_store, vault_db_path
from kompend.core.store.tables import runs
from kompend.core.utils.time import iso_local, now_local
from kompend.logging import attach_run_log, detach_run_log, get_logger
from kompend.types import ConfigValue

_SOFT_CAP_ROWS = 1000
_KEEP_ROWS = 500

logger = get_logger(__name__)


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
    state_dir, display = _resolve_run_context(wiki_dir)
    if state_dir is None:
        return None
    db_path = vault_db_path(state_dir)
    run_id = uuid.uuid4().hex[:8]
    start_iso = iso_local()
    cmd = _command_label(cmd_tokens or [], display)
    try:
        with open_store(db_path) as conn, core_connection(conn) as core:
            core.execute(insert(runs).values(event="start", run_id=run_id, cmd=cmd, ts=start_iso))
    except sqlite3.Error as exc:  # never let logging crash the command
        logger.debug("run-log start write failed: %s", exc)
        return None
    _maybe_prune(db_path)
    # The ledger (and any other per-run writer) correlates its rows with this
    # run through the handle module's context — set once, here.
    set_current_run_id(run_id)
    # Flight recorder: INFO+ straight to the run log, the DEBUG ring dumped
    # on failure (kompend.logging.RunLogHandler). The path rides the handle
    # so the CLI footer can name it. The log lives next to the state dir's
    # own artifacts, so a custom ``paths.state_dir`` keeps its log with it.
    handler = attach_run_log(run_id, state_dir / LOGS_SUBDIR)
    logger.info("run %s started: %s", run_id, cmd)
    return RunHandle(
        run_id=run_id,
        cmd=cmd,
        db_path=db_path,
        start_iso=start_iso,
        error_chars=display.log_error_chars,
        log_file=handler.path,
    )


def resolve_state_dir(wiki_dir: str | None = None) -> Path | None:
    """Resolve the vault ``.kompend`` dir for the current invocation.

    Parameters
    ----------
    wiki_dir:
        Explicit wiki directory from ``--wiki-dir`` / ``-w``. The
        ``KOMP_WIKI_DIR`` env var is **not** checked here — that is
        handled by Typer's ``WikiDir`` type in the CLI layer.
    """
    state_dir, _ = _resolve_run_context(wiki_dir)
    return state_dir


def _resolve_run_context(wiki_dir: str | None = None) -> tuple[Path | None, DisplayConfig]:
    """Resolve the vault state dir and the display limits for this run.

    The display limits are read from the *merged* config, not from the
    validated model: the run log is written for every command, so a vault
    whose config does not validate yet (``kompend config path`` on a stale
    config) must still run and still be logged.
    """
    cli_overrides = {PATHS_WIKI_DIR: wiki_dir} if wiki_dir else {}
    cfg = resolve_config(cli_overrides=cli_overrides)
    raw = cfg.get(PATHS_STATE_DIR)
    display = _display_limits(cfg)
    if raw is None:
        return None, display
    state_dir = Path(str(raw))
    if state_dir.is_dir() and (state_dir / CONFIG_FILE).is_file():
        return state_dir, display
    return None, display


def _display_limits(cfg: ResolvedConfig) -> DisplayConfig:
    """Read the run-log display limits from the resolved config.

    Unset or non-numeric values fall back to the schema defaults — the run
    log never fails because of a config value.
    """
    defaults = DisplayConfig()
    return defaults.model_copy(
        update={
            "cli_arg_preview": _int_or(cfg.get(DISPLAY_CLI_ARG_PREVIEW), defaults.cli_arg_preview),
            "log_error_chars": _int_or(cfg.get(DISPLAY_LOG_ERROR_CHARS), defaults.log_error_chars),
        }
    )


def _int_or(value: ConfigValue, fallback: int) -> int:
    """Return *value* when it is an int, else *fallback*."""
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _command_label(cmd_tokens: list[str], display: DisplayConfig) -> str:
    """Render the first few command tokens as the run label."""
    return " ".join(cmd_tokens[: display.cli_arg_preview]) or "(none)"


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
    # The recorder's own bookends: even a run that logs nothing else yields a
    # file naming the command, its outcome and its duration — the footer points
    # at a log that always explains itself.
    logger.info("run %s ended: status=%s duration=%.2fs", handle.run_id, status, duration)
    # Flight recorder: dump the DEBUG ring before detaching, while the file is
    # still open — anything but status "ok" (a crash or a user cancel) counts
    # as "did not finish clean" and keeps its debug trail.
    detach_run_log(failed=status != "ok", reason=f"run ended with status={status!r}")
    capped_error: str | None = None
    if error:
        limit = handle.error_chars if handle.error_chars is not None else DisplayConfig().log_error_chars
        capped_error = error[:limit]
    try:
        with open_store(handle.db_path) as conn, core_connection(conn) as core:
            core.execute(
                insert(runs).values(
                    event="end",
                    run_id=handle.run_id,
                    cmd=handle.cmd,
                    ts=iso_local(),
                    status=status,
                    duration_s=round(duration, 2),
                    exit=exit_code,
                    error=capped_error,
                )
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
        with open_store(db_path) as conn, core_connection(conn) as core:
            count = core.execute(select(func.count()).select_from(runs)).scalar_one()
            if count <= _SOFT_CAP_ROWS:
                return
            kept = select(runs.c.id).order_by(runs.c.id.desc()).limit(_KEEP_ROWS)
            core.execute(delete(runs).where(runs.c.id.not_in(kept)))
    except sqlite3.Error as exc:
        logger.debug("run-log prune failed: %s", exc)
