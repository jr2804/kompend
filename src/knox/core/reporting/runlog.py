"""Append-only JSONL audit trail of every CLI command run.

Each command writes a ``start`` line on entry and an ``end`` line on exit,
linked by a short ``run_id``. Tailing ``.knox/runs.jsonl`` answers "did the
build run through, and if not, where did it stop?":

- an ``end`` line with ``status="ok"``  → command finished cleanly
- an ``end`` line with ``status="error"`` → command crashed (error captured)
- a ``start`` line with no matching ``end`` → process was killed / hung

Entries are tiny (one JSON object per line); the file is pruned to a soft
cap so ``.knox`` stays bounded.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from knox.core.config import DEFAULTS, resolve_config
from knox.core.config.keys import DISPLAY_CLI_ARG_PREVIEW, DISPLAY_LOG_ERROR_CHARS, PATHS_STATE_DIR, PATHS_WIKI_DIR
from knox.core.paths import CONFIG_FILE, RUNS_FILE
from knox.core.utils.time import iso_local, now_local
from knox.logging import get_logger

logger = get_logger(__name__)

_SOFT_CAP_LINES = 1000
_KEEP_LINES = 500


@dataclass
class RunHandle:
    """Handle returned by :func:`log_run_start`, passed to :func:`log_run_end`."""

    run_id: str
    cmd: str
    runs_path: Path
    start_iso: str


def log_run_start(wiki_dir: str | None = None) -> RunHandle | None:
    """Append a ``start`` line for the current command. No-op without a vault."""
    state_dir = resolve_state_dir(wiki_dir=wiki_dir)
    if state_dir is None:
        return None
    runs_path = state_dir / RUNS_FILE
    run_id = uuid.uuid4().hex[:8]
    start_iso = iso_local()
    try:
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        with runs_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "event": "start",
                        "run_id": run_id,
                        "cmd": _command_label(),
                        "ts": start_iso,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError as exc:  # never let logging crash the command
        logger.debug("run-log start write failed: %s", exc)
        return None
    _maybe_prune(runs_path)
    return RunHandle(run_id=run_id, cmd=_command_label(), runs_path=runs_path, start_iso=start_iso)


def resolve_state_dir(wiki_dir: str | None = None) -> Path | None:
    """Resolve the vault ``.knox`` dir for the current invocation.

    Parameters
    ----------
    wiki_dir:
        Explicit wiki directory from ``--wiki-dir`` / ``-w``. The
        ``KNOX_WIKI_DIR`` env var is **not** checked here — that is
        handled by Typer's ``WikiDir`` type in the CLI layer.
    """
    cli_overrides = {PATHS_WIKI_DIR: wiki_dir} if wiki_dir else {}
    cfg = resolve_config(cli_overrides=cli_overrides)
    state_dir = Path(cfg.get(PATHS_STATE_DIR))
    if state_dir.is_dir() and (state_dir / CONFIG_FILE).is_file():
        return state_dir
    return None


def _command_label() -> str:
    """Human-readable command label, e.g. ``lint run`` or ``compile``.

    Strips the global ``--wiki-dir`` flags so the label is just the command.
    """
    cleaned: list[str] = []
    skip_next = False
    for t in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if t in ("--wiki-dir", "-w"):
            skip_next = True
            continue
        if t.startswith("--wiki-dir="):
            continue
        cleaned.append(t)
    arg_preview_limit = DEFAULTS.get(DISPLAY_CLI_ARG_PREVIEW, 4)
    return " ".join(cleaned[:arg_preview_limit]) or "(none)"


def log_run_end(
    handle: RunHandle | None,
    *,
    status: str,
    exit_code: int | None = None,
    error: str | None = None,
) -> None:
    """Append the matching ``end`` line for a run started by :func:`log_run_start`."""
    if handle is None:
        return
    duration = (now_local() - _parse_iso(handle.start_iso)).total_seconds()
    payload: dict[str, Any] = {
        "event": "end",
        "run_id": handle.run_id,
        "cmd": handle.cmd,
        "status": status,
        "duration_s": round(duration, 2),
        "ts": iso_local(),
    }
    if exit_code is not None:
        payload["exit"] = exit_code
    if error:
        log_error_limit = DEFAULTS.get(DISPLAY_LOG_ERROR_CHARS, 500)
        payload["error"] = error[:log_error_limit]
    try:
        with handle.runs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.debug("run-log end write failed: %s", exc)


def _parse_iso(iso: str) -> datetime:
    """Parse an :func:`iso_local` timestamp back to an aware datetime."""
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return now_local()


def _maybe_prune(runs_path: Path) -> None:
    """Trim the run log to the last ``_KEEP_LINES`` once it exceeds the cap."""
    try:
        line_count = sum(1 for _ in runs_path.open(encoding="utf-8"))
    except OSError:
        return
    if line_count <= _SOFT_CAP_LINES:
        return
    try:
        with runs_path.open(encoding="utf-8") as fh:
            lines = fh.readlines()
        with runs_path.open("w", encoding="utf-8") as fh:
            fh.writelines(lines[-_KEEP_LINES:])
    except OSError as exc:
        logger.debug("run-log prune failed: %s", exc)
