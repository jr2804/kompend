"""``kompend log`` — regenerate the pipeline log report on demand.

The human-readable ``log.md`` is no longer auto-written during builds.
This command rebuilds it from the structured stores (the ``runs``,
``lint_runs``, and ``wiki_snapshot`` tables in ``.kompend/kompend.db``) and prints it, or
writes it to ``.kompend/log.md`` with ``--write``.

The rendering lives in :mod:`kompend.core.log_report` so it is callable from
the API (``build_log_report``); this module is a thin Typer wrapper.
"""

from __future__ import annotations

import typer

from kompend.cli._shared import build_cli_overrides, vault_paths_from_ctx
from kompend.cli.app import app
from kompend.core.config import resolve_config
from kompend.core.reporting.log import build_log_report
from kompend.logging import get_console


@app.command(name="log")
def log_cmd(
    ctx: typer.Context,
    last: int = typer.Option(
        25,
        "--last",
        "-n",
        help="Maximum number of command/lint entries to include.",
    ),
    write: bool = typer.Option(
        False,
        "--write",
        help="Write the report to .kompend/log.md instead of printing it.",
    ),
    config: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a kompend.toml config file.",
    ),
) -> None:
    """Regenerate the pipeline log from structured stores (no LLM calls).

    Aggregates the last compile, the command timeline (runs.jsonl), and the
    lint trend (lint_history.jsonl) into a single report. Print it (default)
    or persist it to ``.kompend/log.md`` with ``--write``.
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    base_dir = vp.base_dir

    report = build_log_report(base_dir, last_n=_last_n(last))

    if write:
        log_file = vp.log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: overwrite rather than prepend — the report is always
        # rebuilt fresh from the structured stores, so the file mirrors the
        # current state.
        log_file.write_text(report, encoding="utf-8")
        get_console().print(f"Wrote log report to {log_file}")
    else:
        get_console().print(report, markup=False)


def _last_n(last: int | None) -> int:
    return last if (last is not None and last > 0) else 25
