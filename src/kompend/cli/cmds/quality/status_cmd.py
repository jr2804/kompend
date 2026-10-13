"""``kompend status`` — show per-wiki statistics."""

from __future__ import annotations

import typer

from kompend.cli._shared import build_cli_overrides, vault_paths_from_ctx
from kompend.cli.app import app
from kompend.cli.args import JsonOutput
from kompend.cli.cmds.pipeline.status import collect_stats, print_status
from kompend.core.config import resolve_config


@app.command(name="status")
def status_cmd(
    ctx: typer.Context,
    json_output: JsonOutput = False,
) -> None:
    """Show per-wiki statistics.

    Displays source count, article count, word count, last compile time,
    and config path.
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    stats = collect_stats(
        sources_dir=vp.sources_dir if vp.sources_dir.exists() else None,
        output_dir=vp.output_dir if vp.output_dir.exists() else None,
        state_dir=vp.kompend_dir if vp.kompend_dir.exists() else None,
    )
    print_status(stats, json_output=json_output)
