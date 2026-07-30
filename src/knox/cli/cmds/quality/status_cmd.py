"""``knox status`` — show per-wiki statistics."""

from __future__ import annotations

import typer

from knox.cli._shared import build_cli_overrides, resolve_project_paths, vault_paths_from_ctx
from knox.cli.app import app
from knox.cli.args import JsonOutput
from knox.cli.cmds.pipeline.status import collect_stats, print_status
from knox.core.config import resolve_config


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
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=str(vp.base_dir)))
    sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)
    stats = collect_stats(
        sources_dir=sources_dir if sources_dir.exists() else None,
        output_dir=output_dir if output_dir.exists() else None,
        state_dir=state_dir if state_dir.exists() else None,
    )
    print_status(stats, json_output=json_output)
