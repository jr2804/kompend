"""``knox doctor`` — validate configuration and connectivity."""

from __future__ import annotations

import typer

from knox.cli._shared import build_cli_overrides, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import ConfigPath, JsonOutput, Verbose
from knox.cli.cmds.quality.doctor import print_doctor_results, run_doctor
from knox.core.config import resolve_config
from knox.core.config.keys import COMPILE_EMBEDDING_MODEL, COMPILE_LLM_DEFAULT_MODEL, COMPILE_LLM_VLM_MODEL
from knox.logging import get_err_console


@app.command(name="doctor")
def doctor_cmd(
    ctx: typer.Context,
    wiki_dir: str | None = typer.Option(
        None,
        "--wiki-dir",
        "-w",
        help="Project root directory. Default: current working directory or KNOX_WIKI_DIR.",
        envvar="KNOX_WIKI_DIR",
    ),
    verbose: Verbose = False,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Validate configuration and connectivity for a wiki.

    Requires a wiki context: either ``--wiki-dir`` must be set explicitly,
    or the current directory must contain a compiled wiki (i.e. the
    resolved output directory must exist).
    """
    wiki_dir_override = wiki_dir or (ctx.obj.get("wiki_dir") if ctx.obj else None)
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override), config_path=config)
    sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    if not wiki_dir_override and not output_dir.exists():
        get_err_console().print(
            f"No wiki found at {output_dir}. Use --wiki-dir or run from a wiki directory.",
        )
        raise typer.Exit(1)

    results = run_doctor(
        sources_dir=sources_dir,
        output_dir=output_dir,
        default_model=cfg.get(COMPILE_LLM_DEFAULT_MODEL),
        vlm_model=cfg.get(COMPILE_LLM_VLM_MODEL),
        embed_model=cfg.get(COMPILE_EMBEDDING_MODEL),
        cfg=cfg.raw,
        verbose=verbose,
    )
    exit_code = print_doctor_results(results, json_output=json_output)
    if exit_code:
        raise typer.Exit(1)
