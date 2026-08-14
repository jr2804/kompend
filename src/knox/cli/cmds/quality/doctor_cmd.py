"""``knox doctor`` — validate configuration and connectivity."""

from __future__ import annotations

import typer

from knox.cli._shared import build_cli_overrides, vault_paths_from_ctx
from knox.cli.app import app
from knox.cli.args import ConfigPath, JsonOutput, Verbose
from knox.cli.cmds.quality.doctor import print_doctor_results, run_doctor
from knox.core.config import resolve_config
from knox.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_VLM_MODEL,
    EnvVar,
)
from knox.core.paths import vault_paths
from knox.logging import get_err_console


@app.command(name="doctor")
def doctor_cmd(
    ctx: typer.Context,
    wiki_dir: str | None = typer.Option(
        None,
        "--wiki-dir",
        "-w",
        help="Project root directory. Default: current working directory or KNOX_WIKI_DIR.",
        envvar=EnvVar.WIKI_DIR,
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
    vp = vault_paths_from_ctx(ctx)
    if wiki_dir:
        vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir

    if not wiki_dir and not ctx.obj.get("wiki_dir") and not output_dir.exists():
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
