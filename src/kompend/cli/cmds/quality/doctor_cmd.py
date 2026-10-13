"""``kompend doctor`` — validate configuration and connectivity."""

from __future__ import annotations

import typer

from kompend.cli._shared import build_cli_overrides, vault_paths_from_ctx
from kompend.cli.app import app
from kompend.cli.args import ConfigPath, JsonOutput, Verbose
from kompend.cli.cmds.quality.doctor import print_doctor_results, run_doctor
from kompend.core.config import resolve_config
from kompend.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_VLM_MODEL,
    EnvVar,
)
from kompend.core.paths import vault_paths
from kompend.core.utils.dicts import as_str
from kompend.logging import get_err_console


@app.command(name="doctor")
def doctor_cmd(
    ctx: typer.Context,
    wiki_dir: str | None = typer.Option(
        None,
        "--wiki-dir",
        "-w",
        help="Project root directory. Default: current working directory or KOMP_WIKI_DIR.",
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
        default_model=as_str(cfg.get(COMPILE_LLM_DEFAULT_MODEL)),
        vlm_model=as_str(cfg.get(COMPILE_LLM_VLM_MODEL)),
        embed_model=as_str(cfg.get(COMPILE_EMBEDDING_MODEL)),
        cfg=cfg.raw,
        verbose=verbose,
    )
    exit_code = print_doctor_results(results, json_output=json_output)
    if exit_code:
        raise typer.Exit(1)
