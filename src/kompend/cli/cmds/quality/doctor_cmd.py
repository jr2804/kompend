"""``kompend doctor`` — validate configuration and connectivity."""

from __future__ import annotations

from kompend.cli.args import JsonOutput, Verbose, WikiDir
from kompend.cli.cmds.quality.doctor import print_doctor_results
from kompend.cli.context import current_run_context
from kompend.cli.settings import (
    build_cli_overrides,
    config_file_from_ctx,
    vault_paths_from_ctx,
)
from kompend.core.config import resolve_config
from kompend.core.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_VLM_MODEL,
)
from kompend.core.paths import vault_paths
from kompend.core.utils.dicts import as_str
from kompend.diagnostics import run_doctor
from kompend.logging import get_err_console


def doctor_cmd(
    *,
    wiki_dir: WikiDir = None,
    verbose: Verbose = False,
    json_output: JsonOutput = False,
) -> None:
    """Validate configuration and connectivity for a wiki.

    Requires a wiki context: either ``--wiki-dir`` must be set explicitly,
    or the current directory must contain a compiled wiki (i.e. the
    resolved output directory must exist).
    """
    vp = vault_paths_from_ctx()
    if wiki_dir:
        vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file_from_ctx())
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir

    if not wiki_dir and not current_run_context().wiki_dir and not output_dir.exists():
        get_err_console().print(
            f"No wiki found at {output_dir}. Use --wiki-dir or run from a wiki directory.",
        )
        raise SystemExit(1)

    results = run_doctor(
        sources_dir=sources_dir,
        output_dir=output_dir,
        default_model=as_str(cfg.get(COMPILE_LLM_DEFAULT_MODEL)),
        vlm_model=as_str(cfg.get(COMPILE_LLM_VLM_MODEL)),
        embed_model=as_str(cfg.get(COMPILE_EMBEDDING_MODEL)),
        config=cfg,
        verbose=verbose,
    )
    exit_code = print_doctor_results(results, json_output=json_output)
    if exit_code:
        raise SystemExit(1)
