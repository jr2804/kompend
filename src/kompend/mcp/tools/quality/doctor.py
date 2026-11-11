"""MCP doctor tool - configuration and connectivity report for a wiki."""

from __future__ import annotations

from typing import Any

from kompend.cli.settings import build_cli_overrides
from kompend.core.config import resolve_config
from kompend.core.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_VLM_MODEL,
)
from kompend.core.paths import vault_paths
from kompend.core.reporting.last_run import build_last_run_report
from kompend.core.utils.dicts import as_str
from kompend.diagnostics import run_doctor
from kompend.mcp.tools._helpers import ToolStatus
from kompend.rag import recorded_embedding_model


def doctor(
    wiki_dir: str | None = None,
    verbose: bool = False,
    json_output: bool = False,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Validate configuration and connectivity for a wiki."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir
    embed_model = as_str(cfg.get(COMPILE_EMBEDDING_MODEL))

    if not wiki_dir and not output_dir.exists():
        return {"status": ToolStatus.ERROR, "message": f"No wiki found at {output_dir}. Use wiki_dir parameter."}
    results = run_doctor(
        sources_dir=sources_dir,
        output_dir=output_dir,
        default_model=as_str(cfg.get(COMPILE_LLM_DEFAULT_MODEL)),
        vlm_model=as_str(cfg.get(COMPILE_LLM_VLM_MODEL)),
        embed_model=embed_model,
        recorded_embed_model=recorded_embedding_model(vp.kompend_db_file) if embed_model else None,
        config=cfg,
        verbose=verbose,
    )
    # We can't easily capture print_doctor_results output, so return the raw results
    last_run = build_last_run_report(vp.base_dir).to_dict()
    if json_output:
        return {"status": ToolStatus.OK, "results": [r.to_dict() for r in results], "last_run": last_run}
    # Build a text summary
    lines = []
    all_ok = True
    for r in results:
        if not r.passed:
            all_ok = False
        lines.append(f"{r.name}: {'OK' if r.passed else 'FAIL'}")
        if verbose and r.message:
            lines.append(f"  {r.message}")
    return {"status": ToolStatus.OK if all_ok else ToolStatus.ERROR, "summary": "\n".join(lines), "last_run": last_run}
