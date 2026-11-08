"""MCP explain tool - what the pipeline did to one source file."""

from __future__ import annotations

from typing import Any

from kompend.cli.settings import build_cli_overrides
from kompend.core.config import resolve_config
from kompend.core.paths import vault_paths
from kompend.core.reporting.explain import AmbiguousSourceError, build_explain_report, render_explain


def explain(
    source: str,
    wiki_dir: str | None = None,
    run_id: str | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Explain what the pipeline did to one source file: stage timings, facts and degradations."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    try:
        report = build_explain_report(vp.base_dir, source, run_id=run_id)
    except AmbiguousSourceError as exc:
        return {"status": "error", "message": str(exc), "candidates": exc.candidates}
    if report is None:
        return {"status": "error", "message": f"No ledger rows found for {source!r}."}
    return {"status": "ok", "report": report, "report_text": render_explain(report)}
