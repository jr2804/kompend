"""MCP log tool - regenerate the pipeline log from structured stores."""

from __future__ import annotations

from typing import Any

from kompend.cli.settings import build_cli_overrides
from kompend.core.config import resolve_config
from kompend.core.paths import vault_paths
from kompend.core.reporting.log import build_log_report


def log(
    wiki_dir: str | None = None,
    last: int = 25,
    write: bool = False,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Regenerate the pipeline log from structured stores."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    base_dir = vp.base_dir
    report = build_log_report(base_dir, last_n=last if (last is not None and last > 0) else 25)

    if write:
        log_file = vp.log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(report, encoding="utf-8")
        return {"status": "ok", "message": f"Wrote log report to {log_file}"}
    return {"status": "ok", "report": report}
