"""Generate the pages that coverage-gap findings name."""

from __future__ import annotations

import json

from kompend.agents.remediate import make_remediation_agent
from kompend.cli.args import JsonOutput, LintDryRun
from kompend.cli.settings import build_cli_overrides, config_file_from_ctx, vault_paths_from_ctx
from kompend.core.cache.wiki import write_wiki_cache
from kompend.core.config import resolve_config
from kompend.core.models import load_cached_wiki
from kompend.core.paths import LINT_HISTORY_FILE
from kompend.lint.base import RemediationStatus
from kompend.lint.details import details_for_storage
from kompend.lint.history import load_lint_history, report_from_dict
from kompend.lint.remediation import remediate, remediation_to_text
from kompend.logging import get_console, get_err_console

__all__ = ["lint_remediate"]


def lint_remediate(
    *,
    dry_run: LintDryRun = False,
    json_output: JsonOutput = False,
) -> None:
    """Generate missing concept pages for coverage-gap findings.

    Uses ``compile.llm.default_model`` from config. Coverage-gap findings
    are matched against cached phase-1 extractions; matched concepts get
    concept pages generated from that source material. Unmatched concepts
    are skipped — ``kompend lint gaps`` shows recommendations for those.

    Examples:
        kompend lint remediate                          # generate pages
        kompend lint remediate --dry-run                # preview without writing
    """
    vp = vault_paths_from_ctx()
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config_file_from_ctx(),
    )
    output_dir = vp.output_dir
    state_dir = vp.kompend_dir
    base_dir = vp.base_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        get_err_console().print("No lint history found. Run `kompend lint run` or compile with --lint first.")
        raise SystemExit(1)

    report_payload = entries[-1].report
    report = report_from_dict(report_payload)

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'kompend compile' first.")
        raise SystemExit(1)

    default_model = cfg.kompend.compile.llm.default_model
    agent = (
        make_remediation_agent(
            model_string=default_model,
            timeout=float(cfg.kompend.compile.llm.timeout),
            llm=cfg.kompend.compile.llm,
            memory_dir=vp.agent_memory_dir,
        )
        if default_model
        else None
    )

    actions = remediate(
        wiki,
        report,
        base_dir=base_dir,
        output_dir=output_dir,
        agent=agent,
        dry_run=dry_run,
    )

    # Persist changes unless dry-run
    applied_count = sum(1 for a in actions if a.status == RemediationStatus.APPLIED)
    if applied_count > 0 and not dry_run:
        write_wiki_cache(wiki, base_dir)
        get_console().print(f"Persisted {applied_count} remediation(s) to wiki cache.\n")

    if json_output:
        get_console().print(
            json.dumps(
                [
                    {
                        "category": a.category,
                        "description": a.description,  # nocheck: CLI JSON output payload
                        "status": a.status,
                        "details": details_for_storage(a.details),
                    }
                    for a in actions
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        ts = entries[-1].timestamp
        score = report_payload.health_score
        get_console().print(f"\nBased on lint run from {ts} (score: {score:.0f}%)\n")
        get_console().print(remediation_to_text(actions))
