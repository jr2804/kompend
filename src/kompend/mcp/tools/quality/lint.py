"""MCP lint tools - the deterministic and LLM lint families plus their history.

Nine tools: ``lint_all`` (the full pipeline), ``lint_check`` and
``lint_semantic`` (one family each), ``lint_history``/``lint_show``/
``lint_prune`` (past runs), ``lint_gaps`` (missing pages) and
``lint_remediate``/``lint_format`` (writes).

The lint engine itself lives in ``kompend.lint``; these are the MCP
wrappers that resolve the vault, run the flow and shape the payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kompend.agents.remediate import make_agentic_remediation_agent
from kompend.cli.settings import build_cli_overrides
from kompend.core.cache.wiki import write_wiki_cache
from kompend.core.config import resolve_config
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.models import load_wiki_from_files
from kompend.core.paths import INDEX_FILE, LINT_HISTORY_FILE, OVERVIEW_MD, VaultPaths, vault_paths
from kompend.lint.base import RemediationStatus
from kompend.lint.flow import CHECK_STEP_RULES, _make_lint_agent, persist_lint_fixes, run_full_lint_flow
from kompend.lint.flow import _run_pass as _run_rules_pass
from kompend.lint.gap_analysis import analyze_gaps, gaps_to_text
from kompend.lint.history import (
    LintLlmMeta,
    history_to_json,
    history_to_text,
    load_lint_history,
    persist_history,
    prune_lint_history,
    report_from_dict,
)
from kompend.lint.llm_rules import LLM_RULE_BY_NAME
from kompend.lint.log import latest_lint_log, lint_log_dir, read_lint_log
from kompend.lint.remediation import remediate, remediation_to_text
from kompend.lint.runner import report_to_text
from kompend.logging import get_logger
from kompend.output.renderer import format_output

logger = get_logger(__name__)


# ── lint_all ─────────────────────────────────────────────────────────────


def lint_all(
    wiki_dir: str | None = None,
    fix: bool = True,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Run the full lint pipeline: check, format, semantic, gaps, remediate.

    Deterministic steps weight 1/item, LLM steps weight 100/item on the
    shared progress flow. LLM steps are skipped gracefully when no model
    is configured. Results are appended to .kompend/lint_history.jsonl.
    """
    try:
        vp = vault_paths(wiki_dir)
        cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)

        merged = run_full_lint_flow(vp, cfg, fix=fix)
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 — surface any flow failure to the MCP client
        return {"status": "error", "message": str(exc)}
    return {
        "status": "ok",
        "health_score": merged.health_score,
        "passes": merged.passes,
        "failures": merged.failures,
        "skips": merged.skips,
        "report": merged.model_dump(mode="json"),
    }


def lint_check(
    wiki_dir: str | None = None,
    fix: bool = True,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Run deterministic structural lint rules (no LLM)."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    wiki = load_wiki_from_files(vp.output_dir)
    if wiki is None:
        return {"status": "error", "message": f"No wiki pages found in {vp.output_dir}. Run 'wiki_compile' first."}

    report, elapsed = _run_rules_pass(wiki, vp, cfg, rules=CHECK_STEP_RULES, enable_llm=False, lint_agent=None, fix=fix)
    fixes_applied = 0
    if fix and any(r.fix_applied for r in report.results):
        fixes_applied = persist_lint_fixes(wiki, vp)
    persist_history(report, vp=vp, config=cfg.raw, context="check", elapsed=elapsed, fix=fix, node_count=len(wiki.nodes), llm=None)
    return {
        "status": "ok",
        "health_score": report.health_score,
        "passes": report.passes,
        "total_checks": report.total_checks,
        "fixes_applied": fixes_applied,
        "report_text": report_to_text(report),
    }


def lint_semantic(
    wiki_dir: str | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Run the LLM-powered lint rules (needs compile.lint.model or compile.llm.default_model)."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    wiki = load_wiki_from_files(vp.output_dir)
    if wiki is None:
        return {"status": "error", "message": f"No wiki pages found in {vp.output_dir}. Run 'wiki_compile' first."}

    lint_agent, model = _make_lint_agent(cfg)
    if lint_agent is None:
        return {"status": "error", "message": "LLM lint rules require a model. Set compile.lint.model or compile.llm.default_model."}

    report, elapsed = _run_rules_pass(wiki, vp, cfg, rules=list(LLM_RULE_BY_NAME.keys()), enable_llm=True, lint_agent=lint_agent, fix=False)
    llm_meta = LintLlmMeta(model=model, usage=lint_agent.usage.to_snapshot()) if model is not None and lint_agent.usage is not None else None
    persist_history(report, vp=vp, config=cfg.raw, context="semantic", elapsed=elapsed, fix=False, node_count=len(wiki.nodes), llm=llm_meta)
    return {
        "status": "ok",
        "health_score": report.health_score,
        "passes": report.passes,
        "total_checks": report.total_checks,
        "report_text": report_to_text(report),
    }


# ── lint_history ───────────────────────────────────────────────────────────────────────────

# ── lint_history ──────────────────────────────────────────────────────────


def lint_history(
    wiki_dir: str | None = None,
    last: int = 10,
    trend: bool = False,
    json_output: bool = False,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Query the lint history log."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    state_dir = vp.kompend_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=last)

    if json_output:
        return {"status": "ok", "entries": history_to_json(entries)}
    return {"status": "ok", "entries_text": history_to_text(entries, show_trend=trend)}


def lint_show(
    wiki_dir: str | None = None,
    run: str | None = None,
    json_output: bool = False,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Return the detailed report of a lint run (default: the newest)."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    directory = lint_log_dir(vp.base_dir)
    if run:
        path = directory / f"{run}.{'json' if json_output else 'md'}"
        if not path.exists():
            return {"status": "error", "message": f"No lint report named '{run}' in {directory}."}
    else:
        markdown = latest_lint_log(vp.base_dir)
        if markdown is None:
            return {"status": "error", "message": "No lint report found - run 'kompend lint all' or 'kompend compile' first."}
        path = markdown.with_suffix(".json") if json_output else markdown
    content = read_lint_log(path)
    if json_output:
        return {"status": "ok", "entry": content}
    return {"status": "ok", "report": content}


# ── lint_prune ────────────────────────────────────────────────────────────


def lint_prune(
    wiki_dir: str | None = None,
    keep_last: int | None = None,
    max_age_days: int | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Remove old entries from lint history."""
    if keep_last is None and max_age_days is None:
        return {"status": "error", "message": "At least one of keep_last or max_age_days must be provided."}

    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    state_dir = vp.kompend_dir

    removed = prune_lint_history(
        state_dir / LINT_HISTORY_FILE,
        keep_last=keep_last,
        max_age_days=max_age_days,
    )
    return {"status": "ok", "removed": removed, "message": f"Removed {removed} entr{'y' if removed == 1 else 'ies'}."}


# ── lint_gaps ─────────────────────────────────────────────────────────────


def lint_gaps(
    wiki_dir: str | None = None,
    json_output: bool = False,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Analyze the latest lint run for knowledge gaps and source recommendations."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    state_dir = vp.kompend_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        return {"status": "error", "message": "No lint history found. Run 'lint_all' or compile with --lint first."}

    report_payload = entries[-1].report
    report = report_from_dict(report_payload)
    recs = analyze_gaps(report, display=cfg.kompend.display)

    if json_output:
        return {
            "status": "ok",
            "recommendations": [{"priority": r.priority, "category": r.category, "finding": r.finding, "action": r.action} for r in recs],
            "based_on": {"timestamp": entries[-1].timestamp, "score": report_payload.health_score},
        }
    return {
        "status": "ok",
        "recommendations_text": gaps_to_text(recs),
        "based_on": {"timestamp": entries[-1].timestamp, "score": report_payload.health_score},
    }


# ── lint_remediate ────────────────────────────────────────────────────────


def lint_remediate(
    wiki_dir: str | None = None,
    dry_run: bool = False,
    json_output: bool = False,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Generate missing concept pages for coverage-gap findings."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    output_dir = vp.output_dir
    state_dir = vp.kompend_dir
    base_dir = vp.base_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        return {"status": "error", "message": "No lint history found. Run 'lint_all' or compile with --lint first."}

    report_payload = entries[-1].report
    report = report_from_dict(report_payload)

    wiki = load_wiki_from_files(VaultPaths(base_dir).output_dir)
    if wiki is None:
        return {"status": "error", "message": f"No wiki pages found in {output_dir}. Run 'wiki_compile' first."}

    default_model = cfg.kompend.compile.llm.default_model
    agent = (
        make_agentic_remediation_agent(
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

    applied_count = sum(1 for a in actions if a.status == RemediationStatus.APPLIED)
    if applied_count > 0 and not dry_run:
        write_wiki_cache(wiki, base_dir)

    if json_output:
        return {
            "status": "ok",
            "actions": [
                {
                    "category": a.category,
                    "description": a.description,  # nocheck: MCP response payload
                    "status": a.status.value if hasattr(a.status, "value") else str(a.status),
                    "details": a.details,
                }
                for a in actions
            ],
            "applied": applied_count,
            "dry_run": dry_run,
        }
    return {
        "status": "ok",
        "actions_text": remediation_to_text(actions),
        "applied": applied_count,
        "dry_run": dry_run,
        "based_on": {"timestamp": entries[-1].timestamp, "score": report_payload.health_score},
    }


# ── lint_format ───────────────────────────────────────────────────────────


def lint_format(
    wiki_dir: str | None = None,
    check: bool = False,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Run rumdl fmt over the compiled output."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    rumdl_cfg = resolve_rumdl_config(vp.base_dir, cfg.kompend.compile.output.rumdl_config)
    extra_files: list[Path] = []
    for name in (INDEX_FILE, OVERVIEW_MD):
        p = vp.base_dir / name
        if p.exists():
            extra_files.append(p)
    changed = format_output(
        vp.output_dir,
        flavor=cfg.kompend.compile.output.flavor,
        rumdl_config=rumdl_cfg,
        extra_files=extra_files or None,
        check_only=check,
    )
    if check:
        if changed:
            return {
                "status": "error",
                "message": f"{len(changed)} file(s) need formatting.",
                "changed": [str(c) for c in changed],
            }
        return {"status": "ok", "message": "All generated markdown is formatted."}
    if changed:
        return {"status": "ok", "message": f"Formatted {len(changed)} file(s) with rumdl.", "changed": [str(c) for c in changed]}
    return {"status": "ok", "message": "No formatting changes needed."}
