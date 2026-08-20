"""Quality & diagnostics MCP tools (lint, doctor, review, trace, log, list-parsers)."""

import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from knox.agents.lint import LintRuleAgent
from knox.agents.remediate import RemediationAgent
from knox.cli._post_compile import write_wiki_cache
from knox.cli._shared import build_cli_overrides
from knox.cli.cmds.quality.doctor import run_doctor
from knox.core.config import resolve_config
from knox.core.config._configs import LintConfig
from knox.core.config.rumdl import resolve_rumdl_config
from knox.core.models import CompiledWiki, MetadataField, load_cached_wiki
from knox.core.paths import INDEX_FILE, LINT_HISTORY_FILE, REVIEW_DIR, vault_paths
from knox.core.reporting.log import build_log_report
from knox.lint.base import LintReport, LintReportResult, RemediationStatus
from knox.lint.gap_analysis import analyze_gaps, gaps_to_text
from knox.lint.history import (
    entry_from_report,
    history_to_json,
    history_to_text,
    load_lint_history,
    prune_lint_history,
    save_lint_entry,
)
from knox.lint.llm_rules import LLM_RULE_BY_NAME
from knox.lint.remediation import remediate, remediation_to_text
from knox.lint.runner import LintRunner, report_to_json, report_to_text
from knox.logging import get_logger
from knox.output.pages.log import log_stage
from knox.output.renderer import format_output
from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers
from knox.review import ReviewQueue

logger = get_logger(__name__)


# ── lint_run ──────────────────────────────────────────────────────────────


def lint_run(
    wiki_dir: str | None = None,
    fix: bool = False,
    llm: bool = True,
    rule: list[str] | None = None,
    json_output: bool = False,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Run structural (and optionally LLM-powered) lint rules.

    Results are appended to .knox/lint_history.jsonl.
    """
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir
    state_dir = vp.knox_dir
    sources_dir_str = str(sources_dir) if sources_dir.exists() else None

    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        return {"status": "error", "message": f"No compiled wiki found in {output_dir}. Run 'wiki_compile' first."}

    default_model = cfg.knox.compile.llm.default_model
    lint_agent = (
        LintRuleAgent(default_model, timeout=float(cfg.knox.compile.llm.timeout), retries=cfg.knox.compile.llm.output_retries) if default_model else None
    )
    if _uses_llm_rule(rule) and lint_agent is None:
        return {"status": "error", "message": "LLM-powered lint rules require a model. Set compile.llm.default_model in config."}
    if llm and lint_agent is None:
        llm = False

    lint_config = LintConfig(
        fuzzy_threshold=float(cfg.knox.lint.fuzzy_threshold),
    )
    runner = LintRunner(
        rules=list(rule) if rule else None,
        enable_llm=llm,
        lint_agent=lint_agent,
        fix=fix,
        sources_dir=sources_dir_str,
        lint_config=lint_config,
        rumdl_config=resolve_rumdl_config(vp.base_dir, cfg.knox.compile.output.rumdl_config),
    )
    lint_start = time.monotonic()
    report = runner.run(wiki)
    lint_elapsed = time.monotonic() - lint_start

    fixes_applied = 0
    if fix and any(r.fix_applied for r in report.results):
        fixes_applied = _persist_lint_fixes(wiki, output_dir, vp.base_dir)

    outcome_counts = Counter(r.outcome.value if hasattr(r.outcome, "value") else str(r.outcome) for r in report.results)

    llm_meta = None
    if lint_agent is not None and lint_agent.usage is not None:
        llm_meta = {
            "model": default_model,
            "usage": lint_agent.usage.to_dict(),
        }

    options = {
        "rules": list(rule) if rule else None,
        "enable_llm": llm,
        "fix": fix,
    }
    entry = entry_from_report(
        report,
        duration_seconds=lint_elapsed,
        options=options,
        config=cfg.raw,
        node_count=len(wiki.nodes),
        llm=llm_meta,
    )
    save_lint_entry(entry, state_dir / LINT_HISTORY_FILE)

    log_stage(
        vp,
        action="Lint",
        detail=f"Health score {report.health_score:.1f}% ({report.passes}/{report.total_checks} checks passed)",
    )

    result = {
        "status": "ok",
        "health_score": report.health_score,
        "passes": report.passes,
        "total_checks": report.total_checks,
        "outcome_counts": dict(outcome_counts),
        "fixes_applied": fixes_applied,
    }
    if json_output:
        result["report"] = report_to_json(report)
    else:
        result["report_text"] = report_to_text(report)
    return result


def _uses_llm_rule(rule: list[str] | None) -> bool:
    if not rule:
        return False
    return any(r in LLM_RULE_BY_NAME for r in rule)


def _persist_lint_fixes(
    wiki: CompiledWiki,
    output_dir: Path,
    base_dir: Path,
) -> int:
    updated = 0
    for node in wiki.nodes:
        fm = node.metadata.get("frontmatter", {})
        if not isinstance(fm, dict) or "title" not in fm:
            continue
        md_file = output_dir / f"node_{node.id}.md"
        if not md_file.exists():
            continue
        text = md_file.read_text(encoding="utf-8")
        normalized = text.replace("\r\n", "\n")
        if normalized.startswith("---\n"):
            end_idx = normalized.find("\n---\n", 4)
            if end_idx < 0:
                continue
            old_fm_text = normalized[4:end_idx]
            body = normalized[end_idx + 5 :]
            old_fm = yaml.safe_load(old_fm_text) or {}
            old_fm.update(fm)
            new_fm = yaml.dump(old_fm, default_flow_style=False, sort_keys=False, allow_unicode=True)
            md_file.write_text(f"---\n{new_fm}---\n{body}", encoding="utf-8")
            updated += 1
    write_wiki_cache(wiki, base_dir)
    return updated


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
    state_dir = vp.knox_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=last)

    if json_output:
        return {"status": "ok", "entries": history_to_json(entries)}
    return {"status": "ok", "entries_text": history_to_text(entries, show_trend=trend)}


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
    state_dir = vp.knox_dir

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
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file)
    state_dir = vp.knox_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        return {"status": "error", "message": "No lint history found. Run 'lint_run' or compile with --lint first."}

    report_dict = entries[-1].get("report", {})
    report = _to_lint_report(report_dict)
    recs = analyze_gaps(report)

    if json_output:
        return {
            "status": "ok",
            "recommendations": [{"priority": r.priority, "category": r.category, "finding": r.finding, "action": r.action} for r in recs],
            "based_on": {"timestamp": entries[-1].get("timestamp", "unknown"), "score": report_dict.get("health_score", 0.0)},
        }
    return {
        "status": "ok",
        "recommendations_text": gaps_to_text(recs),
        "based_on": {"timestamp": entries[-1].get("timestamp", "unknown"), "score": report_dict.get("health_score", 0.0)},
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
    state_dir = vp.knox_dir
    base_dir = vp.base_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        return {"status": "error", "message": "No lint history found. Run 'lint_run' or compile with --lint first."}

    report_dict = entries[-1].get("report", {})
    report = _to_lint_report(report_dict)

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        return {"status": "error", "message": f"No compiled wiki found in {output_dir}. Run 'wiki_compile' first."}

    default_model = cfg.knox.compile.llm.default_model
    agent = RemediationAgent(default_model, timeout=float(cfg.knox.compile.llm.timeout)) if default_model else None

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
                    "description": a.description,
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
        "based_on": {"timestamp": entries[-1].get("timestamp", "unknown"), "score": report_dict.get("health_score", 0.0)},
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
    rumdl_cfg = resolve_rumdl_config(vp.base_dir, cfg.knox.compile.output.rumdl_config)
    extra_files: list[Path] = []
    for name in (INDEX_FILE, "overview.md"):
        p = vp.base_dir / name
        if p.exists():
            extra_files.append(p)
    changed = format_output(
        vp.output_dir,
        flavor=cfg.knox.compile.output.flavor,
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


# ── doctor ────────────────────────────────────────────────────────────────


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

    if not wiki_dir and not output_dir.exists():
        return {"status": "error", "message": f"No wiki found at {output_dir}. Use wiki_dir parameter."}
    results = run_doctor(
        sources_dir=sources_dir,
        output_dir=output_dir,
        default_model=cfg.get("compile.llm.default_model"),
        vlm_model=cfg.get("compile.llm.vlm_model"),
        embed_model=cfg.get("compile.embedding.model"),
        cfg=cfg.raw,
        verbose=verbose,
    )
    # We can't easily capture print_doctor_results output, so return the raw results
    if json_output:
        return {"status": "ok", "results": [r.to_dict() for r in results]}
    # Build a text summary
    lines = []
    all_ok = True
    for r in results:
        if not r.passed:
            all_ok = False
        lines.append(f"{r.name}: {'OK' if r.passed else 'FAIL'}")
        if verbose and r.message:
            lines.append(f"  {r.message}")
    return {"status": "ok" if all_ok else "error", "summary": "\n".join(lines)}


# ── review ────────────────────────────────────────────────────────────────


def review(
    wiki_dir: str | None = None,
    action: str = "list",
    slug: str | None = None,
    reason: str | None = None,
    approve_all: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    """Manage the review queue for compiled pages.

    Actions: list, approve, reject, lock, unlock
    """
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    queue = ReviewQueue(review_dir=vp.knox_dir / REVIEW_DIR)

    if action == "list":
        entries = queue.list_entries()
        if json_output:
            return {"status": "ok", "entries": [e.to_dict() for e in entries]}
        if not entries:
            return {"status": "ok", "message": "No pages in the review queue."}
        return {"status": "ok", "entries_text": "\n".join(f"  [{e.status:9s}] {e.slug}  ({e.title})" for e in entries)}

    if action == "lock":
        if not slug:
            return {"status": "error", "message": "Lock requires a slug."}
        if queue.lock(slug):
            return {"status": "ok", "message": f"Locked: {slug}"}
        return {"status": "error", "message": f"Cannot lock {slug!r}: no approved page found."}

    if action == "unlock":
        if not slug:
            return {"status": "error", "message": "Unlock requires a slug."}
        lock_file = queue.review_dir / f"{slug}.locked"
        if lock_file and lock_file.exists():
            lock_file.unlink()
            return {"status": "ok", "message": f"Unlocked: {slug}"}
        return {"status": "error", "message": f"No lock for {slug!r}."}

    if action == "approve":
        approved = queue.approve(slug or "", approve_all=approve_all)
        if not approved:
            return {"status": "error", "message": "No matching pages to approve."}
        return {"status": "ok", "approved": [e.slug for e in approved]}

    if action == "reject":
        if not slug:
            return {"status": "error", "message": "Reject requires a slug."}
        rejected = queue.reject(slug, reason=reason or "")
        if not rejected:
            return {"status": "error", "message": f"No pending page with slug {slug!r}."}
        return {"status": "ok", "rejected": [e.slug for e in rejected]}

    return {"status": "error", "message": f"Unknown review action: {action!r}. Use list|approve|reject|lock|unlock."}


# ── provenance ────────────────────────────────────────────────────────────


def provenance(
    wiki_dir: str | None = None,
    page_slugs: list[str] | None = None,
    with_content: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    """Trace compiled wiki page(s) back to their source chunks (provenance).

    Named ``provenance`` to avoid confusion with the OpenTelemetry tracing
    feature (see ``docs/guides/observability.md``).
    """
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    sources_dir = vp.sources_dir
    base_dir = vp.base_dir

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        return {"status": "error", "message": "No compiled wiki found. Run 'wiki_compile' first."}

    slugs = page_slugs or []

    if not slugs:
        pages = sorted(
            ((n.id, n.title, len(_node_sources(n, sources_dir))) for n in wiki.nodes),
            key=lambda p: p[0],
        )
        if json_output:
            return {"status": "ok", "pages": [{"slug": s, "title": t, "sources": c} for s, t, c in pages]}
        lines = [f"{len(pages)} page(s) in compiled wiki:\n"]
        for slug, title, count in pages:
            lines.append(f"  {slug:<40} ({count} sources)  {title}")
        return {"status": "ok", "pages_text": "\n".join(lines)}

    seen_slugs: set[str] = set()
    matched: list = []
    missing: list[str] = []
    for slug in slugs:
        key = slug.lower()
        if key in seen_slugs:
            continue
        seen_slugs.add(key)
        nodes = [n for n in wiki.nodes if n.id == slug or n.title.lower() == key]
        if nodes:
            matched.append(nodes[0])
        else:
            missing.append(slug)

    if json_output:
        entries = [{"page": n.title, "slug": n.id, "sources": _node_sources(n, sources_dir)} for n in matched]
        return {"status": "ok" if not missing else "error", "pages": entries[0] if len(entries) == 1 else entries, "missing": missing}

    lines = []
    for node in matched:
        node_sources = _node_sources(node, sources_dir)
        lines.append(f"# {node.title}\n")
        if node_sources:
            for src in node_sources:
                lines.append(f"  Source: {src}")
        else:
            lines.append("  (no sources tracked)")
        lines.append("")

    if missing:
        lines.append("Missing: " + ", ".join(missing))
        return {"status": "error", "pages_text": "\n".join(lines), "missing": missing}

    return {"status": "ok", "pages_text": "\n".join(lines)}


def _node_sources(node: Any, sources_dir: Path) -> list[str]:
    chunks: list[str] = list(getattr(node, "source_chunks", None) or [])
    if not chunks:
        metadata = getattr(node, "metadata", None) or {}
        prov = metadata.get(MetadataField.PROVENANCE) or []
        chunks = [str(s) for s in prov if s]
    return [_shorten(s, sources_dir) for s in chunks]


def _shorten(source: str, sources_dir: Path) -> str:
    source_path = Path(source)
    if not source_path.is_absolute():
        return source
    try:
        return source_path.relative_to(sources_dir).as_posix()
    except ValueError:
        return source_path.name


# ── log ───────────────────────────────────────────────────────────────────


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


# ── list_parsers ──────────────────────────────────────────────────────────


def list_parsers(
    json_output: bool = False,
) -> dict[str, Any]:
    """List all registered parsers (built-in + external + HTTP)."""
    build_default_router()
    parsers = get_registered_parsers()

    if json_output:
        return {"status": "ok", "parsers": parsers}

    if not parsers:
        return {"status": "ok", "message": "No parsers registered."}

    lines = [f"Registered parsers ({len(parsers)}):"]
    for p in parsers:
        exts = ", ".join(p.get("extensions", []) or [])
        ptype = p.get("type", "built-in")
        lines.append(f"  - {p['name']} ({ptype})  [{exts}]")
    return {"status": "ok", "parsers_text": "\n".join(lines)}


def _to_lint_report(report_dict: dict[str, Any]) -> LintReport:
    results = [
        LintReportResult(
            rule=r.get("rule", ""),
            outcome=r.get("outcome", ""),
            details=r.get("details", {}),
            message=r.get("message", ""),
            page_id=r.get("page_id", ""),
        )
        for r in report_dict.get("results", [])
    ]
    return LintReport(results=results)
