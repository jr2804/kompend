"""Quality & diagnostics MCP tools (lint, doctor, review, provenance, log, list-parsers)."""

from pathlib import Path
from typing import Any

from kompend.agents.remediate import make_agentic_remediation_agent
from kompend.cli._post_compile import write_wiki_cache
from kompend.cli._shared import build_cli_overrides
from kompend.cli.cmds.quality.doctor import run_doctor
from kompend.cli.cmds.quality.lint import CHECK_STEP_RULES, _make_lint_agent, _persist_history
from kompend.core.config import resolve_config
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_VLM_MODEL,
)
from kompend.core.models import MetadataField, load_wiki_from_files
from kompend.core.paths import INDEX_FILE, LINT_HISTORY_FILE, OVERVIEW_MD, REVIEW_DIR, VaultPaths, vault_paths
from kompend.core.reporting.log import build_log_report
from kompend.core.utils.dicts import as_str
from kompend.lint.base import RemediationStatus
from kompend.lint.flow import _run_pass as _run_rules_pass
from kompend.lint.flow import persist_lint_fixes, run_full_lint_flow
from kompend.lint.gap_analysis import analyze_gaps, gaps_to_text
from kompend.lint.history import (
    history_to_json,
    history_to_text,
    load_lint_history,
    prune_lint_history,
    report_from_dict,
)
from kompend.lint.llm_rules import LLM_RULE_BY_NAME
from kompend.lint.log import latest_lint_log, lint_log_dir, read_lint_log
from kompend.lint.remediation import remediate, remediation_to_text
from kompend.lint.runner import report_to_text
from kompend.logging import get_logger
from kompend.output.renderer import format_output
from kompend.preprocessing.parsers._registry import get_registered_parsers
from kompend.review import ReviewQueue

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
        "health_score": merged.get("health_score", 0.0),
        "passes": merged.get("passes", 0),
        "failures": merged.get("failures", 0),
        "skips": merged.get("skips", 0),
        "report": merged,
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
    _persist_history(report, vp=vp, cfg=cfg, context="check", elapsed=elapsed, fix=fix, node_count=len(wiki.nodes), llm_meta=None)
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
    llm_meta = {MetadataField.MODEL: model, "usage": lint_agent.usage.to_dict()} if lint_agent.usage is not None else None
    _persist_history(report, vp=vp, cfg=cfg, context="semantic", elapsed=elapsed, fix=False, node_count=len(wiki.nodes), llm_meta=llm_meta)
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

    report_dict = entries[-1].get("report", {})
    report = report_from_dict(report_dict)
    recs = analyze_gaps(report, display=cfg.kompend.display)

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
    state_dir = vp.kompend_dir
    base_dir = vp.base_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        return {"status": "error", "message": "No lint history found. Run 'lint_all' or compile with --lint first."}

    report_dict = entries[-1].get("report", {})
    report = report_from_dict(report_dict)

    wiki = load_wiki_from_files(VaultPaths(base_dir).output_dir)
    if wiki is None:
        return {"status": "error", "message": f"No wiki pages found in {output_dir}. Run 'wiki_compile' first."}

    default_model = cfg.kompend.compile.llm.default_model
    agent = (
        make_agentic_remediation_agent(
            model_string=default_model,
            timeout=float(cfg.kompend.compile.llm.timeout),
            llm=cfg.kompend.compile.llm,
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
        default_model=as_str(cfg.get(COMPILE_LLM_DEFAULT_MODEL)),
        vlm_model=as_str(cfg.get(COMPILE_LLM_VLM_MODEL)),
        embed_model=as_str(cfg.get(COMPILE_EMBEDDING_MODEL)),
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


def review(  # noqa: PLR0917, PLR0911 — MCP tool with many kwargs
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
    queue = ReviewQueue(review_dir=vp.kompend_dir / REVIEW_DIR)

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

    wiki = load_wiki_from_files(VaultPaths(base_dir).output_dir)
    if wiki is None:
        return {"status": "error", "message": "No wiki pages found. Run 'wiki_compile' first."}

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
