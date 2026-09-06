"""Full lint pipeline shared by `knox lint all` and `knox compile --lint-all`.

One weighted progress bar (deterministic steps weight 1/item, LLM steps
weight 100/item), one merged history entry. LLM steps are skipped
gracefully with a note when no model is configured (compile.lint.model
override, then compile.llm.default_model).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from knox.agents.lint import LintRuleAgent
from knox.core.cache.wiki import write_wiki_cache
from knox.core.config._configs import LintConfig
from knox.core.config.rumdl import resolve_rumdl_config
from knox.core.models import CompiledWiki, load_cached_wiki
from knox.core.paths import INDEX_FILE, LINT_HISTORY_FILE, VaultPaths
from knox.lint.base import LintReport, LintReportResult
from knox.lint.gap_analysis import analyze_gaps
from knox.lint.history import entry_from_report, save_lint_entry
from knox.lint.remediation import RemediationStatus, remediate
from knox.lint.runner import LintRunner, report_to_text
from knox.logging import get_console, get_err_console, get_logger
from knox.output.renderer import format_output

LLM_STEP_WEIGHT = 100
DETERMINISTIC_STEP_WEIGHT = 1

logger = get_logger(__name__)


def run_full_lint_flow(  # noqa: PLR0917, PLR0915 — pipeline orchestration takes many params
  # noqa: PLR0915 — pipeline orchestration; the length is intrinsic
    vp: VaultPaths,
    cfg: Any,
    *,
    fix: bool = True,
    json_output: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run every lint phase; return the merged report dict.

    Called by `knox lint all` and `knox compile --lint-all`. One merged
    history entry per call.
    """
    wiki_cache = vp.knox_dir / "last_wiki.json"
    if not wiki_cache.exists():
        raise FileNotFoundError(f"No compiled wiki found in {vp.output_dir}. Run 'knox compile' first.")

    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        raise FileNotFoundError(f"No compiled wiki found in {vp.output_dir}. Run 'knox compile' first.")

    node_count = len(wiki.nodes)
    lint_agent, lint_model = _make_lint_agent(cfg)
    steps = _plan(vp, cfg, lint_agent)
    grand_total = sum(s["weight"] * s["items"] for s in steps)

    progress = _build_progress() if show_progress else None
    task_id = progress.add_task("lint", total=grand_total) if progress is not None else None
    completed = 0.0
    check_report: LintReport | None = None
    semantic_report: LintReport | None = None
    flow_start = time.monotonic()

    try:
        for step in steps:
            name = step["name"]
            weight = int(step["weight"])
            items = int(step["items"])
            if progress is not None and task_id is not None:
                progress.update(task_id, description=f"lint [{name}]")
            if name == "check":
                check_report, _ = _run_pass(
                    wiki, vp, cfg, rules=None, enable_llm=False, lint_agent=None, fix=fix
                )
            elif name == "format":
                _format_output_phase(vp, cfg)
            elif name == "semantic":
                semantic_report, _ = _run_pass(
                    wiki, vp, cfg, rules=None, enable_llm=True, lint_agent=lint_agent, fix=False
                )
            elif name == "gaps":
                merged_results = _merge_results(check_report, semantic_report)
                analyze_gaps(LintReport(results=merged_results), _load_degree_map(vp.base_dir))
            elif name == "remediate":
                merged_results = _merge_results(check_report, semantic_report)
                actions = _remediate_phase(wiki, vp, cfg, merged_results, lint_agent, fix)
                step["items"] = max(len(actions), 1)
                if progress is not None and task_id is not None:
                    progress.update(task_id, total=sum(s["weight"] * s["items"] for s in steps))
            completed += weight * items
            if progress is not None and task_id is not None:
                progress.update(task_id, completed=completed, description=f"lint [{name}] done")
    finally:
        if progress is not None and task_id is not None:
            progress.update(
                task_id,
                completed=progress.tasks[task_id].total or completed,
                description="lint complete",
            )
            progress.stop()

    if fix and check_report and any(r.fix_applied for r in check_report.results):
        _persist_lint_fixes(wiki, vp)

    merged_results = _merge_results(check_report, semantic_report)
    passes = sum(1 for r in merged_results if r.outcome == "pass")
    total_checks = len(merged_results)
    failures = sum(1 for r in merged_results if r.outcome == "fail")
    skips = sum(1 for r in merged_results if r.outcome == "skip")
    health_score = round(100.0 * passes / total_checks, 1) if total_checks else 0.0

    llm_meta: dict[str, Any] | None = None
    if lint_agent is not None:
        usage = getattr(lint_agent, "usage", None)
        llm_meta = {"model": lint_model}
        if usage is not None and hasattr(usage, "to_dict"):
            llm_meta["usage"] = usage.to_dict()

    entry = entry_from_report(
        LintReport(results=merged_results),
        duration_seconds=time.monotonic() - flow_start,
        options={"context": "all", "fix": fix},
        config=cfg.raw,
        node_count=node_count,
        llm=llm_meta,
    )
    entry["report"] = {
        "results": [
            {
                "rule": r.rule,
                "page_id": r.page_id,
                "outcome": r.outcome,
                "severity": r.severity.name if hasattr(r.severity, "name") else str(r.severity),
                "fix_applied": r.fix_applied,
                "details": r.details,
            }
            for r in merged_results
        ],
        "passes": passes,
        "failures": failures,
        "skips": skips,
        "total_checks": total_checks,
        "health_score": health_score,
    }
    save_lint_entry(entry, vp.knox_dir / LINT_HISTORY_FILE)

    merged_dict = entry["report"]
    if json_output:
        get_console().print_json(merged_dict)
    else:
        get_console().print(report_to_text(LintReport(results=merged_results)))
    return merged_dict


def _make_lint_agent(cfg: Any) -> tuple[LintRuleAgent | None, str | None]:
    """Lint agent from the lint model override (None when no model configured)."""
    model = _lint_model(cfg)
    if not model:
        return None, None
    return (
        LintRuleAgent(model, timeout=float(cfg.knox.compile.llm.timeout), retries=cfg.knox.compile.llm.output_retries),
        model,
    )


def _lint_model(cfg: Any) -> str | None:
    """LLM model for lint steps: compile.lint.model override, else default_model."""
    return cfg.knox.lint.model or cfg.knox.compile.llm.default_model


def _run_pass(
    wiki: CompiledWiki,
    vp: VaultPaths,
    cfg: Any,
    *,
    rules: list[str] | None,
    enable_llm: bool,
    lint_agent: LintRuleAgent | None,
    fix: bool,
) -> tuple[LintReport, float]:
    """Run one LintRunner pass and return (report, elapsed)."""
    runner = LintRunner(
        rules=rules,
        enable_llm=enable_llm,
        lint_agent=lint_agent,
        fix=fix,
        sources_dir=str(vp.sources_dir) if vp.sources_dir.exists() else None,
        lint_config=LintConfig(fuzzy_threshold=float(cfg.knox.lint.fuzzy_threshold)),
        rumdl_config=resolve_rumdl_config(vp.base_dir, cfg.knox.compile.output.rumdl_config),
    )
    start = time.monotonic()
    report = runner.run(wiki)
    return report, time.monotonic() - start


def _merge_results(
    check_report: LintReport | None,
    semantic_report: LintReport | None,
) -> list[LintReportResult]:
    return (check_report.results if check_report else []) + (semantic_report.results if semantic_report else [])


def _load_degree_map(base_dir: Path) -> dict[str, int] | None:
    cache_file = base_dir / ".knox" / "last_wiki.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    degree: dict[str, int] = {}
    for edge in data.get("edges", []):
        if isinstance(edge, dict):
            for endpoint in (edge.get("source"), edge.get("target")):
                if endpoint:
                    degree[str(endpoint)] = degree.get(str(endpoint), 0) + 1
    return degree or None


def _format_output_phase(vp: VaultPaths, cfg: Any) -> None:
    """Step: run rumdl format pass on the output dir."""
    rumdl_cfg = resolve_rumdl_config(vp.base_dir, cfg.knox.compile.output.rumdl_config)
    extra_files = [p for fname in (INDEX_FILE, "overview.md") if (p := vp.base_dir / fname).exists()]
    format_output(
        vp.output_dir,
        flavor=cfg.knox.compile.output.flavor,
        rumdl_config=rumdl_cfg,
        extra_files=extra_files or None,
        check_only=False,
    )


def _remediate_phase(  # noqa: PLR0917 — orchestrates one step with 6 inputs
    wiki: CompiledWiki,
    vp: VaultPaths,
    cfg: Any,
    merged_results: list[LintReportResult],
    lint_agent: Any,
    fix: bool,
) -> list[Any]:
    """Step: generate pages for source-covered missing topics; return actions."""
    actions = remediate(
        wiki,
        LintReport(results=merged_results),
        base_dir=vp.base_dir,
        output_dir=vp.output_dir,
        agent=lint_agent,
        dry_run=not fix,
    )
    if fix and any(a.status == RemediationStatus.APPLIED for a in actions):
        write_wiki_cache(wiki, vp.base_dir)
    return actions


def _persist_lint_fixes(wiki: CompiledWiki, vp: VaultPaths) -> int:
    """Write lint-fixed frontmatter to markdown files; returns updated count."""
    updated = 0
    for node in wiki.nodes:
        fm = node.metadata.get("frontmatter", {})
        if not isinstance(fm, dict) or "title" not in fm:
            continue
        md_file = vp.output_dir / f"node_{node.id}.md"
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
    write_wiki_cache(wiki, vp.base_dir)
    return updated


def _build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=get_err_console(),
        transient=False,
        refresh_per_second=2,
    )


def _plan(vp: VaultPaths, cfg: Any, lint_agent: Any) -> list[dict[str, Any]]:
    node_count = sum(1 for _ in vp.output_dir.rglob("node_*.md")) or 1
    fmt_files = sum(1 for _ in vp.output_dir.rglob("*.md"))
    steps: list[dict[str, Any]] = [
        {"name": "check", "weight": DETERMINISTIC_STEP_WEIGHT, "items": node_count},
        {"name": "format", "weight": DETERMINISTIC_STEP_WEIGHT, "items": max(fmt_files, 1)},
        {"name": "gaps", "weight": DETERMINISTIC_STEP_WEIGHT, "items": 1},
    ]
    if lint_agent is not None:
        steps.insert(2, {"name": "semantic", "weight": LLM_STEP_WEIGHT, "items": node_count})
        steps.append({"name": "remediate", "weight": LLM_STEP_WEIGHT, "items": 0})
    else:
        get_err_console().print(
            "Note: no LLM configured (compile.lint.model / compile.llm.default_model) "
            "- skipping semantic and remediate steps.\n",
        )
    return steps
