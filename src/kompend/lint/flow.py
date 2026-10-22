"""Full lint pipeline shared by `kompend lint all` and `kompend compile --lint-all`.

One weighted progress bar (deterministic steps weight 1/item, LLM steps
weight 100/item), one merged history entry. LLM steps are skipped
gracefully with a note when no model is configured (compile.lint.model
override, then compile.llm.default_model).
"""

from __future__ import annotations

import time
from typing import Any

import yaml
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from kompend.agents.lint import LintRuleAgent
from kompend.cli._progress import EtaColumn, PercentageColumn, _should_show_progress
from kompend.core.cache.wiki import write_wiki_cache
from kompend.core.config._configs import LintConfig
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.models import CompiledWiki, MetadataField, load_cached_wiki
from kompend.core.paths import INDEX_FILE, LINT_HISTORY_FILE, VaultPaths
from kompend.lint.base import LintReport, LintResult
from kompend.lint.gap_analysis import analyze_gaps, load_degree_map
from kompend.lint.history import entry_from_report, save_lint_entry
from kompend.lint.remediation import RemediationStatus, remediate
from kompend.lint.runner import LintRunner, report_to_text
from kompend.logging import get_console, get_err_console, get_logger
from kompend.output.renderer import format_output

LLM_STEP_WEIGHT = 100
DETERMINISTIC_STEP_WEIGHT = 1

logger = get_logger(__name__)


def run_full_lint_flow(  # noqa: PLR0917, PLR0915 — pipeline orchestration takes many params
    # noqa: PLR0915 — pipeline orchestration; the length is intrinsic
    vp: VaultPaths,
    cfg: Any,
    *,
    fix: bool = True,
    fix_destructive: bool = False,
    json_output: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run every lint phase; return the merged report dict.

    Called by `kompend lint all` and `kompend compile --lint-all`. One merged
    history entry per call.
    """
    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        raise FileNotFoundError(f"No compiled wiki found in {vp.output_dir}. Run 'kompend compile' first.")

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
                check_report, _ = _run_pass(wiki, vp, cfg, rules=None, enable_llm=False, lint_agent=None, fix=fix, fix_destructive=fix_destructive)
            elif name == "format":
                _format_output_phase(vp, cfg)
            elif name == "semantic":
                semantic_report, _ = _run_pass(wiki, vp, cfg, rules=None, enable_llm=True, lint_agent=lint_agent, fix=False)
            elif name == "gaps":
                merged_results = _merge_results(check_report, semantic_report)
                analyze_gaps(LintReport(results=merged_results), load_degree_map(vp.base_dir), cfg.kompend.display)
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
        persist_lint_fixes(wiki, vp)

    merged_results = _merge_results(check_report, semantic_report)
    merged_report = _rich_report(merged_results)

    llm_meta: dict[str, Any] | None = None
    if lint_agent is not None:
        usage = getattr(lint_agent, "usage", None)
        llm_meta = {MetadataField.MODEL: lint_model}
        if usage is not None and hasattr(usage, "to_dict"):
            llm_meta["usage"] = usage.to_dict()

    entry = entry_from_report(
        merged_report,
        duration_seconds=time.monotonic() - flow_start,
        options={"context": "all", "fix": fix},
        config=cfg.raw,
        node_count=node_count,
        llm=llm_meta,
    )
    save_lint_entry(entry, vp.kompend_dir / LINT_HISTORY_FILE)

    merged_dict = entry["report"]
    if json_output:
        get_console().print_json(data=merged_dict)
    else:
        get_console().print(report_to_text(merged_report))
    return merged_dict


def _make_lint_agent(cfg: Any) -> tuple[LintRuleAgent | None, str | None]:
    """Lint agent from the lint model override (None when no model configured)."""
    model = _lint_model(cfg)
    if not model:
        return None, None
    return (
        LintRuleAgent(
            model,
            timeout=float(cfg.kompend.compile.llm.timeout),
            retries=cfg.kompend.compile.llm.output_retries if cfg.kompend.compile.llm.output_retries_enabled else 0,
            llm=cfg.kompend.compile.llm,
            budget=cfg.kompend.compile.budget,
        ),
        model,
    )


def _lint_model(cfg: Any) -> str | None:
    """LLM model for lint steps: compile.lint.model override, else default_model."""
    return cfg.kompend.lint.model or cfg.kompend.compile.llm.default_model


def _run_pass(
    wiki: CompiledWiki,
    vp: VaultPaths,
    cfg: Any,
    *,
    rules: list[str] | None,
    enable_llm: bool,
    lint_agent: LintRuleAgent | None,
    fix: bool,
    fix_destructive: bool = False,
) -> tuple[LintReport, float]:
    """Run one LintRunner pass and return (report, elapsed)."""
    runner = LintRunner(
        rules=rules,
        enable_llm=enable_llm,
        lint_agent=lint_agent,
        fix=fix,
        fix_destructive=fix_destructive,
        sources_dir=str(vp.sources_dir) if vp.sources_dir.exists() else None,
        lint_config=LintConfig(fuzzy_threshold=float(cfg.kompend.lint.fuzzy_threshold)),
        rumdl_config=resolve_rumdl_config(vp.base_dir, cfg.kompend.compile.output.rumdl_config),
    )
    start = time.monotonic()
    report = runner.run(wiki)
    return report, time.monotonic() - start


def run_runner_with_progress(
    runner: LintRunner,
    wiki: CompiledWiki,
    *,
    show_progress: bool | None = None,
) -> LintReport:
    """Run a :class:`LintRunner` with a per-rule progress bar (ETA included).

    Shared by ``kompend lint check`` and the post-compile ``--lint`` pass so
    both surfaces get the same progress experience. The bar is skipped when
    *show_progress* is False or stdout is not a TTY (auto mode).
    """
    if not _should_show_progress(show_progress):
        return runner.run(wiki)
    progress = _build_progress()
    task_id = progress.add_task("lint", total=None)  # total set on first on_rule
    with progress:

        def _on_rule(name: str, current: int, total: int) -> None:
            progress.update(task_id, total=total, completed=current, description=f"lint [{name}]")

        return runner.run(wiki, on_rule=_on_rule)


def _merge_results(
    check_report: LintReport | None,
    semantic_report: LintReport | None,
) -> list[LintResult]:
    return (check_report.results if check_report else []) + (semantic_report.results if semantic_report else [])


def _rich_report(results: list[LintResult]) -> LintReport:
    """Populate summary fields from a flat result list (single source of
    truth for the merge; mirrors ``runner._build_report`` aggregation).
    """
    passes = sum(1 for r in results if r.outcome == "pass")
    failures = sum(1 for r in results if r.outcome == "fail")
    skips = sum(1 for r in results if r.outcome == "skip")
    total_checks = len(results)
    health_score = round(100.0 * passes / total_checks, 1) if total_checks else 0.0
    return LintReport(
        results=results,
        total_checks=total_checks,
        failures=failures,
        passes=passes,
        skips=skips,
        health_score=health_score,
    )


def _format_output_phase(vp: VaultPaths, cfg: Any) -> None:
    """Step: run rumdl format pass on the output dir."""
    rumdl_cfg = resolve_rumdl_config(vp.base_dir, cfg.kompend.compile.output.rumdl_config)
    extra_files = [p for fname in (INDEX_FILE, "overview.md") if (p := vp.base_dir / fname).exists()]
    format_output(
        vp.output_dir,
        flavor=cfg.kompend.compile.output.flavor,
        rumdl_config=rumdl_cfg,
        extra_files=extra_files or None,
        check_only=False,
    )


def _remediate_phase(  # noqa: PLR0917 — orchestrates one step with 6 inputs
    wiki: CompiledWiki,
    vp: VaultPaths,
    cfg: Any,
    merged_results: list[LintResult],
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


def persist_lint_fixes(wiki: CompiledWiki, vp: VaultPaths) -> int:
    """Write lint-fixed frontmatter to the written pages and refresh the wiki cache.

    Public because the MCP lint tool needs it too; it used to exist in two
    copies (here and in the CLI), and the CLI's — three positional args — was
    the one its callers kept using, so the MCP path called this one with too
    many arguments and raised TypeError.

    Returns the number of files updated.
    """
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
        PercentageColumn(),
        EtaColumn(),
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
            "Note: no LLM configured (compile.lint.model / compile.llm.default_model) - skipping semantic and remediate steps.\n",
        )
    return steps
