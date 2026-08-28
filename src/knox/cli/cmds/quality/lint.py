"""The ``knox lint`` command group.

Sub-commands
------------

* ``knox lint run``       — run lint rules and save to history.
* ``knox lint history``   — query past lint runs.
* ``knox lint gaps``      — analyze lint results for knowledge gaps.
* ``knox lint remediate`` — LLM-generate missing pages for lint findings.
* ``knox lint prune``     — remove old history entries.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from knox.agents.lint import LintRuleAgent
from knox.agents.remediate import make_agentic_remediation_agent
from knox.cli._pipeline_helpers import write_wiki_cache
from knox.cli._shared import build_cli_overrides, vault_paths_from_ctx
from knox.cli.app import app
from knox.cli.args import (
    ConfigPath,
    JsonOutput,
    LintDryRun,
    LintFix,
    LintFormatCheck,
    LintHistoryLast,
    LintHistoryTrend,
    LintLlm,
    LintPruneAgeDays,
    LintPruneKeep,
    LintRule,
)
from knox.core.config import resolve_config
from knox.core.config._configs import LintConfig
from knox.core.config.rumdl import resolve_rumdl_config
from knox.core.models import CompiledWiki, load_cached_wiki
from knox.core.paths import INDEX_FILE, LINT_HISTORY_FILE
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
from knox.logging import get_console, get_err_console
from knox.output.pages.log import log_stage
from knox.output.renderer import format_output

lint_app = typer.Typer(
    name="lint",
    help="Lint a compiled wiki for quality issues and query lint history.",
    no_args_is_help=True,
)
app.add_typer(lint_app)


# ── Sub-command: run ─────────────────────────────────────────────────────


@lint_app.command("run", help="Run lint rules against the compiled wiki.")
def lint_run(
    ctx: typer.Context,
    fix: LintFix = False,
    llm: LintLlm = True,
    rule: LintRule = None,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Run structural (and optionally LLM-powered) lint rules.

    Results are printed and also appended to
    ``.knox/lint_history.jsonl`` for later querying via
    ``knox lint history``.

    LLM-powered rules are on by default; the model comes from
    ``compile.llm.default_model`` in config. Use ``--no-llm`` for
    structural rules only. When LLM is on but no model is configured,
    LLM rules are skipped with a note (structural rules still run).

    Examples:
        knox lint run                       # structural + LLM rules (default)
        knox lint run --no-llm              # structural rules only
        knox lint run --fix                 # auto-fix what's possible
        knox lint run --rule broken-wikilinks --rule orphan-page
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir
    state_dir = vp.knox_dir
    sources_dir_str = str(sources_dir) if sources_dir.exists() else None

    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'knox compile' first.")
        raise typer.Exit(1)

    default_model = cfg.knox.compile.llm.default_model
    lint_agent = (
        LintRuleAgent(default_model, timeout=float(cfg.knox.compile.llm.timeout), retries=cfg.knox.compile.llm.output_retries) if default_model else None
    )
    if _uses_llm_rule(rule) and lint_agent is None:
        get_err_console().print(
            "LLM-powered lint rules require a model. Set compile.llm.default_model in config (e.g. openai/gpt-4o).",
        )
        raise typer.Exit(1)
    if llm and lint_agent is None:
        get_console().print(
            "No LLM model configured (compile.llm.default_model); running structural rules only. Use --no-llm to silence this note.",
        )
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
    progress = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=get_err_console(),
            transient=False,
            refresh_per_second=2,
        )
        if not json_output
        else None
    )
    if progress is not None:
        progress.start()
    lint_task = progress.add_task("linting", total=0) if progress is not None else None

    def _on_rule(rule_name: str, current: int, total: int) -> None:
        if progress is not None and lint_task is not None:
            progress.update(
                lint_task,
                total=total,
                completed=current - 1,
                description=f"linting [{current}/{total}] {rule_name}",
            )

    report = runner.run(wiki, on_rule=_on_rule)
    if progress is not None and lint_task is not None:
        progress.update(lint_task, completed=progress.tasks[lint_task].total or 0, description="linting complete")
        progress.stop()
    lint_elapsed = time.monotonic() - lint_start

    fixes_applied = 0
    if fix and any(r.fix_applied for r in report.results):
        fixes_applied = _persist_lint_fixes(wiki, output_dir, vp.base_dir)
        get_console().print(f"Persisted fixes to {fixes_applied} file(s).")

    # Rule summary: "<n> fail / <n> pass / <n> skip" aggregated by outcome
    outcome_counts = Counter(r.outcome.value if hasattr(r.outcome, "value") else str(r.outcome) for r in report.results)
    " / ".join(f"{count} {label.lower()}" for label, count in sorted(outcome_counts.items()))

    # ── Persist to lint history ──────────────────────────────────────────
    llm_meta: dict | None = None
    if lint_agent is not None and lint_agent.usage is not None:
        llm_meta = {
            "model": default_model,  # nocheck: lint metadata, not wiki metadata
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

    if json_output:
        get_console().print(report_to_json(report))
    else:
        get_console().print(report_to_text(report))
    log_stage(
        vp,
        action="Lint",
        detail=f"Health score {report.health_score:.1f}% ({report.passes}/{report.total_checks} checks passed)",
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _uses_llm_rule(rule: list[str] | None) -> bool:
    """Return True if any requested rule name is LLM-powered."""
    if not rule:
        return False
    return any(r in LLM_RULE_BY_NAME for r in rule)


def _persist_lint_fixes(
    wiki: CompiledWiki,
    output_dir: Path,
    base_dir: Path,
) -> int:
    """Write lint-fixed frontmatter to markdown files and update the wiki cache.

    Returns the number of files updated.
    """
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


# ── Sub-command: history ─────────────────────────────────────────────────


@lint_app.command("history", help="Show past lint runs and score trends.")
def lint_history(
    ctx: typer.Context,
    last: LintHistoryLast = 10,
    trend: LintHistoryTrend = False,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Query the lint history log.

    Examples:
        knox lint history              # last 10 runs
        knox lint history -n 5         # last 5 runs
        knox lint history --trend      # with score sparkline
        knox lint history --json       # machine-readable
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    state_dir = vp.knox_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=last)

    if json_output:
        get_console().print(history_to_json(entries))
    else:
        get_console().print(history_to_text(entries, show_trend=trend))


# ── Sub-command: prune ───────────────────────────────────────────────────


@lint_app.command("prune", help="Remove old entries from the lint history.")
def lint_prune(
    ctx: typer.Context,
    keep_last: LintPruneKeep = None,
    max_age_days: LintPruneAgeDays = None,
    config: ConfigPath = None,
) -> None:
    """Remove old entries from ``.knox/lint_history.jsonl``.

    At least one of --keep-last or --max-age-days must be provided.

    Examples:
        knox lint prune --keep-last 50        # keep only 50 newest
        knox lint prune --max-age-days 30     # remove entries older than 30 days
    """
    if keep_last is None and max_age_days is None:
        get_err_console().print("At least one of --keep-last or --max-age-days must be provided.")
        raise typer.Exit(1)

    vp = vault_paths_from_ctx(ctx)
    resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    state_dir = vp.knox_dir

    removed = prune_lint_history(
        state_dir / LINT_HISTORY_FILE,
        keep_last=keep_last,
        max_age_days=max_age_days,
    )
    get_console().print(f"Removed {removed} entr{'y' if removed == 1 else 'ies'}.")


# ── Sub-command: gaps ────────────────────────────────────────────────────


@lint_app.command("gaps", help="Analyze lint results for knowledge gaps and source recommendations.")
def lint_gaps(
    ctx: typer.Context,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Analyze the latest lint run to identify missing knowledge and recommend sources.

    Reads the most recent entry from ``.knox/lint_history.jsonl`` and extracts
    actionable gap signals: missing topics (broken wikilinks), coverage gaps,
    poor-quality pages, empty pages, orphan topics, and contradictions.

    Examples:
        knox lint gaps              # analyze latest lint run
        knox lint gaps --json       # machine-readable output
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    state_dir = vp.knox_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        get_err_console().print("No lint history found. Run `knox lint run` or compile with --lint first.")
        raise typer.Exit(1)

    report_dict = entries[-1].get("report", {})
    report = _to_lint_report(report_dict)
    recs = analyze_gaps(report)

    if json_output:
        get_console().print(
            json.dumps(
                [{"priority": r.priority, "category": r.category, "finding": r.finding, "action": r.action} for r in recs],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        ts = entries[-1].get("timestamp", "unknown")
        score = report_dict.get("health_score", 0.0)
        get_console().print(f"\nBased on lint run from {ts} (score: {score:.0f}%)\n")
        get_console().print(gaps_to_text(recs))


# ── Sub-command: remediate ───────────────────────────────────────────────


@lint_app.command("remediate", help="LLM-generate missing pages for lint findings.")
def lint_remediate(
    ctx: typer.Context,
    dry_run: LintDryRun = False,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Generate missing concept pages for coverage-gap findings.

    Uses ``compile.llm.default_model`` from config. Coverage-gap findings
    are matched against cached phase-1 extractions; matched concepts get
    concept pages generated from that source material. Unmatched concepts
    are skipped — ``knox lint gaps`` shows recommendations for those.

    Examples:
        knox lint remediate                          # generate pages
        knox lint remediate --dry-run                # preview without writing
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    output_dir = vp.output_dir
    state_dir = vp.knox_dir
    base_dir = vp.base_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        get_err_console().print("No lint history found. Run `knox lint run` or compile with --lint first.")
        raise typer.Exit(1)

    report_dict = entries[-1].get("report", {})
    report = _to_lint_report(report_dict)

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'knox compile' first.")
        raise typer.Exit(1)

    default_model = cfg.knox.compile.llm.default_model
    agent = make_agentic_remediation_agent(model_string=default_model, timeout=float(cfg.knox.compile.llm.timeout)) if default_model else None

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
                        "description": a.description,
                        "status": a.status,
                        "details": a.details,
                    }
                    for a in actions
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        ts = entries[-1].get("timestamp", "unknown")
        score = report_dict.get("health_score", 0.0)
        get_console().print(f"\nBased on lint run from {ts} (score: {score:.0f}%)\n")
        get_console().print(remediation_to_text(actions))


# ── Sub-command: format ─────────────────────────────────────────────────


@lint_app.command("format", help="Format generated wiki markdown with rumdl fmt.")
def lint_format(
    ctx: typer.Context,
    check: LintFormatCheck = False,
    config: ConfigPath = None,
) -> None:
    """Run ``rumdl fmt`` over the compiled output (bundle + root index).

    Formats every generated ``.md`` file with the vault's rumdl config
    (``compile.output.rumdl_config``, default ``.knox/rumdl.toml``). Works
    without a compiled-wiki cache — the files on disk are the source of
    truth. ``--check`` exits 1 if any formatting changes are needed (CI).

    Examples:
        knox lint format                 # format generated markdown
        knox lint format --check         # CI: exit 1 if changes needed
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
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
            get_err_console().print(
                f"{len(changed)} file(s) need formatting. Run 'knox lint format' to fix.",
            )
            raise typer.Exit(1)
        get_console().print("All generated markdown is formatted.")
        return
    if changed:
        get_console().print(f"Formatted {len(changed)} file(s) with rumdl.")
    else:
        get_console().print("No formatting changes needed.")


def _to_lint_report(report_dict: dict[str, Any]) -> LintReport:
    """Convert a raw ``report`` dict from ``lint_history.jsonl`` to :class:`LintReport`."""
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
