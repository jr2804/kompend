"""The ``knox lint`` command group.

Sub-commands
------------

* ``knox lint run``       — run lint rules and save to history.
* ``knox lint history``   — query past lint runs.
* ``knox lint gaps``      — analyze lint results for knowledge gaps.
* ``knox lint remediate`` — auto-fix, LLM-generate, or recommend fixes for lint findings.
* ``knox lint prune``     — remove old history entries.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import typer
import yaml

from knox.cli._pipeline_helpers import _write_wiki_cache
from knox.cli._shared import build_cli_overrides, load_hooks, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import (
    ConfigPath,
    JsonOutput,
    LintDryRun,
    LintFix,
    LintHistoryLast,
    LintHistoryTrend,
    LintLlm,
    LintPruneAgeDays,
    LintPruneKeep,
    LintRule,
    LlmContextBudget,
    LlmDefaultModel,
)
from knox.core.config import resolve_config
from knox.core.hooks import HookEvent, fire_hook
from knox.core.models import CompiledWiki, load_cached_wiki
from knox.core.paths import LINT_HISTORY_FILE, LOG_FILE
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
from knox.llm.client import LLMClient
from knox.logging import get_console, get_err_console
from knox.output.wiki_index import generate_stage_log_entry, prepend_log

lint_app = typer.Typer(
    name="lint",
    help="Lint a compiled wiki for quality issues and query lint history.",
    no_args_is_help=True,
)
app.add_typer(lint_app)


# ── Helpers ──────────────────────────────────────────────────────────────


def _uses_llm_rule(rule: list[str] | None) -> bool:
    """Return True if any requested rule name is LLM-powered."""
    if not rule:
        return False
    return any(r in LLM_RULE_BY_NAME for r in rule)


def _persist_lint_fixes(
    wiki: CompiledWiki,  # ty: ignore
    output_dir: Path,
    base_dir: Path,
) -> int:
    """Write lint-fixed frontmatter to markdown files and update the wiki cache.

    Returns the number of files updated.
    """
    updated = 0
    for node in wiki.nodes:  # ty: ignore
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

    _write_wiki_cache(wiki, base_dir)  # ty: ignore
    return updated


# ── Sub-command: run ─────────────────────────────────────────────────────


@lint_app.command("run", help="Run lint rules against the compiled wiki.")
def lint_run(
    ctx: typer.Context,
    fix: LintFix = False,
    llm: LintLlm = False,
    rule: LintRule = None,
    json_output: JsonOutput = False,
    llm_default_model: LlmDefaultModel = None,
    llm_context_budget: LlmContextBudget = None,
    config: ConfigPath = None,
) -> None:
    """Run structural (and optionally LLM-powered) lint rules.

    Results are printed and also appended to
    ``.knox/lint_history.jsonl`` for later querying via
    ``knox lint history``.

    Examples:
        knox lint run                       # structural rules only
        knox lint run --llm                 # structural + LLM rules
        knox lint run --fix                 # auto-fix what's possible
        knox lint run --rule broken-wikilinks --rule orphan-page
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(
        cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override),
        config_path=config,
    )
    sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)
    sources_dir_str = str(sources_dir) if sources_dir.exists() else None

    wiki = load_cached_wiki(state_dir.parent)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'knox compile' first.")
        raise typer.Exit(1)

    llm_client = None
    if llm or _uses_llm_rule(rule):
        if not llm_default_model:
            get_err_console().print(
                "LLM-powered lint rules require --llm-default-model (e.g. --llm-default-model openai/gpt-4o).",
            )
            raise typer.Exit(1)

        llm_client = LLMClient(default_model=llm_default_model)

    runner = LintRunner(
        rules=list(rule) if rule else None,
        enable_llm=llm,
        llm_client=llm_client,
        lint_page_budget=llm_context_budget,
        fix=fix,
        sources_dir=sources_dir_str,
    )
    lint_start = time.monotonic()
    report = runner.run(wiki)
    lint_elapsed = time.monotonic() - lint_start
    load_hooks()
    fire_hook(HookEvent.ON_LINT, report=report, wiki=wiki)

    fixes_applied = 0
    if fix and any(r.fix_applied for r in report.results):
        fixes_applied = _persist_lint_fixes(wiki, output_dir, state_dir.parent)
        get_console().print(f"Persisted fixes to {fixes_applied} file(s).")

    # Rule summary: "<n> fail / <n> pass / <n> skip" aggregated by outcome
    outcome_counts = Counter(r.outcome.value if hasattr(r.outcome, "value") else str(r.outcome) for r in report.results)
    rule_summary = " / ".join(f"{count} {label.lower()}" for label, count in sorted(outcome_counts.items()))

    log_file = state_dir / LOG_FILE
    prepend_log(
        log_file,
        generate_stage_log_entry(
            stage="lint",
            duration=lint_elapsed,
            source_count=len(wiki.nodes),
            health_score=report.health_score,
            rule_summary=rule_summary,
            fixes_applied=fixes_applied or None,
            llm_model=llm_default_model,
            llm_usage=llm_client.usage.to_dict() if llm_client else None,
        ),
    )

    # ── Persist to lint history ──────────────────────────────────────────
    llm_meta: dict | None = None
    if llm_client:
        llm_meta = {
            "model": llm_default_model,
            "usage": llm_client.usage.to_dict(),
        }

    options = {
        "rules": list(rule) if rule else None,
        "enable_llm": llm,
        "fix": fix,
        "llm_context_budget": llm_context_budget,
    }
    entry = entry_from_report(
        report,
        duration_seconds=lint_elapsed,
        options=options,
        config=cfg,
        node_count=len(wiki.nodes),
        llm=llm_meta,
    )
    save_lint_entry(entry, state_dir / LINT_HISTORY_FILE)

    if json_output:
        get_console().print(report_to_json(report))
    else:
        get_console().print(report_to_text(report))


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
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(
        cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override),
        config_path=config,
    )
    _sources_dir, _output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)

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

    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(
        cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override),
        config_path=config,
    )
    _sources_dir, _output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)

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
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(
        cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override),
        config_path=config,
    )
    _sources_dir, _output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        get_err_console().print("No lint history found. Run `knox lint run` or compile with --lint first.")
        raise typer.Exit(1)

    report_dict = entries[-1].get("report", {})
    recs = analyze_gaps(report_dict)

    if json_output:
        get_console().print(
            json.dumps(
                [{"priority": r.priority, "category": r.category, "finding": r.finding, "action": r.action} for r in recs],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        ts = entries[-1].get("timestamp", "?")
        score = report_dict.get("health_score", 0.0)
        get_console().print(f"\nBased on lint run from {ts} (score: {score:.0f}%)\n")
        get_console().print(gaps_to_text(recs))


# ── Sub-command: remediate ───────────────────────────────────────────────


@lint_app.command("remediate", help="Auto-fix, LLM-generate, or recommend fixes for lint findings.")
def lint_remediate(
    ctx: typer.Context,
    dry_run: LintDryRun = False,
    llm_default_model: LlmDefaultModel = None,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Remediate lint findings from the latest lint run.

    Three tiers of remediation:

    * **Tier 1** (auto-fix, no LLM): image embed syntax errors.
    * **Tier 2** (LLM-generated): coverage gaps filled from cached extractions.
    * **Tier 3** (recommendations): issues needing new sources or manual review.

    Tier 2 requires ``--llm-default-model``. Without it, only tier 1 runs
    and tier 2/3 are reported as recommendations.

    Examples:
        knox lint remediate                          # tier 1 only
        knox lint remediate --dry-run                # preview all tiers
        knox lint remediate --llm-default-model ollama/llama3.2
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(
        cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override),
        config_path=config,
    )
    sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)
    base_dir = state_dir.parent

    # Load latest lint history entry
    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        get_err_console().print("No lint history found. Run `knox lint run` or compile with --lint first.")
        raise typer.Exit(1)

    report_dict = entries[-1].get("report", {})

    # Load current wiki
    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'knox compile' first.")
        raise typer.Exit(1)

    # Set up LLM client for tier 2
    llm_client = None
    if llm_default_model:
        llm_client = LLMClient(default_model=llm_default_model)

    sources_dir_path = sources_dir if sources_dir.exists() else None

    actions = remediate(
        wiki,
        report_dict,
        base_dir=base_dir,
        sources_dir=sources_dir_path,
        llm_client=llm_client,
        dry_run=dry_run,
    )

    # Persist changes unless dry-run
    applied_count = sum(1 for a in actions if a.status == "applied")
    if applied_count > 0 and not dry_run:
        _write_wiki_cache(wiki, base_dir)
        get_console().print(f"Persisted {applied_count} remediation(s) to wiki cache.\n")

    if json_output:
        get_console().print(
            json.dumps(
                [
                    {
                        "tier": a.tier,
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
        ts = entries[-1].get("timestamp", "?")
        score = report_dict.get("health_score", 0.0)
        get_console().print(f"\nBased on lint run from {ts} (score: {score:.0f}%)\n")
        get_console().print(remediation_to_text(actions))
