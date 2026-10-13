"""The ``kompend lint`` command group.

Sub-commands
------------

* ``kompend lint run``       — run lint rules and save to history.
* ``kompend lint history``   — query past lint runs.
* ``kompend lint gaps``      — analyze lint results for knowledge gaps.
* ``kompend lint remediate`` — LLM-generate missing pages for lint findings.
* ``kompend lint prune``     — remove old history entries.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import typer

from kompend.agents.remediate import make_agentic_remediation_agent
from kompend.cli._pipeline_helpers import write_wiki_cache
from kompend.cli._shared import build_cli_overrides, vault_paths_from_ctx
from kompend.cli.app import app
from kompend.cli.args import (
    ConfigPath,
    JsonOutput,
    LintDryRun,
    LintFix,
    LintFixDestructive,
    LintFormatCheck,
    LintHistoryLast,
    LintHistoryTrend,
    LintPruneAgeDays,
    LintPruneKeep,
    LintRule,
)
from kompend.core.config import resolve_config
from kompend.core.config._configs import LintConfig
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.models import MetadataField, load_cached_wiki
from kompend.core.paths import INDEX_FILE, LINT_HISTORY_FILE
from kompend.lint.base import LintReport, RemediationStatus
from kompend.lint.flow import (
    _make_lint_agent,
    persist_lint_fixes,
    run_full_lint_flow,  # noqa: F401 — re-export
    run_runner_with_progress,
)
from kompend.lint.flow import (
    _run_pass as _run_rules_pass,
)
from kompend.lint.gap_analysis import analyze_gaps, gaps_to_text, load_degree_map
from kompend.lint.history import (
    entry_from_report,
    history_to_json,
    history_to_text,
    load_lint_history,
    prune_lint_history,
    report_from_dict,
    save_lint_entry,
)
from kompend.lint.llm_rules import LLM_RULE_BY_NAME
from kompend.lint.remediation import remediate, remediation_to_text
from kompend.lint.runner import LintRunner, report_to_json, report_to_text
from kompend.lint.semantic_gaps import verify_topics
from kompend.lint.structural import STRUCTURAL_RULES
from kompend.logging import get_console, get_err_console
from kompend.output.pages.log import log_stage
from kompend.output.renderer import format_output

lint_app = typer.Typer(
    name="lint",
    help="Lint a compiled wiki for quality issues and query lint history.",
    no_args_is_help=True,
)
app.add_typer(lint_app)


# ── Sub-command: run ─────────────────────────────────────────────────────


LLM_STEP_WEIGHT = 100
"""Progress weight per item for LLM-backed lint steps (dominates wall-clock)."""

DETERMINISTIC_STEP_WEIGHT = 1
"""Progress weight per item for deterministic lint steps."""


# ── Helpers ──────────────────────────────────────────────────────────────


#: Deterministic rules for the `check` step — everything structural except
#: rumdl (which is the separate `format` step).
CHECK_STEP_RULES: list[str] = [r.name for r in STRUCTURAL_RULES] + ["duplicate-references"]


@lint_app.command("check", help="Run deterministic structural lint rules (no LLM).")
def lint_check(
    ctx: typer.Context,
    fix: LintFix = True,
    fix_destructive: LintFixDestructive = False,
    rule: LintRule = None,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Run the deterministic structural rules against the compiled wiki.

    Covers: broken-wikilinks, image-embed, kompend-reference, missing-frontmatter,
    orphan-page, duplicate-title, empty-page, heading-skip, stale-page,
    source-not-found, orphan-attachments, duplicate-references.
    Formatting (rumdl) is the separate `kompend lint format` step.

    With --fix (default), repairable findings are fixed in place: alias
    links are rewritten to resolved markdown links, image embeds corrected,
    missing frontmatter filled. Destructive fixes (file deletion, e.g.
    orphan-attachments) require an explicit --fix-destructive — plain
    --fix never deletes files (p863 wiki roundup 2026-09-10).

    Examples:
        kompend lint check                  # structural rules, fix what's possible
        kompend lint check --no-fix         # report only
        kompend lint check --fix-destructive  # also delete orphan attachments
        kompend lint check --rule broken-wikilinks
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config)
    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {vp.output_dir}. Run 'kompend compile' first.")
        raise typer.Exit(1)

    start = time.monotonic()
    runner = LintRunner(
        rules=rule or CHECK_STEP_RULES,
        enable_llm=False,
        lint_agent=None,
        fix=fix,
        fix_destructive=fix_destructive,
        sources_dir=str(vp.sources_dir) if vp.sources_dir.exists() else None,
        lint_config=LintConfig(fuzzy_threshold=float(cfg.kompend.lint.fuzzy_threshold)),
        rumdl_config=resolve_rumdl_config(vp.base_dir, cfg.kompend.compile.output.rumdl_config),
    )
    report = run_runner_with_progress(runner, wiki, show_progress=None)
    elapsed = time.monotonic() - start
    if fix and any(r.fix_applied for r in report.results):
        updated = persist_lint_fixes(wiki, vp)
        get_console().print(f"Persisted fixes to {updated} file(s).")

    _persist_history(report, vp=vp, cfg=cfg, context="check", elapsed=elapsed, fix=fix, node_count=len(wiki.nodes), llm_meta=None)
    _print_report(report, json_output, vp)


@lint_app.command("semantic", help="Run LLM-powered lint rules (factual-consistency, clarity, coverage-gap, contradictions).")
def lint_semantic(
    ctx: typer.Context,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Run the LLM-powered lint rules against the compiled wiki.

    Requires an LLM: compile.lint.model, falling back to
    compile.llm.default_model. Read-only — findings feed `kompend lint gaps`
    and `kompend lint remediate`.

    Examples:
        kompend lint semantic               # LLM rules only
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config)
    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {vp.output_dir}. Run 'kompend compile' first.")
        raise typer.Exit(1)

    lint_agent, model = _make_lint_agent(cfg)
    if lint_agent is None:
        get_err_console().print(
            "LLM lint rules require a model. Set compile.lint.model or compile.llm.default_model in config.",
        )
        raise typer.Exit(1)

    report, elapsed = _run_rules_pass(wiki, vp, cfg, rules=list(LLM_RULE_BY_NAME.keys()), enable_llm=True, lint_agent=lint_agent, fix=False)
    llm_meta = {MetadataField.MODEL: model, "usage": lint_agent.usage.to_dict()} if lint_agent.usage is not None else None
    _persist_history(report, vp=vp, cfg=cfg, context="semantic", elapsed=elapsed, fix=False, node_count=len(wiki.nodes), llm_meta=llm_meta)
    _print_report(report, json_output, vp)


def _persist_history(
    report: LintReport,
    *,
    vp: Any,
    cfg: Any,
    context: str,
    elapsed: float,
    fix: bool,
    node_count: int,
    llm_meta: dict[str, Any] | None,
) -> None:
    options: dict[str, Any] = {"context": context, "fix": fix}
    entry = entry_from_report(
        report,
        duration_seconds=elapsed,
        options=options,
        config=cfg.raw,
        node_count=node_count,
        llm=llm_meta,
    )
    save_lint_entry(entry, vp.kompend_dir / LINT_HISTORY_FILE)


def _print_report(report: LintReport, json_output: bool, vp: Any) -> None:
    if json_output:
        get_console().print(report_to_json(report))
    else:
        get_console().print(report_to_text(report))
    log_stage(
        vp,
        action="Lint",
        detail=f"Health score {report.health_score:.1f}% ({report.passes}/{report.total_checks} checks passed)",
    )


@lint_app.command("all", help="Run the full lint pipeline: check, format, semantic, gaps, remediate.")
def lint_all(
    ctx: typer.Context,
    fix: LintFix = True,
    fix_destructive: LintFixDestructive = False,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Run every lint step in one flow with a single weighted progress bar.

    Steps (deterministic steps weight 1/item, LLM steps weight 100/item):

    | Step      | What                                    | LLM |
    |-----------|-----------------------------------------|-----|
    | check     | structural rules, auto-fix              | no  |
    | format    | rumdl formatting                        | no  |
    | semantic  | factual/clarity/coverage/contradiction  | yes |
    | gaps      | gap analysis + semantic source check    | no  |
    | remediate | generate missing pages from extractions | yes |

    Destructive fixes (file deletion, e.g. orphan-attachments) require an
    explicit --fix-destructive — plain --fix never deletes files
    (p863 wiki roundup 2026-09-10).

    LLM steps are skipped gracefully (with a note) when no model is
    configured (compile.lint.model / compile.llm.default_model).

    Examples:
        kompend lint all                    # everything, auto-fix on
        kompend lint all --no-fix           # report only
        kompend lint all --json
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config)
    merged_dict = run_full_lint_flow(vp, cfg, fix=fix, fix_destructive=fix_destructive)
    if json_output:
        get_console().print(json.dumps(merged_dict, indent=2, ensure_ascii=False))
    else:
        get_console().print(
            f"Lint complete: health {merged_dict.get('health_score', 0.0):.1f}% "
            f"({merged_dict.get('passes', 0)} pass / {merged_dict.get('failures', 0)} fail / "
            f"{merged_dict.get('skips', 0)} skip)"
        )
        log_stage(
            vp,
            action="Lint",
            detail=f"Health score {merged_dict.get('health_score', 0.0):.1f}% ({merged_dict.get('passes', 0)} checks passed)",
        )


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
        kompend lint history              # last 10 runs
        kompend lint history -n 5         # last 5 runs
        kompend lint history --trend      # with score sparkline
        kompend lint history --json       # machine-readable
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    state_dir = vp.kompend_dir

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
    """Remove old entries from ``.kompend/lint_history.jsonl``.

    At least one of --keep-last or --max-age-days must be provided.

    Examples:
        kompend lint prune --keep-last 50        # keep only 50 newest
        kompend lint prune --max-age-days 30     # remove entries older than 30 days
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
    state_dir = vp.kompend_dir

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
    semantic: bool = typer.Option(
        True,
        "--semantic/--no-semantic",
        help="Verify missing topics against source chunks via the embedding model (needs compile.embedding.model).",
    ),
    fix: bool = typer.Option(False, "--fix", help="Generate pages for source-covered missing topics (same as lint remediate)."),
    config: ConfigPath = None,
) -> None:
    """Analyze the latest lint run to identify missing knowledge and recommend sources.

    Reads the most recent entry from ``.kompend/lint_history.jsonl`` and extracts
    actionable gap signals: missing topics (broken wikilinks), coverage gaps,
    poor-quality pages, empty pages, orphan topics, and contradictions.

    Missing-topic findings are coverage-weighted: referenced by many pages or
    by a hub page → HIGH (knowledge-core hole); a single leaf-page reference
    → LOW (fringe term).

    With the embedding model configured, missing topics are additionally
    verified against the source chunks: a topic the sources already discuss
    is reclassified from "add a source" to "covered — generate the page"
    (run ``kompend lint remediate`` or pass ``--fix``).

    Examples:
        kompend lint gaps              # analyze latest lint run
        kompend lint gaps --json       # machine-readable output
        kompend lint gaps --fix        # generate pages for source-covered topics
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    state_dir = vp.kompend_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        get_err_console().print("No lint history found. Run `kompend lint run` or compile with --lint first.")
        raise typer.Exit(1)

    report_dict = entries[-1].get("report", {})
    report = report_from_dict(report_dict)

    # Coverage weighting: node degree from the cached wiki graph.
    degree_map = load_degree_map(vp.base_dir)
    recs = analyze_gaps(report, degree_map)

    # Semantic source-coverage verification for missing topics.
    embed_model = cfg.kompend.compile.embedding.model
    covered_topics: dict[str, Any] = {}
    if semantic and embed_model:
        missing = [r for r in recs if r.resolution == "missing" and r.category == "Missing topic"]
        topics: list[str] = []
        for r in missing:
            m = re.match(r"'(.+?)'", r.finding)
            topics.append(m.group(1) if m else r.finding)
        coverage = verify_topics(topics, vp.base_dir, embed_model)
        for rec in missing:
            m = re.match(r"'(.+?)'", rec.finding)
            topic = m.group(1) if m else rec.finding
            cov = coverage.get(topic)
            if cov is not None and cov.covered:
                covered_topics[topic] = {"score": cov.best_score, "source": cov.best_source}
                rec.category = "Covered in sources (no page)"
                rec.resolution = "source-covered"
                rec.action = (
                    f"Sources already cover this topic (best match: {cov.best_source}, "
                    f"score {cov.best_score:.2f}) — run `kompend lint remediate` to generate the page."
                )
    elif semantic and not embed_model:
        get_err_console().print("Note: semantic coverage check skipped — no compile.embedding.model configured.\n")

    if fix:
        _run_remediation_for_gaps(ctx, vp, cfg, report)

    if json_output:
        payload: list[dict[str, Any]] = [
            {"priority": r.priority, "category": r.category, "finding": r.finding, "action": r.action, "resolution": r.resolution} for r in recs
        ]
        if covered_topics:
            payload.append({"category": "_semantic_coverage", "topics": covered_topics})
        get_console().print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        ts = entries[-1].get("timestamp", "unknown")
        score = report_dict.get("health_score", 0.0)
        get_console().print(f"\nBased on lint run from {ts} (score: {score:.0f}%)\n")
        get_console().print(gaps_to_text(recs))


def _run_remediation_for_gaps(ctx: typer.Context, vp: Any, cfg: Any, report: Any) -> None:
    """Run the remediation flow (same as `kompend lint remediate`) after analysis."""
    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        get_err_console().print("--fix: no compiled wiki found — run 'kompend compile' first.")
        return
    default_model = cfg.kompend.compile.llm.default_model
    if not default_model:
        get_err_console().print("--fix: no compile.llm.default_model configured — cannot generate pages.")
        return
    agent = make_agentic_remediation_agent(model_string=default_model, timeout=float(cfg.kompend.compile.llm.timeout))
    actions = remediate(
        wiki,
        report,
        base_dir=vp.base_dir,
        output_dir=vp.output_dir,
        agent=agent,
        dry_run=False,
    )
    applied = [a for a in actions if a.status == RemediationStatus.APPLIED]
    if applied:
        write_wiki_cache(wiki, vp.base_dir)
    get_console().print(remediation_to_text(actions))


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
    are skipped — ``kompend lint gaps`` shows recommendations for those.

    Examples:
        kompend lint remediate                          # generate pages
        kompend lint remediate --dry-run                # preview without writing
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    output_dir = vp.output_dir
    state_dir = vp.kompend_dir
    base_dir = vp.base_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        get_err_console().print("No lint history found. Run `kompend lint run` or compile with --lint first.")
        raise typer.Exit(1)

    report_dict = entries[-1].get("report", {})
    report = report_from_dict(report_dict)

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'kompend compile' first.")
        raise typer.Exit(1)

    default_model = cfg.kompend.compile.llm.default_model
    agent = make_agentic_remediation_agent(model_string=default_model, timeout=float(cfg.kompend.compile.llm.timeout)) if default_model else None

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
    (``compile.output.rumdl_config``, default ``.kompend/rumdl.toml``). Works
    without a compiled-wiki cache — the files on disk are the source of
    truth. ``--check`` exits 1 if any formatting changes are needed (CI).

    Examples:
        kompend lint format                 # format generated markdown
        kompend lint format --check         # CI: exit 1 if changes needed
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config,
    )
    rumdl_cfg = resolve_rumdl_config(vp.base_dir, cfg.kompend.compile.output.rumdl_config)
    extra_files: list[Path] = []
    for name in (INDEX_FILE, "overview.md"):
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
            get_err_console().print(
                f"{len(changed)} file(s) need formatting. Run 'kompend lint format' to fix.",
            )
            raise typer.Exit(1)
        get_console().print("All generated markdown is formatted.")
        return
    if changed:
        get_console().print(f"Formatted {len(changed)} file(s) with rumdl.")
    else:
        get_console().print("No formatting changes needed.")
