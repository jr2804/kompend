"""The ``knox lint`` command."""

from __future__ import annotations

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
    LintFix,
    LintLlm,
    LintRule,
    LintThreshold,
    LlmContextBudget,
    LlmDefaultModel,
)
from knox.core.config import resolve_config
from knox.core.hooks import HookEvent, fire_hook
from knox.core.models import CompiledWiki, load_cached_wiki
from knox.lint.llm_rules import LLM_RULE_BY_NAME
from knox.lint.runner import LintRunner, report_to_json, report_to_text
from knox.llm.client import LLMClient
from knox.output.wiki_index import generate_stage_log_entry, prepend_log


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


@app.command(name="lint")
def lint_cmd(
    ctx: typer.Context,
    fix: LintFix = False,
    llm: LintLlm = False,
    rule: LintRule = None,
    threshold: LintThreshold = None,
    json_output: JsonOutput = False,
    llm_default_model: LlmDefaultModel = None,
    llm_context_budget: LlmContextBudget = None,
    config: ConfigPath = None,
) -> None:
    """Lint a compiled wiki for quality issues.

    Runs 8 structural rules by default. Add --llm to also run 3
    LLM-powered rules (slow, expensive). Use --rule to run only specific
    rules and --fix to auto-fix supported issues.

    Examples:
        knox lint                       # structural rules only
        knox lint --llm                 # structural + LLM rules
        knox lint --fix                 # auto-fix what's possible
        knox lint --rule broken-wikilinks --rule orphan-page
        knox lint --threshold 80        # exit 1 if score < 80
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
        typer.echo(f"No compiled wiki found in {output_dir}. Run 'knox compile' first.", err=True)
        raise typer.Exit(1)

    llm_client = None
    if llm or _uses_llm_rule(rule):
        if not llm_default_model:
            typer.echo(
                "LLM-powered lint rules require --llm-default-model (e.g. --llm-default-model openai/gpt-4o).",
                err=True,
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
        typer.echo(f"Persisted fixes to {fixes_applied} file(s).")

    # Rule summary: "<n> fail / <n> pass / <n> skip" aggregated by outcome
    outcome_counts = Counter(r.outcome.value if hasattr(r.outcome, "value") else str(r.outcome) for r in report.results)
    rule_summary = " / ".join(f"{count} {label.lower()}" for label, count in sorted(outcome_counts.items()))

    log_file = state_dir.parent / "log.md"
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

    if json_output:
        typer.echo(report_to_json(report))
    else:
        typer.echo(report_to_text(report))

    if threshold is not None and report.health_score < threshold:
        typer.echo(f"\nHealth score {report.health_score:.0f}% is below threshold {threshold}%.", err=True)
        raise typer.Exit(1)
