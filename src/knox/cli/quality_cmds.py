"""Doctor, status, list-parsers, provenance, review, and lint commands."""

from __future__ import annotations

import json
import logging
import sys
from typing import Annotated

import typer

from knox.cli._shared import build_cli_overrides, load_hooks, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import (
    JsonOutput,
    LintFix,
    LintLlm,
    LintRule,
    LintThreshold,
    LlmContextBudget,
    LlmDefaultModel,
    Verbose,
)
from knox.cli.doctor import print_doctor_results, run_doctor
from knox.cli.status import collect_stats, print_status
from knox.core.config import resolve_config
from knox.core.hooks import HookEvent, fire_hook
from knox.core.models import load_cached_wiki
from knox.lint.llm_rules import LLM_RULE_BY_NAME
from knox.lint.runner import LintRunner, report_to_json, report_to_text
from knox.llm.client import LLMClient
from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers
from knox.review import ReviewQueue

logger = logging.getLogger(__name__)


def _uses_llm_rule(rule: list[str] | None) -> bool:
    """Return True if any requested rule name is LLM-powered."""
    if not rule:
        return False
    return any(r in LLM_RULE_BY_NAME for r in rule)


# ── doctor ──────────────────────────────────────────────────────────────


@app.command(name="doctor")
def doctor_cmd(
    ctx: typer.Context,
    verbose: Verbose = False,
    json_output: JsonOutput = False,
) -> None:
    """Validate configuration and connectivity.

    Runs health checks for source/output directories, parser availability,
    environment variables, and optionally LLM connectivity.
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir_path, output_dir_path, _cache_dir, _state_dir = resolve_project_paths(cfg)
    results = run_doctor(
        sources_dir=sources_dir_path if sources_dir_path.exists() else None,
        output_dir=output_dir_path if output_dir_path.exists() else None,
        verbose=verbose,
    )
    exit_code = print_doctor_results(results, json_output=json_output)
    if exit_code:
        raise typer.Exit(1)


# ── status ──────────────────────────────────────────────────────────────


@app.command(name="status")
def status_cmd(
    ctx: typer.Context,
    json_output: JsonOutput = False,
) -> None:
    """Show per-wiki statistics.

    Displays source count, article count, word count, last compile time,
    and config path.
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir_path, output_dir_path, _cache_dir, _state_dir = resolve_project_paths(cfg)
    stats = collect_stats(
        sources_dir=sources_dir_path if sources_dir_path.exists() else None,
        output_dir=output_dir_path if output_dir_path.exists() else None,
    )
    print_status(stats, json_output=json_output)


# ── list-parsers ────────────────────────────────────────────────────────


@app.command(name="list-parsers")
def list_parsers_cmd(
    json_output: JsonOutput = False,
) -> None:
    """List all registered parsers (built-in + external + HTTP)."""
    build_default_router()

    parsers = get_registered_parsers()

    if json_output:
        json.dump(parsers, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if not parsers:
        typer.echo("No parsers registered.")
        return

    typer.echo(f"Registered parsers ({len(parsers)}):")
    for p in parsers:
        exts = ", ".join(p.get("extensions", []) or [])
        ptype = p.get("type", "built-in")
        typer.echo(f"  • {p['name']} ({ptype})  [{exts}]")


# ── provenance ──────────────────────────────────────────────────────────


@app.command(name="provenance")
def provenance_cmd(
    ctx: typer.Context,
    page_slug: str = typer.Argument(..., help="Page slug or title to show provenance for."),
    with_content: bool = typer.Option(False, "--with-content", help="Show cited content snippets."),
    json_output: JsonOutput = False,
) -> None:
    """Show source-to-article provenance for a compiled wiki page.

    Example:
        knox provenance self-attention
        knox provenance self-attention --with-content
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    _sources_dir, output_dir_path, _cache_dir, state_dir = resolve_project_paths(cfg)
    base_dir = output_dir_path.parent
    wiki_file = output_dir_path / "wiki.md"
    if not wiki_file.exists():
        typer.echo(f"No compiled wiki found at {wiki_file}. Run 'knox compile' first.", err=True)
        raise typer.Exit(1)

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        typer.echo("Could not load compiled wiki metadata.", err=True)
        raise typer.Exit(1)

    nodes = [n for n in wiki.nodes if n.id == page_slug or n.title.lower() == page_slug.lower()]
    if not nodes:
        typer.echo(f"Page '{page_slug}' not found in compiled wiki.", err=True)
        raise typer.Exit(1)

    node = nodes[0]
    source_chunks = node.source_chunks or []

    if json_output:
        data = {"page": node.title, "slug": node.id, "sources": source_chunks}
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    typer.echo(f"# {node.title}\n")
    if source_chunks:
        for chunk_id in source_chunks:
            typer.echo(f"  Source: {chunk_id}")
    else:
        typer.echo("  (no provenance tracked)")


# ── review ──────────────────────────────────────────────────────────────


@app.command(name="review")
def review_cmd(
    ctx: typer.Context,
    action: Annotated[
        str,
        typer.Argument(help="Action: list | approve | reject | lock | unlock"),
    ],
    slug: Annotated[
        str | None,
        typer.Argument(help="Page slug (required for approve/reject/lock/unlock)."),
    ] = None,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Rejection reason (for 'reject')."),
    ] = None,
    approve_all: Annotated[
        bool,
        typer.Option("--all", help="Approve every pending page."),
    ] = False,
    json_output: JsonOutput = False,
) -> None:
    """Manage the review queue for compiled pages."""
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    _sources_dir, output_dir_path, _cache_dir, state_dir = resolve_project_paths(cfg)
    queue = ReviewQueue(review_dir=state_dir / "review")
    _dispatch_review_action(queue, action, slug, reason, approve_all, json_output)


def _review_list(queue: object, json_output: bool) -> None:
    entries = queue.list_entries()
    if json_output:
        typer.echo(json.dumps([e.to_dict() for e in entries], indent=2, default=str))
        return
    if not entries:
        typer.echo("No pages in the review queue.")
        return
    for e in entries:
        typer.echo(f"  [{e.status:9s}] {e.slug}  ({e.title})")


def _review_lock(queue: object, slug: str | None) -> None:
    if not slug:
        typer.echo("Lock requires a slug.", err=True)
        raise typer.Exit(1)
    if queue.lock(slug):
        typer.echo(f"Locked: {slug}")
        return
    typer.echo(f"Cannot lock {slug!r}: no approved page found.", err=True)
    raise typer.Exit(1)


def _review_unlock(queue: object, slug: str | None) -> None:
    lock_path = queue.review_dir / f"{slug}.locked" if slug else None
    if lock_path and lock_path.exists():
        lock_path.unlink()
        typer.echo(f"Unlocked: {slug}")
        return
    typer.echo(f"No lock for {slug!r}.", err=True)
    raise typer.Exit(1)


def _dispatch_review_action(
    queue: object,
    action: str,
    slug: str | None,
    reason: str | None,
    approve_all: bool,
    json_output: bool,
) -> None:
    """Dispatch a review command action. Extracted to keep ruff PLR0911 happy."""
    if action == "list":
        _review_list(queue, json_output)
        return

    if action == "approve":
        approved = queue.approve(slug or "", approve_all=approve_all)
        if not approved:
            typer.echo("No matching pages to approve.", err=True)
            raise typer.Exit(1)
        for e in approved:
            typer.echo(f"Approved: {e.slug}")
        return

    if action == "reject":
        if not slug:
            typer.echo("Reject requires a slug.", err=True)
            raise typer.Exit(1)
        rejected = queue.reject(slug, reason=reason or "")
        if not rejected:
            typer.echo(f"No pending page with slug {slug!r}.", err=True)
            raise typer.Exit(1)
        for e in rejected:
            typer.echo(f"Rejected: {e.slug}")
        return

    if action == "lock":
        _review_lock(queue, slug)
        return

    if action == "unlock":
        _review_unlock(queue, slug)
        return

    typer.echo(f"Unknown review action: {action!r}. Use list|approve|reject|lock|unlock.", err=True)
    raise typer.Exit(1)


# ── lint ────────────────────────────────────────────────────────────────


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
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir_path, output_dir_path, _cache_dir, _state_dir = resolve_project_paths(cfg)
    sources_dir_str = str(sources_dir_path) if sources_dir_path.exists() else None

    wiki = load_cached_wiki(output_dir_path)
    if wiki is None:
        typer.echo(f"No compiled wiki found in {output_dir_path}. Run 'knox compile' first.", err=True)
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
    report = runner.run(wiki)
    load_hooks()
    fire_hook(HookEvent.ON_LINT, report=report, wiki=wiki)

    if json_output:
        typer.echo(report_to_json(report))
    else:
        typer.echo(report_to_text(report))

    if threshold is not None and report.health_score < threshold:
        typer.echo(f"\nHealth score {report.health_score:.0f}% is below threshold {threshold}%.", err=True)
        raise typer.Exit(1)
