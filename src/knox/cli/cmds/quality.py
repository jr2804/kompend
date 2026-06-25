"""Doctor, status, list-parsers, and provenance diagnostic commands."""

from __future__ import annotations

import json
import sys

import typer

from knox.cli._shared import build_cli_overrides, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import ConfigPath, JsonOutput, Verbose
from knox.cli.cmds.doctor import print_doctor_results, run_doctor
from knox.cli.cmds.status import collect_stats, print_status
from knox.core.config import resolve_config
from knox.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_VLM_MODEL,
)
from knox.core.models import load_cached_wiki
from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers

# ── doctor ──────────────────────────────────────────────────────────────


@app.command(name="doctor")
def doctor_cmd(
    ctx: typer.Context,
    verbose: Verbose = False,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Validate configuration and connectivity for a wiki.

    Requires a wiki context: either ``--wiki-dir`` must be set explicitly,
    or the current directory must contain a compiled wiki (i.e. the
    resolved output directory must exist).
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override), config_path=config)
    sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    if not wiki_dir_override and not output_dir.exists():
        typer.echo(
            f"No wiki found at {output_dir}. Use --wiki-dir or run from a wiki directory.",
            err=True,
        )
        raise typer.Exit(1)

    results = run_doctor(
        sources_dir=sources_dir,
        output_dir=output_dir,
        default_model=cfg.get(COMPILE_LLM_DEFAULT_MODEL),
        vlm_model=cfg.get(COMPILE_LLM_VLM_MODEL),
        embed_model=cfg.get(COMPILE_EMBEDDING_MODEL),
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
    sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)
    stats = collect_stats(
        sources_dir=sources_dir if sources_dir.exists() else None,
        output_dir=output_dir if output_dir.exists() else None,
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
    _sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)
    base_dir = output_dir.parent
    wiki_file = output_dir / "wiki.md"
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
