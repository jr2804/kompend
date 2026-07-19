"""Doctor, status, list-parsers, and trace (provenance) diagnostic commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
from knox.core.models import MetadataField, WikiNode, load_cached_wiki
from knox.logging import get_console, get_err_console
from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers

# ── doctor ──────────────────────────────────────────────────────────────


@app.command(name="doctor")
def doctor_cmd(
    ctx: typer.Context,
    wiki_dir: str | None = typer.Option(
        None,
        "--wiki-dir",
        "-w",
        help="Project root directory. Default: current working directory or KNOX_WIKI_DIR.",
        envvar="KNOX_WIKI_DIR",
    ),
    verbose: Verbose = False,
    json_output: JsonOutput = False,
    config: ConfigPath = None,
) -> None:
    """Validate configuration and connectivity for a wiki.

    Requires a wiki context: either ``--wiki-dir`` must be set explicitly,
    or the current directory must contain a compiled wiki (i.e. the
    resolved output directory must exist).
    """
    # Prefer explicit --wiki-dir on this subcommand over the root callback value
    wiki_dir_override = wiki_dir or (ctx.obj.get("wiki_dir") if ctx.obj else None)
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override), config_path=config)
    sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    if not wiki_dir_override and not output_dir.exists():
        get_err_console().print(
            f"No wiki found at {output_dir}. Use --wiki-dir or run from a wiki directory.",
        )
        raise typer.Exit(1)

    results = run_doctor(
        sources_dir=sources_dir,
        output_dir=output_dir,
        default_model=cfg.get(COMPILE_LLM_DEFAULT_MODEL),
        vlm_model=cfg.get(COMPILE_LLM_VLM_MODEL),
        embed_model=cfg.get(COMPILE_EMBEDDING_MODEL),
        cfg=cfg.raw,
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
    sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)
    stats = collect_stats(
        sources_dir=sources_dir if sources_dir.exists() else None,
        output_dir=output_dir if output_dir.exists() else None,
        state_dir=state_dir if state_dir.exists() else None,
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
        get_console().print("No parsers registered.")
        return

    get_console().print(f"Registered parsers ({len(parsers)}):")
    for p in parsers:
        exts = ", ".join(p.get("extensions", []) or [])
        ptype = p.get("type", "built-in")
        get_console().print(f"  • {p['name']} ({ptype})  [{exts}]", markup=False)


# ── trace (provenance) ──────────────────────────────────────────────────────────────


def _shorten(source: str, sources_dir: Path) -> str:
    """Shorten an absolute source path to relative-POSIX or basename form."""
    p = Path(source)
    if not p.is_absolute():
        return source
    try:
        return p.relative_to(sources_dir).as_posix()
    except ValueError:
        return p.name


def _node_sources(node: WikiNode, sources_dir: Path) -> list[str]:
    """Effective source list for a wiki node.

    Prefers ``source_chunks`` (resolved chunk references, used by the
    graph compiler). Falls back to ``metadata['provenance']`` (source-file
    paths cited by the two-phase compiler, which does not track per-chunk
    provenance). Absolute paths under ``sources_dir`` are shortened to
    their relative form for readability.
    """
    chunks: list[str] = list(getattr(node, "source_chunks", None) or [])
    if not chunks:
        metadata = getattr(node, "metadata", None) or {}
        prov = metadata.get(MetadataField.PROVENANCE) or []
        chunks = [str(s) for s in prov if s]
    return [_shorten(s, sources_dir) for s in chunks]


# ── trace (provenance) ──────────────────────────────────────────────────


@app.command(name="trace")
def trace_cmd(
    ctx: typer.Context,
    page_slugs: list[str] = typer.Argument(None, help="Page slug(s) or title(s) to trace. Omit to list all pages with their slugs."),  # noqa: B008
    with_content: bool = typer.Option(False, "--with-content", help="Show cited content snippets."),
    json_output: JsonOutput = False,
) -> None:
    """Trace compiled wiki page(s) back to their source chunks.

    Without a slug, lists every page in the compiled wiki with its slug
    and source count — use this to discover which slugs exist before tracing.
    Pass multiple slugs to trace several pages at once.

    Example:
        knox trace                      # list all pages and their slugs
        knox trace self-attention       # show sources for one page
        knox trace self-attention bert  # show sources for multiple pages
        knox trace self-attention --with-content
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)
    base_dir = output_dir.parent

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        get_err_console().print("No compiled wiki found. Run 'knox compile' first.")
        raise typer.Exit(1)

    slugs = page_slugs or []

    # No slug — list all pages with their slugs and source counts
    if not slugs:
        pages = sorted(
            ((n.id, n.title, len(_node_sources(n, sources_dir))) for n in wiki.nodes),
            key=lambda p: p[0],
        )
        if json_output:
            json.dump(
                [{"slug": s, "title": t, "sources": c} for s, t, c in pages],
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
            return
        get_console().print(f"{len(pages)} page(s) in compiled wiki:\n")
        for slug, title, count in pages:
            get_console().print(f"  {slug:<40} ({count} sources)  {title}")
        return

    # Resolve each requested slug to its node, preserving order, deduping.
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
        # Single slug → one object for backward-friendly parsing;
        # multiple slugs → array of objects.
        entries = [{"page": n.title, "slug": n.id, "sources": _node_sources(n, sources_dir)} for n in matched]
        data = entries[0] if len(entries) == 1 else entries
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for node in matched:
            node_sources = _node_sources(node, sources_dir)
            get_console().print(f"# {node.title}\n")
            if node_sources:
                for src in node_sources:
                    get_console().print(f"  Source: {src}")
            else:
                get_console().print("  (no sources tracked)")
            get_console().print("")

    for slug in missing:
        get_err_console().print(f"Page '{slug}' not found in compiled wiki.")
    if missing:
        raise typer.Exit(1)
