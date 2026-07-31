"""``knox trace`` — trace compiled wiki pages back to their source chunks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from knox.cli._shared import build_cli_overrides, vault_paths_from_ctx
from knox.cli.app import app
from knox.cli.args import JsonOutput
from knox.core.config import resolve_config
from knox.core.models import MetadataField, WikiNode, load_cached_wiki
from knox.logging import get_console, get_err_console


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
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir
    state_dir = vp.knox_dir
    base_dir = vp.base_dir

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


def _shorten(source: str, sources_dir: Path) -> str:
    """Shorten an absolute source path to relative-POSIX or basename form."""
    p = Path(source)
    if not p.is_absolute():
        return source
    try:
        return p.relative_to(sources_dir).as_posix()
    except ValueError:
        return p.name
