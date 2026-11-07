"""MCP provenance tool - trace compiled pages back to their source chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kompend.cli.settings import build_cli_overrides
from kompend.core.config import resolve_config
from kompend.core.models import MetadataField, load_wiki_from_files
from kompend.core.paths import VaultPaths, vault_paths


def provenance(
    wiki_dir: str | None = None,
    page_slugs: list[str] | None = None,
    with_content: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    """Trace compiled wiki page(s) back to their source chunks (provenance).

    Named ``provenance`` to avoid confusion with the OpenTelemetry tracing
    feature (see ``docs/guides/observability.md``).
    """
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    sources_dir = vp.sources_dir
    base_dir = vp.base_dir

    wiki = load_wiki_from_files(VaultPaths(base_dir).output_dir)
    if wiki is None:
        return {"status": "error", "message": "No wiki pages found. Run 'wiki_compile' first."}

    slugs = page_slugs or []

    if not slugs:
        pages = sorted(
            ((n.id, n.title, len(_node_sources(n, sources_dir))) for n in wiki.nodes),
            key=lambda p: p[0],
        )
        if json_output:
            return {"status": "ok", "pages": [{"slug": s, "title": t, "sources": c} for s, t, c in pages]}
        lines = [f"{len(pages)} page(s) in compiled wiki:\n"]
        for slug, title, count in pages:
            lines.append(f"  {slug:<40} ({count} sources)  {title}")
        return {"status": "ok", "pages_text": "\n".join(lines)}

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
        return {"status": "ok" if not missing else "error", "pages": entries[0] if len(entries) == 1 else entries, "missing": missing}

    lines = []
    for node in matched:
        node_sources = _node_sources(node, sources_dir)
        lines.append(f"# {node.title}\n")
        if node_sources:
            for src in node_sources:
                lines.append(f"  Source: {src}")
        else:
            lines.append("  (no sources tracked)")
        lines.append("")

    if missing:
        lines.append("Missing: " + ", ".join(missing))
        return {"status": "error", "pages_text": "\n".join(lines), "missing": missing}

    return {"status": "ok", "pages_text": "\n".join(lines)}


def _node_sources(node: Any, sources_dir: Path) -> list[str]:
    chunks: list[str] = list(getattr(node, "source_chunks", None) or [])
    if not chunks:
        metadata = getattr(node, "metadata", None) or {}
        prov = metadata.get(MetadataField.PROVENANCE) or []
        chunks = [str(s) for s in prov if s]
    return [_shorten(s, sources_dir) for s in chunks]


def _shorten(source: str, sources_dir: Path) -> str:
    source_path = Path(source)
    if not source_path.is_absolute():
        return source
    try:
        return source_path.relative_to(sources_dir).as_posix()
    except ValueError:
        return source_path.name
