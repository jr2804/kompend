"""View helpers for the KNOX web server — wiki loading and rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knox.core.models import MetadataField, WikiField
from knox.core.paths import VaultPaths
from knox.logging import get_logger
from knox.search.retriever import SearchResult

_PROJECT_DIR: Path = Path.cwd()

logger = get_logger(__name__)


def project_dir() -> Path:
    """Return the current project directory."""
    return _PROJECT_DIR


def render_result(r: SearchResult) -> str:
    """Render one retrieval result as an HTML card."""
    source_line = f'<br><span style="font-size:0.85rem;color:#666">Source: {r.source_file}</span>' if r.source_file else ""
    return (
        '<div style="margin-bottom:1rem;padding:1rem;background:#f9f9f9;border-radius:4px">'
        f'<strong>{r.title}</strong> <span style="color:#666;font-size:0.85rem">(score: {r.score:.3f})</span>'
        f"{source_line}"
        f"<p>{r.snippet}</p></div>"
    )


def source_of(node: dict[str, Any]) -> str:
    """Extract source file from a wiki node's metadata, or '' if absent."""
    meta = node.get(WikiField.METADATA) or {}
    return meta.get("source", "")


def load_wiki() -> dict[str, Any]:
    """Load the compiled wiki from cache."""
    cache = VaultPaths(_PROJECT_DIR).wiki_cache_path
    if not cache.exists():
        return {WikiField.NODES: [], WikiField.EDGES: [], WikiField.TOC: [], WikiField.QUALITY_SCORE: 0.0, MetadataField.NODE_COUNT: 0}
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        data[MetadataField.NODE_COUNT] = len(data.get(WikiField.NODES, []))
        return data
    except Exception as exc:
        logger.warning("Failed to load wiki cache: %s", exc, exc_info=True)
        return {WikiField.NODES: [], WikiField.EDGES: [], WikiField.TOC: [], WikiField.QUALITY_SCORE: 0.0, MetadataField.NODE_COUNT: 0}
