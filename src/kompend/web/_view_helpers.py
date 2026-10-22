"""View helpers for the kompend web server — wiki loading and rendering."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path
from typing import Any

from kompend.core.config import DisplayConfig, resolve_config
from kompend.core.models import MetadataField, WikiField, load_cached_wiki
from kompend.core.paths import VaultPaths
from kompend.logging import get_logger
from kompend.search.retriever import SearchResult

_PROJECT_DIR: Path = Path.cwd()

logger = get_logger(__name__)


def project_dir() -> Path:
    """Return the current project directory."""
    return _PROJECT_DIR


def display_config() -> DisplayConfig:
    """Return the display limits configured for the current project."""
    return _display_config(_PROJECT_DIR)


@lru_cache(maxsize=8)
def _display_config(base_dir: Path) -> DisplayConfig:
    """Resolved display limits for *base_dir*, cached per project."""
    return resolve_config(base_dir=base_dir).kompend.display


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
    """Load the compiled wiki from the vault store snapshot."""
    empty: dict[str, Any] = {
        WikiField.NODES: [],
        WikiField.EDGES: [],
        WikiField.TOC: [],
        WikiField.QUALITY_SCORE: 0.0,
        MetadataField.NODE_COUNT: 0,
    }
    wiki = load_cached_wiki(VaultPaths(_PROJECT_DIR).base_dir)
    if wiki is None:
        return empty
    nodes = [dataclasses.asdict(node) for node in wiki.nodes]
    return {
        WikiField.NODES: nodes,
        WikiField.EDGES: list(wiki.edges),
        WikiField.TOC: list(wiki.toc),
        WikiField.QUALITY_SCORE: wiki.quality_score,
        MetadataField.NODE_COUNT: len(nodes),
    }
