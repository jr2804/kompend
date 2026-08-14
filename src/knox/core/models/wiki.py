"""Compiled wiki graph models (Tier 4: Compilation output) and cache loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.core.models.enums import WikiField
from knox.core.paths import VaultPaths

__all__ = [
    "CompiledWiki",
    "CrossReferenceIssue",
    "CrossReferenceResult",
    "WikiNode",
    "load_cached_wiki",
]


@dataclass
class CrossReferenceIssue:
    """A single consistency issue found by cross-reference checking."""

    issue_type: str  # "conflict", "unresolved", "contradiction"
    description: str
    entity: str = ""
    locations: list[str] = field(default_factory=list)


@dataclass
class CrossReferenceResult:
    """Output of Phase 3 cross-reference checking at data level."""

    issues: list[CrossReferenceIssue] = field(default_factory=list)
    entity_count: int = 0
    resolved_count: int = 0


@dataclass
class WikiNode:
    """A node in the compiled knowledge graph."""

    id: str
    title: str
    content: str
    node_type: str  # e.g. "concept", "procedure", "entity", "section"
    source_chunks: list[str] | None = None
    structured_sections: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    heading_chain: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class CompiledWiki:
    """Output of Tier 4: Compilation."""

    nodes: list[WikiNode]
    edges: list[dict[str, Any]]
    toc: list[dict[str, Any]]
    quality_score: float = 0.0
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def load_cached_wiki(base_dir: Path) -> CompiledWiki | None:
    """Load the last compiled wiki from the on-disk cache.

    Args:
        base_dir: Project root directory (parent of the wiki output directory).

    Returns ``None`` if no cache exists (so callers can prompt the user to
    compile). A corrupt cache file is a bug — it raises so the real cause
    surfaces instead of producing a misleading "run compile first" message.
    Reconstructs nested ``WikiNode`` objects from their dict representation.
    """
    cache_file = VaultPaths(base_dir).wiki_cache_file
    if not cache_file.exists():
        return None
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    # Reconstruct WikiNode objects from plain dicts
    raw_nodes = data.get(WikiField.NODES, [])
    nodes = [WikiNode(**n) if isinstance(n, dict) else n for n in raw_nodes]
    return CompiledWiki(
        nodes=nodes,
        edges=data.get(WikiField.EDGES, []),
        toc=data.get(WikiField.TOC, []),
        quality_score=data.get(WikiField.QUALITY_SCORE, 0.0),
        iteration=data.get(WikiField.ITERATION, 0),
        metadata=data.get(WikiField.METADATA, {}),
    )
