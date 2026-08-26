"""Compiled wiki graph models (Tier 4: Compilation output) and cache loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import yaml

from knox.core.models.enums import WikiField
from knox.core.paths import VaultPaths

__all__ = [
    "CompiledWiki",
    "CrossReferenceIssue",
    "CrossReferenceResult",
    "GraphEdge",
    "TocEntry",
    "WikiNode",
    "load_cached_wiki",
    "load_wiki_from_files",
]

# Frontmatter ``kind`` extension maps back to the KNOX node_type.
_KIND_TO_NODE_TYPE = {
    "Concept": "concept",
    "Entity": "entity",
    "Overview": "overview",
    "Procedure": "procedure",
    "Section": "section",
}

# Non-content pages that are not wiki nodes.
_SKIP_PAGES = {"index.md", "log.md", "references.md"}

# Links: ``[title](path)`` and ``[[wikilink]]``.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\.md\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


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
class ExtractionStats:
    """Per-source extraction timing and counts metadata."""

    source: str = ""
    summary: str = ""
    duration_s: float = 0.0
    summary_chars: int = 0
    summary_words: int = 0
    entity_count: int = 0
    concept_count: int = 0
    tag_count: int = 0
    batches: int = 0


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
    edges: list[GraphEdge]
    toc: list[TocEntry]
    quality_score: float = 0.0
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphEdge(TypedDict):
    """Schema for one entry in ``CompiledWiki.edges``.

    ``entity`` is only present on ``entity_reference`` edges;
    ``confidence`` only on typed relation edges.
    """

    source: str
    target: str
    type: str
    entity: NotRequired[str]
    confidence: NotRequired[float]


class TocEntry(TypedDict):
    """Schema for one entry in ``CompiledWiki.toc``."""

    title: str
    id: str
    level: int


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


def load_wiki_from_files(output_dir: Path) -> CompiledWiki | None:
    """Load a compiled wiki from the generated markdown files alone.

    **Primary loader** for query/read paths — the markdown bundle is the
    source of truth (markdown-only invariant), with no ``.knox/`` state
    and no LLM. Builds nodes from the ``.md`` files on disk (frontmatter +
    body + links). Returns ``None`` when *output_dir* has no pages.
    """
    if not output_dir.is_dir():
        return None

    nodes: list[WikiNode] = []
    edges: list[GraphEdge] = []
    slug_to_title: dict[str, str] = {}

    files = sorted(p for p in output_dir.rglob("*.md") if p.is_file() and p.name not in _SKIP_PAGES)
    if not files:
        return None

    for path in files:
        slug = path.stem
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        title = str(frontmatter.get("title") or slug.replace("-", " ").title())
        kind = str(frontmatter.get("kind") or _KIND_TO_NODE_TYPE.get(str(frontmatter.get("type", "")), "concept"))
        metadata: dict[str, Any] = {}
        if frontmatter.get("description"):
            metadata["description"] = frontmatter["description"]
        tags = frontmatter.get("tags")
        if tags:
            metadata["tags"] = tags if isinstance(tags, list) else [tags]
        source_files = frontmatter.get("source_files")
        if source_files:
            metadata["provenance"] = source_files if isinstance(source_files, list) else [source_files]

        nodes.append(WikiNode(id=slug, title=title, content=body.strip(), node_type=kind, metadata=metadata))
        slug_to_title[slug] = title

    # Edges from body links: ``[title](concepts/x.md)`` and ``[[x]]``.
    title_to_slug = {title: slug for slug, title in slug_to_title.items()}
    for node in nodes:
        body = node.content
        targets: set[str] = set()
        for m in _MD_LINK_RE.finditer(body):
            target_slug = Path(m.group(2)).stem
            targets.add(target_slug)
        for m in _WIKILINK_RE.finditer(body):
            target = m.group(1).strip()
            targets.add(title_to_slug.get(target, target))
        for target in targets:
            if target in slug_to_title and target != node.id:
                edges.append(GraphEdge(source=node.id, target=target, type="reference"))

    return CompiledWiki(nodes=nodes, edges=edges, toc=[], metadata={})


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split markdown into (frontmatter dict, body). Empty dict when absent."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    try:
        fm = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}, text
    return (fm if isinstance(fm, dict) else {}), text[end + 4 :].lstrip("\n")
