"""Data models for the KNOX pipeline."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias

from knox.core.paths import VaultPaths


class ElementType(StrEnum):
    """Document element types for structure-preserving extraction.

    StrEnum serializes to plain strings in JSON: ``"text"``, ``"table"``, etc.
    """

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    CODE = "code"
    METADATA = "metadata"


ATOMIC_TYPES: frozenset[ElementType] = frozenset({ElementType.TABLE, ElementType.IMAGE, ElementType.FORMULA, ElementType.CODE})
"""Element types that must never be split during chunking."""


@dataclass
class DocumentElement:
    """A single element from a parsed document (text, heading, table, etc.)."""

    type: ElementType
    content: str | list[list[str]] | bytes
    page: int | None = None
    reading_order: int | None = None
    heading_level: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: Path | None = None


@dataclass
class ParsedDocument:
    """Output of Tier 1: Preprocessing."""

    source_file: Path
    elements: list[DocumentElement]
    title: str = ""
    author: str = ""
    language: str = "en"
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Output of Tier 2: Chunking (leaf content)."""

    id: str
    text: str
    element_types: list[ElementType]
    source_elements: list[int]
    source_file: Path | None = None
    heading_context: str = ""
    token_count: int = 0
    reading_order: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Structured data types (preserved, not stringified) ──────────────────


class ChunkStrategy(StrEnum):
    """How a document was chunked — determines downstream processing depth.

    - ``SINGLE_PASS``: document fits in context window; trivial tree (root +
      leaves). No summaries needed.
    - ``TWO_LEVEL``: document is <= 8× context window; root + section nodes +
      leaves. Root-level summary generated for navigation.
    - ``MULTI_RESOLUTION``: document is >> 8× context window; full-depth tree
      with summaries at every internal node for multi-resolution navigation.
    """

    SINGLE_PASS = "single_pass"  # noqa: S105
    TWO_LEVEL = "two_level"
    MULTI_RESOLUTION = "multi_resolution"


@dataclass
class EquationVariable:
    """A single variable in an equation with its definition."""

    name: str
    definition: str = ""
    units: str = ""


@dataclass
class StructuredTable:
    """Table preserved as typed records instead of pipe-delimited text."""

    columns: list[str]
    rows: list[dict[str, str]]
    caption: str = ""


@dataclass
class StructuredEquation:
    """Equation preserved as LaTeX + parsed variables."""

    latex: str
    variables: list[EquationVariable] = field(default_factory=list)
    caption: str = ""


@dataclass
class StructuredFigure:
    """Figure with caption, alt text, and optional VLM description."""

    caption: str
    alt_text: str = ""
    image_ref: str = ""
    description: str = ""


StructuredData: TypeAlias = StructuredTable | StructuredEquation | StructuredFigure  # noqa: UP040


# ── Hierarchical chunk tree (replaces flat list[Chunk]) ─────────────────


@dataclass
class ChunkNode:
    """A node in the hierarchical chunk tree.

    Leaf nodes carry content (``chunk`` for prose, ``structured`` for
    tables/equations/figures). Internal nodes carry a ``summary`` and
    aggregate ``token_count`` of their subtree.
    """

    id: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    heading_path: tuple[str, ...] = ()
    chunk: Chunk | None = None
    structured: StructuredData | None = None
    summary: str = ""
    token_count: int = 0
    contextual_context: str = ""
    contextual_embedding: list[float] | None = None


@dataclass
class ChunkTree:
    """Hierarchical tree of ChunkNodes for one source document.

    Replaces the flat ``list[Chunk]`` produced by the legacy chunker. The tree
    mirrors the document's heading structure, enabling multi-resolution
    retrieval and structured content preservation.
    """

    root_id: str
    nodes: dict[str, ChunkNode] = field(default_factory=dict)
    source_file: Path | None = None
    strategy: ChunkStrategy = ChunkStrategy.SINGLE_PASS

    def leaves(self) -> list[ChunkNode]:
        """Return all leaf nodes (nodes with no children)."""
        return [n for n in self.nodes.values() if not n.children_ids]

    def walk(self) -> Iterator[ChunkNode]:
        """Depth-first traversal from root."""
        yield from self._walk(self.root_id)

    def get_path(self, node_id: str) -> list[ChunkNode]:
        """Return the path from root to ``node_id`` (inclusive)."""
        path: list[ChunkNode] = []
        current = self.nodes.get(node_id)
        while current is not None:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def siblings(self, node_id: str) -> list[ChunkNode]:
        """Return siblings of ``node_id`` (excluding itself)."""
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return []
        parent = self.nodes[node.parent_id]
        return [self.nodes[cid] for cid in parent.children_ids if cid != node_id]

    def subtree_tokens(self, node_id: str) -> int:
        """Total token count of a node's subtree (node + all descendants)."""
        node = self.nodes.get(node_id)
        if node is None:
            return 0
        total = node.token_count
        for child_id in node.children_ids:
            total += self.subtree_tokens(child_id)
        return total

    def _walk(self, node_id: str) -> Iterator[ChunkNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return
        yield node
        for child_id in node.children_ids:
            yield from self._walk(child_id)


# ── Cross-reference result (stub for Phase 3) ───────────────────────────


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
class ResolvedChunk:
    """Output of Tier 3: Resolution."""

    chunk: Chunk
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    resolved_references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
    heading_path: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class CompiledWiki:
    """Output of Tier 4: Compilation."""

    nodes: list[WikiNode]
    edges: list[dict[str, Any]]
    toc: list[dict[str, Any]]
    quality_score: float = 0.0
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Serialization helpers ────────────────────────────────────────────────


def load_cached_wiki(base_dir: Path) -> CompiledWiki | None:
    """Load the last compiled wiki from the on-disk cache.

    Args:
        base_dir: Project root directory (parent of the wiki output directory).

    Returns ``None`` if no cache exists (so callers can prompt the user to
    compile). A corrupt cache file is a bug — it raises so the real cause
    surfaces instead of producing a misleading "run compile first" message.
    Reconstructs nested ``WikiNode`` objects from their dict representation.
    """
    cache_file = VaultPaths(base_dir).wiki_cache_path
    if not cache_file.exists():
        return None
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    # Reconstruct WikiNode objects from plain dicts
    raw_nodes = data.get("nodes", [])
    nodes = [WikiNode(**n) if isinstance(n, dict) else n for n in raw_nodes]
    return CompiledWiki(
        nodes=nodes,
        edges=data.get("edges", []),
        toc=data.get("toc", []),
        quality_score=data.get("quality_score", 0.0),
        iteration=data.get("iteration", 0),
        metadata=data.get("metadata", {}),
    )


# ── ChunkTree serialization ─────────────────────────────────────────────

CHUNK_TREE_CACHE_VERSION: int = 1
"""Increment when ``tree_to_dict`` / ``tree_from_dict`` format changes."""


def _structured_to_dict(s: StructuredData) -> dict[str, Any]:
    """Serialize a StructuredData variant with a ``_type`` discriminator."""
    if isinstance(s, StructuredTable):
        return {"_type": "table", "columns": s.columns, "rows": s.rows, "caption": s.caption}
    if isinstance(s, StructuredEquation):
        return {
            "_type": "equation",
            "latex": s.latex,
            "variables": [{"name": v.name, "definition": v.definition, "units": v.units} for v in s.variables],
            "caption": s.caption,
        }
    if isinstance(s, StructuredFigure):
        return {
            "_type": "figure",
            "caption": s.caption,
            "alt_text": s.alt_text,
            "image_ref": s.image_ref,
            "description": s.description,
        }
    raise TypeError(f"Unknown StructuredData type: {type(s).__name__}")


def _structured_from_dict(d: dict[str, Any]) -> StructuredData:
    """Deserialize a StructuredData variant using the ``_type`` discriminator."""
    kind = d.get("_type")
    if kind == "table":
        return StructuredTable(
            columns=d.get("columns", []),
            rows=d.get("rows", []),
            caption=d.get("caption", ""),
        )
    if kind == "equation":
        variables = [
            EquationVariable(
                name=v.get("name", ""),
                definition=v.get("definition", ""),
                units=v.get("units", ""),
            )
            for v in d.get("variables", [])
        ]
        return StructuredEquation(
            latex=d.get("latex", ""),
            variables=variables,
            caption=d.get("caption", ""),
        )
    if kind == "figure":
        return StructuredFigure(
            caption=d.get("caption", ""),
            alt_text=d.get("alt_text", ""),
            image_ref=d.get("image_ref", ""),
            description=d.get("description", ""),
        )
    raise ValueError(f"Unknown StructuredData _type: {kind!r}")


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Serialize a Chunk to a JSON-safe dict."""
    return {
        "id": chunk.id,
        "text": chunk.text,
        "element_types": [str(et) for et in chunk.element_types],
        "source_elements": chunk.source_elements,
        "source_file": str(chunk.source_file) if chunk.source_file else None,
        "heading_context": chunk.heading_context,
        "token_count": chunk.token_count,
        "reading_order": list(chunk.reading_order) if chunk.reading_order else None,
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(d: dict[str, Any]) -> Chunk:
    """Deserialize a Chunk from a dict."""
    return Chunk(
        id=d["id"],
        text=d["text"],
        element_types=[ElementType(et) for et in d.get("element_types", [])],
        source_elements=d.get("source_elements", []),
        source_file=Path(d.get("source_file") or d.get("source_path")) if (d.get("source_file") or d.get("source_path")) else None,
        heading_context=d.get("heading_context", ""),
        token_count=d.get("token_count", 0),
        reading_order=tuple(d["reading_order"]) if d.get("reading_order") else None,
        metadata=d.get("metadata", {}),
    )


def _node_to_dict(node: ChunkNode) -> dict[str, Any]:
    """Serialize a ChunkNode to a JSON-safe dict."""
    return {
        "id": node.id,
        "parent_id": node.parent_id,
        "children_ids": node.children_ids,
        "depth": node.depth,
        "heading_path": list(node.heading_path),
        "chunk": _chunk_to_dict(node.chunk) if node.chunk else None,
        "structured": _structured_to_dict(node.structured) if node.structured else None,
        "summary": node.summary,
        "token_count": node.token_count,
    }


def _node_from_dict(d: dict[str, Any]) -> ChunkNode:
    """Deserialize a ChunkNode from a dict."""
    return ChunkNode(
        id=d["id"],
        parent_id=d.get("parent_id"),
        children_ids=d.get("children_ids", []),
        depth=d.get("depth", 0),
        heading_path=tuple(d.get("heading_path", ())),
        chunk=_chunk_from_dict(d["chunk"]) if d.get("chunk") else None,
        structured=_structured_from_dict(d["structured"]) if d.get("structured") else None,
        summary=d.get("summary", ""),
        token_count=d.get("token_count", 0),
    )


def tree_to_dict(tree: ChunkTree) -> dict[str, Any]:
    """Serialize a ChunkTree to a JSON-safe dict for caching."""
    return {
        "version": CHUNK_TREE_CACHE_VERSION,
        "root_id": tree.root_id,
        "nodes": {nid: _node_to_dict(n) for nid, n in tree.nodes.items()},
        "source_file": str(tree.source_file) if tree.source_file else None,
        "strategy": str(tree.strategy),
    }


def tree_from_dict(d: dict[str, Any]) -> ChunkTree:
    """Deserialize a ChunkTree from a dict produced by ``tree_to_dict``."""
    version = d.get("version", 0)
    if version != CHUNK_TREE_CACHE_VERSION:
        raise ValueError(f"ChunkTree cache version mismatch: expected {CHUNK_TREE_CACHE_VERSION}, got {version}")
    return ChunkTree(
        root_id=d["root_id"],
        nodes={nid: _node_from_dict(n) for nid, n in d.get("nodes", {}).items()},
        source_file=Path(d.get("source_file") or d.get("source_path")) if (d.get("source_file") or d.get("source_path")) else None,
        strategy=ChunkStrategy(d.get("strategy", ChunkStrategy.SINGLE_PASS)),
    )


# ── Provenance ────────────────────────────────────────────────────────────


_CITATION_RE = re.compile(r"\^\[([^\]]+?):(\d+)-(\d+)\]")

LIFECYCLE_DRAFT = "draft"
LIFECYCLE_REVIEWED = "reviewed"
LIFECYCLE_VERIFIED = "verified"
LIFECYCLE_STALE = "stale"
LIFECYCLE_ARCHIVED = "archived"

VALID_LIFECYCLE_STATES: tuple[str, ...] = (
    LIFECYCLE_DRAFT,
    LIFECYCLE_REVIEWED,
    LIFECYCLE_VERIFIED,
    LIFECYCLE_STALE,
    LIFECYCLE_ARCHIVED,
)


@dataclass
class Citation:
    """A single ``^[source:start-end]`` citation."""

    source: str
    start: int
    end: int
    context: str = ""

    def to_marker(self) -> str:
        return f"^[{self.source}:{self.start}-{self.end}]"


def extract_citations(text: str) -> list[Citation]:
    """Parse all ``^[source:42-58]`` citations in *text*."""
    out: list[Citation] = []
    for match in _CITATION_RE.finditer(text):
        out.append(
            Citation(
                source=match.group(1),
                start=int(match.group(2)),
                end=int(match.group(3)),
            )
        )
    return out


def merge_duplicate_citations(citations: Iterable[Citation]) -> list[Citation]:
    """Remove duplicates by (source, start, end), preserving first-seen order."""
    seen: set[tuple[str, int, int]] = set()
    out: list[Citation] = []
    for c in citations:
        key = (c.source, c.start, c.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
