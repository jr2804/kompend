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


class MetadataField(StrEnum):
    """Canonical keys for ``WikiNode.metadata`` and provenance data dicts.

    Every metadata / source-file / provenance key that crosses file boundaries
    must live here, not as a bare literal.
    """

    SOURCE = "source"
    SOURCE_FILE = "source_file"
    SOURCE_PATH = "source_path"
    SOURCE_COUNT = "source_count"
    SOURCE_NAMES = "source_names"
    HEADING_CONTEXT = "heading_context"
    READING_ORDER = "reading_order"
    PROVENANCE = "provenance"
    IMAGE_PATH = "image_path"
    IMAGE_BASE64 = "image_base64"


class WikiField(StrEnum):
    """Dict key contract for ``CompiledWiki`` JSON serialization.

    Keys written by :meth:`CompiledWiki.to_dict` (or built by pipeline) and
    read by :func:`CompiledWiki.from_dict` and ``web/server.py``.
    """

    NODES = "nodes"
    EDGES = "edges"
    TOC = "toc"
    QUALITY_SCORE = "quality_score"
    ITERATION = "iteration"
    CHUNK_COUNT = "chunk_count"
    DURATION_SECONDS = "duration_seconds"
    LLM_USAGE = "llm_usage"
    LLM_MODEL = "llm_model"
    LLM_DEGRADATION = "llm_degradation"
    TIER_TIMINGS = "tier_timings"
    METADATA = "metadata"


class ChunkField(StrEnum):
    """Dict key contract for ``Chunk`` JSON serialization."""

    ID = "id"
    TEXT = "text"
    ELEMENT_TYPES = "element_types"
    SOURCE_ELEMENTS = "source_elements"
    SOURCE_FILE = "source_file"
    HEADING_CONTEXT = "heading_context"
    TOKEN_COUNT = "token_count"  # noqa: S105
    READING_ORDER = "reading_order"
    METADATA = "metadata"


class ChunkNodeField(StrEnum):
    """Dict key contract for ``ChunkNode`` JSON serialization."""

    ID = "id"
    PARENT_ID = "parent_id"
    CHILDREN_IDS = "children_ids"
    DEPTH = "depth"
    HEADING_PATH = "heading_path"
    CHUNK = "chunk"
    STRUCTURED = "structured"
    SUMMARY = "summary"
    TOKEN_COUNT = "token_count"  # noqa: S105


class ChunkTreeField(StrEnum):
    """Dict key contract for ``ChunkTree`` JSON serialization."""

    VERSION = "version"
    ROOT_ID = "root_id"
    NODES = "nodes"
    SOURCE_FILE = "source_file"
    STRATEGY = "strategy"


class StructuredField(StrEnum):
    """Dict key contract for ``StructuredData`` JSON serialization.

    Shared discriminator key plus per-variant keys. Each variant reuses
    ``_TYPE`` for its ``_type`` discriminator and inherits common fields
    like ``caption``.
    """

    _TYPE = "_type"
    TYPE_TABLE = "table"
    TYPE_EQUATION = "equation"
    TYPE_FIGURE = "figure"
    COLUMNS = "columns"
    ROWS = "rows"
    CAPTION = "caption"
    LATEX = "latex"
    VARIABLES = "variables"
    NAME = "name"
    DEFINITION = "definition"
    UNITS = "units"
    ALT_TEXT = "alt_text"
    IMAGE_REF = "image_ref"
    DESCRIPTION = "description"


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            ChunkField.ID: self.id,
            ChunkField.TEXT: self.text,
            ChunkField.ELEMENT_TYPES: [str(et) for et in self.element_types],
            ChunkField.SOURCE_ELEMENTS: self.source_elements,
            ChunkField.SOURCE_FILE: self.source_file.as_posix() if self.source_file else None,
            ChunkField.HEADING_CONTEXT: self.heading_context,
            ChunkField.TOKEN_COUNT: self.token_count,
            ChunkField.READING_ORDER: list(self.reading_order) if self.reading_order else None,
            ChunkField.METADATA: self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        src = d.get(ChunkField.SOURCE_FILE) or d.get(MetadataField.SOURCE_PATH)
        return cls(
            id=d[ChunkField.ID],
            text=d[ChunkField.TEXT],
            element_types=[ElementType(et) for et in d.get(ChunkField.ELEMENT_TYPES, [])],
            source_elements=d.get(ChunkField.SOURCE_ELEMENTS, []),
            source_file=Path(src) if src else None,
            heading_context=d.get(ChunkField.HEADING_CONTEXT, ""),
            token_count=d.get(ChunkField.TOKEN_COUNT, 0),
            reading_order=tuple(d[ChunkField.READING_ORDER]) if d.get(ChunkField.READING_ORDER) else None,
            metadata=d.get(ChunkField.METADATA, {}),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            StructuredField._TYPE: StructuredField.TYPE_TABLE.value,
            StructuredField.COLUMNS: self.columns,
            StructuredField.ROWS: self.rows,
            StructuredField.CAPTION: self.caption,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredTable:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            columns=d.get(StructuredField.COLUMNS, []),
            rows=d.get(StructuredField.ROWS, []),
            caption=d.get(StructuredField.CAPTION, ""),
        )


@dataclass
class StructuredEquation:
    """Equation preserved as LaTeX + parsed variables."""

    latex: str
    variables: list[EquationVariable] = field(default_factory=list)
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            StructuredField._TYPE: StructuredField.TYPE_EQUATION.value,
            StructuredField.LATEX: self.latex,
            StructuredField.VARIABLES: [
                {
                    StructuredField.NAME: variable.name,
                    StructuredField.DEFINITION: variable.definition,
                    StructuredField.UNITS: variable.units,
                }
                for variable in self.variables
            ],
            StructuredField.CAPTION: self.caption,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredEquation:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        variables = [
            EquationVariable(
                name=v.get(StructuredField.NAME, ""),
                definition=v.get(StructuredField.DEFINITION, ""),
                units=v.get(StructuredField.UNITS, ""),
            )
            for v in d.get(StructuredField.VARIABLES, [])
        ]
        return cls(
            latex=d.get(StructuredField.LATEX, ""),
            variables=variables,
            caption=d.get(StructuredField.CAPTION, ""),
        )


@dataclass
class StructuredFigure:
    """Figure with caption, alt text, and optional VLM description."""

    caption: str
    alt_text: str = ""
    image_ref: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            StructuredField._TYPE: StructuredField.TYPE_FIGURE.value,
            StructuredField.CAPTION: self.caption,
            StructuredField.ALT_TEXT: self.alt_text,
            StructuredField.IMAGE_REF: self.image_ref,
            StructuredField.DESCRIPTION: self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredFigure:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            caption=d.get(StructuredField.CAPTION, ""),
            alt_text=d.get(StructuredField.ALT_TEXT, ""),
            image_ref=d.get(StructuredField.IMAGE_REF, ""),
            description=d.get(StructuredField.DESCRIPTION, ""),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            ChunkNodeField.ID: self.id,
            ChunkNodeField.PARENT_ID: self.parent_id,
            ChunkNodeField.CHILDREN_IDS: self.children_ids,
            ChunkNodeField.DEPTH: self.depth,
            ChunkNodeField.HEADING_PATH: list(self.heading_path),
            ChunkNodeField.CHUNK: self.chunk.to_dict() if self.chunk else None,
            ChunkNodeField.STRUCTURED: self.structured.to_dict() if self.structured else None,
            ChunkNodeField.SUMMARY: self.summary,
            ChunkNodeField.TOKEN_COUNT: self.token_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkNode:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            id=d[ChunkNodeField.ID],
            parent_id=d.get(ChunkNodeField.PARENT_ID),
            children_ids=d.get(ChunkNodeField.CHILDREN_IDS, []),
            depth=d.get(ChunkNodeField.DEPTH, 0),
            heading_path=tuple(d.get(ChunkNodeField.HEADING_PATH, ())),
            chunk=Chunk.from_dict(d[ChunkNodeField.CHUNK]) if d.get(ChunkNodeField.CHUNK) else None,
            structured=_structured_from_dict(d[ChunkNodeField.STRUCTURED]) if d.get(ChunkNodeField.STRUCTURED) else None,
            summary=d.get(ChunkNodeField.SUMMARY, ""),
            token_count=d.get(ChunkNodeField.TOKEN_COUNT, 0),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for caching."""
        return {
            ChunkTreeField.VERSION: CHUNK_TREE_CACHE_VERSION,
            ChunkTreeField.ROOT_ID: self.root_id,
            ChunkTreeField.NODES: {nid: n.to_dict() for nid, n in self.nodes.items()},
            ChunkTreeField.SOURCE_FILE: self.source_file.as_posix() if self.source_file else None,
            ChunkTreeField.STRATEGY: str(self.strategy),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkTree:
        """Deserialize from a dict produced by :meth:`to_dict`.

        Raises ``ValueError`` on cache-version mismatch so callers can
        regenerate instead of reading a stale format.
        """
        version = d.get(ChunkTreeField.VERSION, 0)
        if version != CHUNK_TREE_CACHE_VERSION:
            msg = f"ChunkTree cache version mismatch: expected {CHUNK_TREE_CACHE_VERSION}, got {version}"
            raise ValueError(msg)
        src = d.get(ChunkTreeField.SOURCE_FILE) or d.get(MetadataField.SOURCE_PATH)
        return cls(
            root_id=d[ChunkTreeField.ROOT_ID],
            nodes={nid: ChunkNode.from_dict(n) for nid, n in d.get(ChunkTreeField.NODES, {}).items()},
            source_file=Path(src) if src else None,
            strategy=ChunkStrategy(d.get(ChunkTreeField.STRATEGY, ChunkStrategy.SINGLE_PASS)),
        )

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


# ── ChunkTree serialization ─────────────────────────────────────────────

CHUNK_TREE_CACHE_VERSION: int = 1
"""Increment when ``ChunkTree.to_dict`` / ``ChunkTree.from_dict`` format changes."""


def _structured_to_dict(s: StructuredData) -> dict[str, Any]:
    """Serialize a StructuredData variant with the canonical discriminator."""
    if isinstance(s, StructuredTable):
        return s.to_dict()
    if isinstance(s, StructuredEquation):
        return s.to_dict()
    if isinstance(s, StructuredFigure):
        return s.to_dict()
    raise TypeError(f"Unknown StructuredData type: {type(s).__name__}")


def _structured_from_dict(d: dict[str, Any]) -> StructuredData:
    """Deserialize a StructuredData variant using the canonical discriminator."""
    kind = d.get(StructuredField._TYPE)
    if kind == StructuredField.TYPE_TABLE.value:
        return StructuredTable.from_dict(d)
    if kind == StructuredField.TYPE_EQUATION.value:
        return StructuredEquation.from_dict(d)
    if kind == StructuredField.TYPE_FIGURE.value:
        return StructuredFigure.from_dict(d)
    raise ValueError(f"Unknown StructuredData _type: {kind!r}")


# ── Provenance ────────────────────────────────────────────────────────────


_CITATION_RE = re.compile(r"\[([^\]]+?):(\d+)-(\d+)\]")
# TODO: make StrEnum for citation types (e.g. "source", "reference", "footnote") if we ever need to distinguish them
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
    """A single ``[source:start-end]`` citation."""

    source: str
    start: int
    end: int
    context: str = ""

    def to_marker(self) -> str:
        return f"[{self.source}:{self.start}-{self.end}]"


def extract_citations(text: str) -> list[Citation]:
    """Parse all ``[source:42-58]`` citations in *text*."""
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
