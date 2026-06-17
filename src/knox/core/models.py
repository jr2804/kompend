"""Data models for the KNOX pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from knox.core.paths import KNOX_DIR, WIKI_CACHE_FILE


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

    source_path: Path
    elements: list[DocumentElement]
    title: str = ""
    author: str = ""
    language: str = "en"
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Output of Tier 2: Chunking."""

    id: str
    text: str
    element_types: list[ElementType]
    source_elements: list[int]
    source_file: Path | None = None
    heading_context: str = ""
    token_count: int = 0
    reading_order: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    metadata: dict[str, Any] = field(default_factory=dict)


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


def load_cached_wiki(output_dir: Path) -> CompiledWiki | None:
    """Load the last compiled wiki from the on-disk cache.

    Returns ``None`` if no cache exists or the data is corrupt.
    Reconstructs nested ``WikiNode`` objects from their dict representation.
    """
    logger = logging.getLogger(__name__)
    cache_file = output_dir / KNOX_DIR / WIKI_CACHE_FILE
    if not cache_file.exists():
        return None
    try:
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
    except Exception as exc:
        logger.debug("Failed to load cached wiki from %s: %s", cache_file, exc, exc_info=True)
        return None
