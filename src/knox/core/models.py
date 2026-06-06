"""Data models for the KNOX pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class ElementType(Enum):
    """Document element types for structure-preserving extraction."""

    TEXT = auto()
    HEADING = auto()
    TABLE = auto()
    IMAGE = auto()
    FORMULA = auto()
    LIST_ITEM = auto()
    CODE = auto()
    METADATA = auto()


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
    metadata: dict[str, Any] = field(default_factory=dict)
