"""Chunk and resolved-chunk data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.core.models.enums import ChunkField, ElementType, MetadataField


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


@dataclass
class ResolvedChunk:
    """Output of Tier 3: Resolution."""

    chunk: Chunk
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    resolved_references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["Chunk", "ResolvedChunk"]
