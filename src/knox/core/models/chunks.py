"""Chunk and resolved-chunk data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from knox.core.models.enums import ChunkField, ElementType, MetadataField
from knox.core.paths import safe_stem

__all__ = ["Chunk", "ImageRef", "ResolvedChunk", "image_source_key", "images_for_provenance"]


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
        src = d.get(ChunkField.SOURCE_FILE)
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


class ImageRef(TypedDict):
    """Schema for one image reference passed through compile into output.

    Built by ``compilation._two_phase_core.extract_images_from_chunks``,
    stored in ``node.metadata["images"]``, consumed by the output writers.
    """

    path: str
    caption: str
    element_type: str
    source_file: str
    """Source document the figure was extracted from (may be a bare label).

    Carried so a page can embed *its* source's figures: without it the only
    option was the run-global image list, which put every figure of every
    source on every page (and copied them into one attachment directory per
    page provenance value).
    """


def images_for_provenance(images: list[ImageRef], provenance: list[str]) -> list[ImageRef]:
    """The subset of *images* extracted from one of *provenance*'s sources.

    A page's provenance is what the generation agent reports it drew on, so it
    decides which figures the page embeds. Provenance entries that name no
    source (agents emit venue/title labels such as ``Proc. DAGA``) match
    nothing and contribute no images — better than attaching every figure in
    the run to a page that has no source to attribute them to.
    """
    wanted = {image_source_key(p) for p in provenance if p}
    if not wanted:
        return []
    return [img for img in images if image_source_key(str(img.get(MetadataField.SOURCE_FILE, ""))) in wanted]


def image_source_key(source: str) -> str:
    """Normalize a source label or path to a comparable key.

    Compares provenance labels against image source documents, which arrive in
    different shapes: ``ECMA-418-2_4th_edition_june_2025`` (model label),
    ``sources/ECMA-418-2_4th_edition_june_2025.pdf`` (relative path), or an
    absolute path to the same file. Extension-aware basename (``safe_stem``,
    memory #779) lowercased is what they share.
    """
    return safe_stem(source).casefold()
