"""Parsed document elements (Tier 1: Preprocessing output)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kompend.core.models.enums import ElementType

__all__ = ["DocumentElement", "ParsedDocument", "element_text"]

#: Rendered in place of binary content. Binary payloads cannot be embedded, and
#: their ``repr`` ("b'\\x89PNG...'") is noise that would pollute the vector
#: space with a meaningless token per page.
BINARY_PLACEHOLDER = "[binary content: {size} bytes, not text]"


def element_text(content: str | list[list[str]] | bytes) -> str:
    """Project element content onto plain text.

    ``DocumentElement.content`` is a union because parsers may hand back a
    structured table or a raw payload, but every downstream consumer -
    the parse cache, chunking, the vector index - needs text. Rendering that
    projection in one place keeps a binary payload from being stringified by
    ``str()`` somewhere upstream, which is how a repr blob reaches the embedder.

    Args:
        content: Raw element content of any of the three shapes.

    Returns:
        The text for prose, Markdown rows for a table, and an explicit
        placeholder for binary content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return BINARY_PLACEHOLDER.format(size=len(content))
    if not content:
        return ""
    # list[list[str]] - a table. Rendered as rows so cell values stay
    # embeddable rather than collapsing into a Python list repr.
    return "\n".join(" | ".join(cell for cell in row) for row in content)


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
