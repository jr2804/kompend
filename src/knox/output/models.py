"""Intermediate document model for KNOX output rendering.

``Document`` is a flavor-agnostic representation of a compiled wiki page.
It is rendered to a specific Markdown flavor by :class:`DocumentRenderer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Intermediate representation of a compiled wiki page, flavor-agnostic.

    Convert a :class:`CompiledWiki` node into ``Document`` instances,
    then pass them to :class:`DocumentRenderer` for output.
    """

    title: str
    """Page title (used as the heading)."""

    body: str
    """Markdown body with ``[[wikilinks]]`` — no flavor-specific syntax."""

    frontmatter: dict[str, Any] = field(default_factory=dict)
    """Metadata rendered as YAML frontmatter (``---`` block)."""

    heading_level: int = 1
    """Heading level for the title (1 = top-level ``#``)."""

    source_chunks: list[str] | None = None
    """Source chunk IDs this page was compiled from (for provenance)."""

    toc_entries: list[dict[str, Any]] | None = None
    """TOC entries contributed by this document (slug + title)."""

    page_slug: str = ""
    """URL-friendly slug for cross-referencing and file naming."""

    node_type: str = ""
    """Compiler-assigned kind (concept/overview/comparison/entity/...).
    Used to route the page into a subfolder in vault mode."""
