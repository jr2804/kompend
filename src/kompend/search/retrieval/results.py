"""Result types shared by every search entry point.

``SearchResult`` is serialized with :func:`dataclasses.asdict` on the MCP
query path (``src/kompend/mcp/tools/query.py``), so its field set and order
are that payload's schema. ``Citation`` is read structurally by the CLI and
MCP renderers, so its key set is a contract too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NotRequired, TypedDict

from kompend.core.models import ImageRef


@dataclass
class SearchResult:
    article_id: str
    title: str
    snippet: str
    score: float
    source_file: str | None = None
    tags: set[str] = field(default_factory=set)
    attributes: dict[str, list[str]] = field(default_factory=dict)
    content: str = ""  # full page body for LLM synthesis; snippet is display-only
    figures: list[ImageRef] = field(default_factory=list)
    """The figures the page shows (published link + caption).

    Read from the page body, so the link is the one the reader can open and
    the caption is the one the page publishes.
    """


@dataclass
class SynthesisResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)


class Citation(TypedDict):
    """Schema for one citation in a synthesis result."""

    article: str
    article_id: str | None
    source: str | None
    snippet: str
    score: float
    figures: NotRequired[list[ImageRef]]
    """The figures the cited page shows — the image the answer talked about.

    Absent when the citation was built without the page body.
    """
