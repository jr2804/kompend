"""Citation parsing and lifecycle-state constants.

Citations are ``[source:start-end]`` markers embedded in wiki content that
link back to the originating source span. Lifecycle states track where a
wiki node sits in the review/verify/decay lifecycle.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[([^\]]+?):(\d+)-(\d+)\]")

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


__all__ = [
    "LIFECYCLE_ARCHIVED",
    "LIFECYCLE_DRAFT",
    "LIFECYCLE_REVIEWED",
    "LIFECYCLE_STALE",
    "LIFECYCLE_VERIFIED",
    "VALID_LIFECYCLE_STATES",
    "Citation",
    "extract_citations",
    "merge_duplicate_citations",
]


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
