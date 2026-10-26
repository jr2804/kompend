r"""Element ↔ markdown fragment conversion for whole-part normalisation.

``element_to_markdown`` / ``markdown_to_element`` wrap an element's content as
the minimal markdown that makes it the right *kind* of block, and strip that
wrapper again — the pair used when an external markdown formatter (rumdl) must
run over individual element contents. Only the content changes: page, reading
order and metadata stay on the element.

This module used to also host the backend-neutral Markdown→element converter
(``markdown_to_elements`` + ``MarkdownContext``) shared by kompend-side
markdown-producing backends (liteparse, marker, mineru, glm-ocr, web). Those
backends are gone — parsecraft maps its IR to elements directly — and with
them the converter. Do not reintroduce a per-backend copy of that walk: the
copies this module replaced recognised only headings, images, tables and
paragraphs, silently swallowing equations, fenced code and list items into
TEXT. If a future backend produces Markdown, the conversion belongs beside
these patterns — never next to the SDK call.
"""

from __future__ import annotations

import re
from typing import Final

from kompend.core.models import DocumentElement, ElementType

__all__ = [
    "FENCE_RE",
    "HEADING_RE",
    "LIST_ITEM_RE",
    "element_to_markdown",
    "markdown_to_element",
]

# ── Shared patterns ────────────────────────────────────────────────────

HEADING_RE: Final = re.compile(r"^(#{1,6})\s+(.*)$")
#: ``` or ~~~ fence, with an optional info string (language).
FENCE_RE: Final = re.compile(r"^(`{3,}|~{3,})\s*([\w+#.-]*)\s*$")
#: ``- item``, ``* item``, ``+ item``, ``1. item``, ``1) item``.
LIST_ITEM_RE: Final = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
#: Block-math openers and their closers, in ``$$``/``\[`` pairs.
_MATH_DELIMITERS: Final[dict[str, str]] = {"$$": "$$", "\\[": "\\]"}  # nocheck: str-registry — LaTeX delimiter syntax, not a vocabulary

#: Tables are grids, images are references, metadata is invisible — already
#: canonical, never round-tripped through a markdown formatter.
_FRAGMENT_TYPE_SKIP: Final = frozenset({ElementType.TABLE, ElementType.IMAGE, ElementType.METADATA})


def element_to_markdown(element: DocumentElement) -> str | None:
    """Wrap an element's text content as a standalone markdown block.

    Returns ``None`` for non-text elements (tables are grids, images are
    references, metadata is invisible) — those are already canonical and must
    not be round-tripped through a markdown formatter.
    """
    if element.type in _FRAGMENT_TYPE_SKIP or not isinstance(element.content, str):
        return None
    content = element.content
    if not content.strip():
        return None
    if element.type is ElementType.HEADING:
        level = max(1, min(element.heading_level or 1, 6))
        return f"{'#' * level} {content}"
    if element.type is ElementType.LIST_ITEM:
        return f"- {content}"
    if element.type is ElementType.CODE:
        language = str((element.metadata or {}).get("language", "") or "")
        return f"```{language}\n{content}\n```"
    if element.type is ElementType.FORMULA:
        return f"$$\n{content}\n$$"
    return content


def markdown_to_element(element_type: ElementType, markdown: str) -> str | None:  # noqa: PLR0911 — one return per kind
    """Strip the wrapper added by :func:`element_to_markdown`.

    Returns the bare content, or ``None`` when *markdown* no longer has the
    expected shape (then the caller keeps the original content — never guess).
    """
    text = markdown.strip()
    if not text:
        return None
    if element_type is ElementType.HEADING:
        match = HEADING_RE.match(text.split("\n", 1)[0].strip())
        return match.group(2).strip() if match else None
    if element_type is ElementType.LIST_ITEM:
        match = LIST_ITEM_RE.match(text.split("\n", 1)[0].strip())
        return match.group(1).strip() if match else None
    if element_type is ElementType.CODE:
        lines = text.splitlines()
        if len(lines) < 2 or not FENCE_RE.match(lines[0].strip()):
            return None
        if not lines[-1].strip().startswith("```") and not lines[-1].strip().startswith("~~~"):
            return None
        return "\n".join(lines[1:-1])
    if element_type is ElementType.FORMULA:
        lines = text.splitlines()
        if len(lines) < 2 or lines[0].strip() not in _MATH_DELIMITERS or lines[-1].strip() != "$$":
            return None
        return "\n".join(lines[1:-1]).strip()
    return text
