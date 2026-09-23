r"""Structured Markdown page construction via the wenmode AST.

:class:`PageBuilder` replaces hand-rolled ``f"## {title}\\n"`` string
assembly with typed wenmode AST nodes rendered through wenmode's
:class:`~wenmode.MarkdownRenderer`. Output formatting (table separators,
list markers, escaping) is owned by the renderer, not by call sites.

Typical use::

    b = PageBuilder()
    b.heading(2, "Overview")
    b.paragraph("Text with ", strong("emphasis"), " and ", link("concepts/x.md", "a link"), ".")
    b.bullet_list(["one", "two"])
    md = b.render()
"""

from __future__ import annotations

from collections.abc import Sequence

from wenmode import MarkdownRenderer
from wenmode.nodes import (
    Code,
    Emphasis,
    Heading,
    Image,
    InlineCode,
    Link,
    List,
    ListItem,
    Node,
    Paragraph,
    Root,
    Strong,
    Table,
    TableCell,
    TableRow,
    Text,
    ThematicBreak,
)

from knox.logging import get_logger

__all__ = [
    "PageBuilder",
    "code_span",
    "em",
    "image",
    "link",
    "strong",
]

#: Shared renderer — stateless per render, safe to reuse.
_RENDERER = MarkdownRenderer()


_ALIGN_KEYS: dict[str | None, str | None] = {
    None: None,
    "left": "left",
    "right": "right",
    "center": "center",
}

logger = get_logger(__name__)

#: Any wenmode node — broad type because wenmode ``children`` parameters
#: accept ``list[Node]``.
Span = Node


class PageBuilder:
    """Accumulates block-level AST nodes and renders them to Markdown."""

    def __init__(self) -> None:
        self._root = Root()

    # ── block builders ────────────────────────────────────────────────

    def heading(self, level: int, *content: str | Span) -> PageBuilder:
        """Add a heading (level clamped to 1–6)."""
        self._root.children.append(Heading(depth=max(1, min(level, 6)), children=_inline_children(content)))
        return self

    def paragraph(self, *content: str | Span) -> PageBuilder:
        """Add a paragraph from mixed plain text and inline spans."""
        para = Paragraph()
        para.children.extend(_inline_children(content))
        self._root.children.append(para)
        return self

    def bullet_list(self, items: Sequence[str | Span | Paragraph]) -> PageBuilder:
        """Add an unordered list; each item becomes a list-item paragraph."""
        lst = List(ordered=False)
        for item in items:
            lst.children.append(_list_item(item))
        self._root.children.append(lst)
        return self

    def numbered_list(self, items: Sequence[str | Span | Paragraph]) -> PageBuilder:
        """Add an ordered list."""
        lst = List(ordered=True)
        for item in items:
            lst.children.append(_list_item(item))
        self._root.children.append(lst)
        return self

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str | Span]],
        *,
        aligns: Sequence[str | None] | None = None,
    ) -> PageBuilder:
        """Add a GFM pipe table. ``aligns`` entries: None/left/right/center."""
        alignment = _parse_aligns(aligns, len(headers))
        tbl = Table(align=alignment)

        header_row = TableRow()
        for col in headers:
            cell = TableCell()
            cell.children.append(Text(value=col))
            header_row.children.append(cell)
        tbl.children.append(header_row)

        for row_values in rows:
            row = TableRow()
            for value in row_values:
                cell = TableCell()
                cell.children.append(Text(value=value) if isinstance(value, str) else value)
                row.children.append(cell)
            # Pad short rows so the renderer sees a rectangular grid.
            while len(row.children) < len(headers):
                cell = TableCell()
                cell.children.append(Text(value=""))
                row.children.append(cell)
            tbl.children.append(row)

        self._root.children.append(tbl)
        return self

    # Canonical PageBuilder AST method (memory #856): part of the public
    # markdown-AST surface, reachable via the builder/renderer rather than by a
    # direct in-repo call site. Do not delete as "unused".
    def code_block(self, value: str, lang: str | None = None) -> PageBuilder:
        """Add a fenced code block."""
        self._root.children.append(Code(value=value, lang=lang))
        return self

    # Canonical PageBuilder AST method (memory #856) — see note on ``code_block``.
    def thematic_break(self) -> PageBuilder:
        """Add a horizontal rule."""
        self._root.children.append(ThematicBreak())
        return self

    # ── output ────────────────────────────────────────────────────────

    def render(self) -> str:
        """Render accumulated blocks to a Markdown string."""
        return _RENDERER.render(self._root).rstrip("\n")


def strong(*content: str | Span) -> Strong:
    """Bold inline span."""
    return Strong(children=_inline_children(content))


def em(*content: str | Span) -> Emphasis:
    """Italic inline span."""
    return Emphasis(children=_inline_children(content))


def code_span(value: str) -> InlineCode:
    """Inline code span."""
    return InlineCode(value=value)


def link(url: str, *content: str | Span, title: str | None = None) -> Link:
    """Hyperlink; content defaults to the URL when omitted."""
    if not content:
        content = (url,)
    return Link(url=url, title=title, children=_inline_children(content))


# Canonical PageBuilder inline helper (memory #856) — no direct caller, but part
# of the public AST surface. Do not delete as "unused".
def image(url: str, alt: str = "", title: str | None = None) -> Image:
    """Image reference."""
    return Image(url=url, title=title, alt=alt)


def _list_item(item: str | Span | Paragraph) -> ListItem:
    """Wrap one item value in a ListItem → Paragraph structure."""
    li = ListItem()
    para = item if isinstance(item, Paragraph) else Paragraph()
    if not isinstance(item, Paragraph):
        para.children.extend(_inline_children([item]))
    li.children.append(para)
    return li


def _inline_children(content: Sequence[str | Span]) -> list[Node]:
    """Normalize mixed str/node content into a node child list."""
    children: list[Node] = []
    for item in content:
        if isinstance(item, str):
            children.append(Text(value=item))
        else:
            children.append(item)
    return children


def _parse_aligns(aligns: Sequence[str | None] | None, width: int) -> list[str | None]:
    """Validate/pad the align spec to exactly ``width`` entries."""
    if aligns is None:
        return [None] * width
    parsed = [_ALIGN_KEYS[a] for a in aligns]
    unknown = {a for a in aligns if a not in _ALIGN_KEYS}
    if unknown:
        logger.warning("Unknown table alignments %s — ignored", sorted(unknown))
    return (parsed + [None] * width)[:width]
