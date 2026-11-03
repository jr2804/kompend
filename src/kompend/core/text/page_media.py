"""Find the media a published page shows.

A compiled page links its figures as markdown images and its tables as
markdown tables. Both are read from the page body: the body is what the
reader sees, and the link it carries is the *published* reference (relative
to the page), not the parse-cache path the figure was extracted to. The alt
text of the link is the caption the backend or the VLM wrote.

Pure text helpers: no ``kompend.*`` imports, no state.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["find_page_figures", "find_page_tables"]

#: ``![caption](path)`` with an optional link title. The caption is the alt
#: text; an empty alt text means the page shows an image without a caption.
_FIGURE_RE: Final[re.Pattern[str]] = re.compile(r"!\[(?P<caption>[^\]]*)\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: Fenced code blocks — their content is example text, not page media.
_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^(?:```|~~~).*?^(?:```|~~~)\s*$", re.DOTALL | re.MULTILINE)

#: A markdown table needs a header row and a delimiter row.
_TABLE_DELIMITER_RE: Final[re.Pattern[str]] = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def find_page_figures(content: str) -> list[tuple[str, str]]:
    """Every image the page shows, as ``(caption, path)`` pairs.

    Document order, duplicates removed (a page that links the same image
    twice shows it once). The caption is the link's alt text — empty when the
    image carries none.
    """
    if not content:
        return []
    seen: set[str] = set()
    figures: list[tuple[str, str]] = []
    for match in _FIGURE_RE.finditer(_without_code_blocks(content)):
        path = match.group("path")
        if not path or path in seen:
            continue
        seen.add(path)
        figures.append((match.group("caption"), path))
    return figures


def find_page_tables(content: str) -> list[str]:
    """Every markdown table in the page, as its raw markdown block.

    The block is returned whole — a table is never cut short, because a
    truncated table is a wrong answer.
    """
    if not content:
        return []
    lines = _without_code_blocks(content).splitlines()
    tables: list[str] = []
    block: list[str] = []
    for line in lines:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if block:
            tables.extend(_table_blocks(block))
            block = []
    if block:
        tables.extend(_table_blocks(block))
    return tables


def _without_code_blocks(content: str) -> str:
    """The page body with its fenced code blocks removed."""
    return _FENCE_RE.sub("", content)


def _table_blocks(lines: list[str]) -> list[str]:
    """The table blocks in one run of pipe-prefixed lines.

    A table starts at the header row that precedes a delimiter row and runs
    to the end of the run: every following pipe row is a body row.
    """
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if index + 1 < len(lines) and _TABLE_DELIMITER_RE.match(lines[index + 1].strip()):
            end = index + 2
            while end < len(lines):
                end += 1
            blocks.append("\n".join(lines[index:end]))
            index = end
            continue
        index += 1
    return blocks
