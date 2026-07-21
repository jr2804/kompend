"""Trafilatura-based web page parser for HTML content.

Converts HTML pages to clean Markdown using trafilatura, which outperforms
docling for general web pages (F1: 0.958 vs docling's noisy output).

Use by including ``"web"`` in the parser list, before ``"docling"``::

    parser = "web,liteparse,docling"
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import trafilatura

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import BaseParser, FileExtension


class TrafilaturaFormat(StrEnum):
    """Valid trafilatura output format strings.

    Prevents typos like ``"text"`` instead of ``"txt"`` that crash
    at runtime. See trafilatura docs for the full list.
    """

    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    XML = "xml"
    JSON = "json"



logger = get_logger(__name__)


class WebPageParser(BaseParser):
    """Convert HTML to clean Markdown via trafilatura.

    Handles ``.html`` and ``.htm`` files.  Preserves paragraphs, headings,
    lists, tables, and code blocks.  Images and formulas are not supported
    (trafilatura strips them) — for those, use the Wikipedia/arXiv PDF
    resolvers so proper PDFs reach docling instead.

    Falls back to raw text extraction if trafilatura returns nothing.
    """

    supported_extensions: frozenset[FileExtension] = frozenset({FileExtension.HTML})

    def __init__(self, *, include_tables: bool = True, include_links: bool = False) -> None:
        """
        Parameters
        ----------
        include_tables:
            Pass table content through to trafilatura's markdown output.
        include_links:
            Whether to preserve hyperlinks in the output.
        """
        self.include_tables = include_tables
        self.include_links = include_links

    def can_parse(self, source_file: Path) -> bool:  # noqa: PLR6301
        return source_file.suffix.lower() in {".html", ".htm"}

    def parse(self, source_file: Path, **kwargs: Any) -> ParsedDocument:  # noqa: ARG002
        html = source_file.read_text(encoding="utf-8", errors="replace")

        # trafilatura's markdown output preserves headings, paragraphs,
        # lists, tables, and code — much cleaner than docling on raw HTML.
        markdown = trafilatura.extract(
            html,
            output_format=TrafilaturaFormat.MARKDOWN,
            include_tables=self.include_tables,
            include_links=self.include_links,
            include_comments=False,
            no_fallback=False,  # allow readability-lxml fallback
        )

        meta = trafilatura.extract_metadata(html) or None
        title = (meta.title or "") if meta else ""
        author = (meta.author or "") if meta else ""

        if not markdown:
            # Fallback: emit the raw HTML-stripped text as a single element.
            markdown = trafilatura.extract(html, output_format=TrafilaturaFormat.TXT) or html

        elements = _markdown_to_elements(markdown, source_file)

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=title,
            author=author,
            language="en",
            page_count=1,
            metadata={"parser": "web", "url": str(source_file)},
        )


def _markdown_to_elements(markdown: str, source_file: Path) -> list[DocumentElement]:
    """Convert a trafilatura markdown string to DocumentElement list.

    Each heading becomes a HEADING element; each non-empty paragraph becomes
    a TEXT element.  Preserves reading order.
    """
    elements: list[DocumentElement] = []
    order = 0
    for block in markdown.split("\n\n"):
        block = block.strip()  # noqa: PLW2901
        if not block:
            continue
        if block.startswith("#"):
            # Count heading level from leading #
            level = len(block) - len(block.lstrip("#"))
            text = block.lstrip("#").strip()
            elements.append(
                DocumentElement(
                    type=ElementType.HEADING,
                    content=text,
                    reading_order=order,
                    heading_level=level,
                    source_file=source_file,
                )
            )
        else:
            elements.append(
                DocumentElement(
                    type=ElementType.TEXT,
                    content=block,
                    reading_order=order,
                    source_file=source_file,
                )
            )
        order += 1
    return elements


def _factory(options: ParserBuildOptions) -> WebPageParser:  # noqa: ARG001
    return WebPageParser()


register_backend("web", _factory, parser_cls=WebPageParser)
