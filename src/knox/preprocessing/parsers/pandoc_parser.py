"""Pandoc-based parser using pypandoc."""

from __future__ import annotations

import json
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import ClassVar, override

import pypandoc

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import BaseParser, FileExtension

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pandoc JSON AST types
# ---------------------------------------------------------------------------


class _PandocBlockTag(StrEnum):
    """Block type tags from pandoc's JSON AST."""

    BULLET_LIST = "BulletList"
    CODE_BLOCK = "CodeBlock"
    HEADER = "Header"
    LINE_BLOCK = "LineBlock"
    MATH = "Math"
    ORDERED_LIST = "OrderedList"
    PARA = "Para"
    PLAIN = "Plain"
    TABLE = "Table"

    @property
    def element_type(self) -> ElementType:
        """Return the KNOX element type for this pandoc block tag."""
        return {
            "BulletList": ElementType.LIST_ITEM,
            "CodeBlock": ElementType.CODE,
            "Header": ElementType.HEADING,
            "LineBlock": ElementType.TEXT,
            "Math": ElementType.FORMULA,
            "OrderedList": ElementType.LIST_ITEM,
            "Para": ElementType.TEXT,
            "Plain": ElementType.TEXT,
            "Table": ElementType.TABLE,
        }[self.value]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class PandocParser(BaseParser):
    """Parse documents via pandoc (DOCX, EPUB, HTML, LaTeX, ODT, RST, RTF, …)."""

    supported_extensions: set[FileExtension] = {
        FileExtension.DOC,
        FileExtension.LTX,
        FileExtension.ODT,
        FileExtension.RST,
        FileExtension.RTF,
        FileExtension.TEX,
        FileExtension.TXT,
    }

    #: Maps each supported extension to its pandoc input reader name.
    #: The canonical :class:`FileExtension` enum is the single source of truth
    #: for *which* extensions exist; this class-level map declares which
    #: *pandoc reader* each one needs — owned by the parser, not the enum.
    _READER_FOR: ClassVar[dict[FileExtension, str]] = {
        FileExtension.DOC: "docx",
        FileExtension.DOCX: "docx",
        FileExtension.EPUB: "epub",
        FileExtension.HTML: "html",
        FileExtension.HTM: "html",
        FileExtension.PPTX: "pptx",
        FileExtension.XLSX: "xlsx",
        FileExtension.ODT: "odt",
        FileExtension.RST: "rst",
        FileExtension.RTF: "rtf",
        FileExtension.TEX: "latex",
        FileExtension.LTX: "latex",
        FileExtension.MD: "markdown",
        FileExtension.MARKDOWN: "markdown",
        FileExtension.TXT: "markdown",
        FileExtension.CSV: "csv",
        FileExtension.ADOC: "asciidoc",
    }

    @override
    def parse(self, source_file: Path) -> ParsedDocument:
        _ensure_pandoc_available()
        try:
            ext = FileExtension(source_file.suffix.lower())
        except ValueError:
            # Unknown text extension (source code, arbitrary suffix):
            # parse like .txt via the markdown reader fallback.
            ext = FileExtension.TXT
        fmt = self._READER_FOR.get(ext, "markdown")
        ast_json = pypandoc.convert_file(str(source_file), "json", format=fmt)
        data = json.loads(ast_json)
        blocks: list[dict[str, object]] = data.get("blocks", [])

        elements: list[DocumentElement] = []
        for idx, block in enumerate(blocks):
            el = _block_to_element(block, idx, source_file)
            if el is not None:
                elements.append(el)

        title = _extract_title(blocks) or source_file.stem
        metadata: dict[str, str] = {"parser": "pandoc", "format": fmt}

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=title,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@cache
def _ensure_pandoc_available() -> None:
    """Ensure the pandoc binary exists; download it on demand if missing.

    Cached via ``functools.cache`` so the body runs at most once per process.
    When pandoc is on PATH this is a fast no-op (~0.3s). When it is missing,
    pypandoc downloads the latest release to a platform-default location.
    """
    try:
        pypandoc.get_pandoc_path()
    except OSError:
        logger.info("pandoc binary not found — downloading on demand (one-time setup)")
        try:
            pypandoc.ensure_pandoc_installed()
            logger.info("pandoc installed successfully")
        except Exception as exc:  # noqa: BLE001 — download failure is non-fatal
            logger.warning("could not download pandoc automatically: %s", exc)


# ── Helpers ──────────────────────────────────────────────────────────────
def _block_to_element(
    block: dict[str, object],
    idx: int,
    source_file: Path,
) -> DocumentElement | None:
    if not isinstance(block, dict):
        return None
    tag = block.get("t")
    if not isinstance(tag, str):
        return None
    try:
        pandoc_tag = _PandocBlockTag(tag)
    except ValueError:
        return None

    el_type = pandoc_tag.element_type

    content = _render_block(block)
    if not content:
        return None

    heading_level: int | None = None
    if pandoc_tag is _PandocBlockTag.HEADER:
        heading_level = _parse_header_level(block)

    return DocumentElement(
        type=el_type,
        content=content,
        reading_order=idx,
        heading_level=heading_level,
        source_file=source_file,
    )


def _parse_header_level(block: dict[str, object]) -> int | None:
    """Extract heading level from a Header block's content list."""
    if not isinstance(block, dict):
        return None
    content = block.get("c")
    if isinstance(content, list) and content:
        raw = content[0]
        if isinstance(raw, (int, str)):
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
    return None


def _extract_title(blocks: list[dict[str, object]]) -> str:
    """Return the text of the first level-1 Header block, or empty string."""
    for block in blocks:
        if isinstance(block, dict) and block.get("t") == _PandocBlockTag.HEADER.value:
            content = block.get("c")
            if isinstance(content, list) and content and content[0] == 1:
                return _render_block(block).lstrip("#").strip()
    return ""


def _render_block(block: dict[str, object]) -> str:
    """Render a single JSON AST block to markdown via a JSON round-trip."""
    single = json.dumps(
        {
            "blocks": [block],
            "meta": {},
            "pandoc-api-version": [1, 23],
        }
    )
    text = pypandoc.convert_text(single, "markdown", format="json")
    return text.strip()


def _factory(_options: ParserBuildOptions) -> PandocParser:
    return PandocParser()


register_backend("pandoc", _factory, parser_cls=PandocParser)
