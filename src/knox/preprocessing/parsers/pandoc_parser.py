"""Pandoc-based parser using pypandoc."""

from __future__ import annotations

import json
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, override

import pypandoc

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import BaseParser, FileExtension

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Format overrides
# ---------------------------------------------------------------------------

# Only formats pandoc cannot infer from the file extension alone.
_EXTENSION_OVERRIDES: dict[FileExtension, str] = {
    FileExtension.DOC: "docx",
    FileExtension.TXT: "markdown",
}

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


# ---------------------------------------------------------------------------
# Block type mapping (pandoc JSON tag → KNOX element type)
# ---------------------------------------------------------------------------

_BLOCK_TYPE_MAP: dict[_PandocBlockTag, ElementType] = {
    _PandocBlockTag.HEADER: ElementType.HEADING,
    _PandocBlockTag.CODE_BLOCK: ElementType.CODE,
    _PandocBlockTag.TABLE: ElementType.TABLE,
    _PandocBlockTag.MATH: ElementType.FORMULA,
    _PandocBlockTag.BULLET_LIST: ElementType.LIST_ITEM,
    _PandocBlockTag.ORDERED_LIST: ElementType.LIST_ITEM,
    _PandocBlockTag.PARA: ElementType.TEXT,
    _PandocBlockTag.PLAIN: ElementType.TEXT,
    _PandocBlockTag.LINE_BLOCK: ElementType.TEXT,
}


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

    @override
    def parse(self, source_file: Path) -> ParsedDocument:
        _ensure_pandoc_available()
        fmt = _resolve_format(source_file)
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


def _resolve_format(source_file: Path) -> str:
    """Return a pandoc format override, or ``""`` to infer from extension."""
    try:
        ext = FileExtension(source_file.suffix.lower())
    except ValueError:
        return ""
    return _EXTENSION_OVERRIDES.get(ext, "")


def _block_to_element(
    block: Any,
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

    el_type = _BLOCK_TYPE_MAP.get(pandoc_tag)
    if el_type is None:
        return None

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


def _parse_header_level(block: Any) -> int | None:
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


def _render_block(block: Any) -> str:
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
