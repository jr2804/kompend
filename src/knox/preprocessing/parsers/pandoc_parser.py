"""Pandoc-based parser using the ``pandoc`` Python library."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, override

import pandoc as _pandoc_module
from pandoc.types import (  # ty: ignore
    Block,  # ty:ignore[unresolved-import]
    BulletList,  # ty:ignore[unresolved-import]
    CodeBlock,  # ty:ignore[unresolved-import]
    Header,  # ty:ignore[unresolved-import]
    LineBlock,  # ty:ignore[unresolved-import]
    Math,  # ty:ignore[unresolved-import]
    OrderedList,  # ty:ignore[unresolved-import]
    Pandoc,  # ty:ignore[unresolved-import]
    Para,  # ty:ignore[unresolved-import]
    Plain,  # ty:ignore[unresolved-import]
    Table,  # ty:ignore[unresolved-import]
)

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser, FileExtension

logger = logging.getLogger(__name__)

_BLOCK_TYPE_MAP: dict[type, ElementType] = {
    Header: ElementType.HEADING,
    CodeBlock: ElementType.CODE,
    Table: ElementType.TABLE,
    Math: ElementType.FORMULA,
    BulletList: ElementType.LIST_ITEM,
    OrderedList: ElementType.LIST_ITEM,
    Para: ElementType.TEXT,
    Plain: ElementType.TEXT,
    LineBlock: ElementType.TEXT,
}


class PandocParser(BaseParser):
    """Parser using the pandoc Python library for universal document conversion.

    Supports DOCX, EPUB, HTML, PPTX, ODT, RST, RTF, LaTeX, Markdown and
    any other format that the pandoc binary can read.
    """

    supported_extensions: set[FileExtension] = {
        # Legacy document formats not handled by DoclingParser.
        FileExtension.DOC,
        FileExtension.ODT,
        FileExtension.RST,
        FileExtension.RTF,
        # Plain text (also used as binaryornot fallback).
        FileExtension.TXT,
    }

    @override
    def parse(self, source_file: Path) -> ParsedDocument:
        fmt = _pandoc_module.format_from_filename(str(source_file))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Pandoc version.*not supported")
            doc = _pandoc_module.read(file=str(source_file), format=fmt)

        blocks: list[Block] = list(doc[1])

        elements: list[DocumentElement] = []
        for idx, block in enumerate(blocks):
            el = _block_to_element(block, idx, source_file)
            if el is not None:
                elements.append(el)

        title = _extract_title(blocks) or source_file.stem
        metadata: dict[str, Any] = {"parser": "pandoc", "format": fmt}

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=title,
            metadata=metadata,
        )


def _block_to_element(
    block: Block,
    idx: int,
    source_file: Path,
) -> DocumentElement | None:
    el_type = _BLOCK_TYPE_MAP.get(type(block))
    if el_type is None:
        return None

    content = _render_block(block)
    if not content:
        return None

    heading_level: int | None = None
    if isinstance(block, Header):
        heading_level = int(block[0])

    return DocumentElement(
        type=el_type,
        content=content,
        reading_order=idx,
        heading_level=heading_level,
        source_file=source_file,
    )


def _render_block(block: Block) -> str:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pandoc version.*not supported")
        text = _pandoc_module.write(Pandoc(_pandoc_module.types.Meta({}), [block]))  # ty: ignore
    return text.strip()


def _extract_title(blocks: list[Block]) -> str:
    for block in blocks:
        if isinstance(block, Header) and int(block[0]) == 1:
            return _render_block(block).lstrip("#").strip()
    return ""
