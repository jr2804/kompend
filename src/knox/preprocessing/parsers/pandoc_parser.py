"""Pandoc-based parser using the ``pandoc`` Python library."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, override

import pandoc as _pandoc_module
from pandoc.types import (
    Block,
    BulletList,
    CodeBlock,
    Header,
    LineBlock,
    Math,
    OrderedList,
    Pandoc,
    Para,
    Plain,
    Table,
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
        FileExtension.DOCX,
        FileExtension.DOC,
        FileExtension.EPUB,
        FileExtension.HTML,
        FileExtension.PPTX,
        FileExtension.ODT,
        FileExtension.RST,
        FileExtension.RTF,
        FileExtension.TEX,
        FileExtension.MD,
    }

    @override
    def parse(self, source_path: Path) -> ParsedDocument:
        fmt = _pandoc_module.format_from_filename(str(source_path))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Pandoc version.*not supported")
            doc = _pandoc_module.read(file=str(source_path), format=fmt)

        blocks: list[Block] = list(doc[1])

        elements: list[DocumentElement] = []
        for idx, block in enumerate(blocks):
            el = _block_to_element(block, idx, source_path)
            if el is not None:
                elements.append(el)

        title = _extract_title(blocks) or source_path.stem
        metadata: dict[str, Any] = {"parser": "pandoc", "format": fmt}

        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=title,
            metadata=metadata,
        )


def _block_to_element(
    block: Block,
    idx: int,
    source_path: Path,
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
        source_file=source_path,
    )


def _render_block(block: Block) -> str:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pandoc version.*not supported")
        text = _pandoc_module.write(Pandoc(_pandoc_module.types.Meta({}), [block]))
    return text.strip()


def _extract_title(blocks: list[Block]) -> str:
    for block in blocks:
        if isinstance(block, Header) and int(block[0]) == 1:
            return _render_block(block).lstrip("#").strip()
    return ""
