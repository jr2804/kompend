"""opendataloader-pdf adapter for PDF parsing.

Requires ``opendataloader-pdf`` to be installed.  The package is *not*
a hard dependency of knox — it must be installed separately::

    pip install opendataloader-pdf

This parser is **not** wired into the default pipeline (which uses
:class:`DoclingParser`).  It is registered via
:class:`~knox.preprocessing.parsers.base.ParserRouter` and can be
selected explicitly when the caller knows ``opendataloader-pdf`` is
available.

The ``opendataloader_pdf`` package is imported at :meth:`parse` time.
If it is not installed, :class:`ImportError` is raised on the first
call — the module itself is always importable.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser, FileExtension

logger = logging.getLogger(__name__)

# Mapping from OpenDataLoader JSON element types -> KNOX ElementType.
_OD_TYPE_MAP: dict[str, ElementType] = {
    "heading": ElementType.HEADING,
    "paragraph": ElementType.TEXT,
    "table": ElementType.TABLE,
    "list": ElementType.LIST_ITEM,
    "image": ElementType.IMAGE,
    "caption": ElementType.TEXT,
    "formula": ElementType.FORMULA,
}


class OpenDataLoaderParser(BaseParser):
    """PDF parser using ``opendataloader-pdf`` as the extraction backend.

    ``opendataloader_pdf`` is imported on first use so that this module
    can always be imported even when the package is not installed.
    """

    supported_extensions = {FileExtension.PDF}

    def __init__(self) -> None:
        import opendataloader_pdf  # noqa: PLC0415 – deliberate lazy import for optional package

        self._mod = opendataloader_pdf

    def parse(self, source_path: Path) -> ParsedDocument:
        """Parse a PDF using opendataloader and map to ParsedDocument."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._mod.convert(
                input_path=[str(source_path)],
                output_dir=tmp_dir,
                format="json",
            )

            json_files = list(Path(tmp_dir).glob("*.json"))
            if not json_files:
                msg = f"OpenDataLoader produced no JSON output for {source_path}"
                raise RuntimeError(msg)

            result = json.loads(json_files[0].read_text(encoding="utf-8"))

        return self._build_parsed_document(result, source_path)

    @staticmethod
    def _build_parsed_document(result: dict[str, Any], source_path: Path) -> ParsedDocument:
        """Convert OpenDataLoader JSON result into :class:`ParsedDocument`."""
        elements: list[DocumentElement] = []
        title: str = source_path.stem
        page_count = 0
        pages_seen: set[int] = set()

        for item in result.get("elements", []):
            od_type = item.get("type", "paragraph")
            el_type = _OD_TYPE_MAP.get(od_type, ElementType.TEXT)
            content = item.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            page_number = item.get("page number")
            if isinstance(page_number, int):
                pages_seen.add(page_number)

            metadata: dict[str, Any] = {
                "od_type": od_type,
                "od_id": item.get("id"),
            }

            # Heading level
            heading_level = item.get("heading level")
            if isinstance(heading_level, int) and heading_level >= 1:
                metadata["heading_level"] = heading_level

            # Bounding box
            bbox = item.get("bounding box")
            if isinstance(bbox, list) and len(bbox) == 4:
                metadata["bbox"] = bbox

            # Font info (optional)
            font = item.get("font")
            if isinstance(font, str):
                metadata["font"] = font

            # List type hint
            if el_type == ElementType.LIST_ITEM and item.get("list_type"):
                metadata["list_type"] = item.get("list_type")

            el = DocumentElement(
                type=el_type,
                content=content,
                metadata=metadata,
                source_file=source_path,
                page=page_number,
            )
            elements.append(el)

            # Use first heading as document title
            if el_type == ElementType.HEADING and page_number in (1, None) and title == source_path.stem:
                title = content.strip()

        page_count = max(pages_seen) if pages_seen else 0

        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=title,
            page_count=page_count,
            metadata={
                "parser": "opendataloader",
                "element_count": len(elements),
            },
        )
