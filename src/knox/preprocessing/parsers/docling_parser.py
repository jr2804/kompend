"""Docling-based PDF/DOCX parser with structure-preserving extraction.

Docling is a hard dependency. However, docling's own module tree tries to
import torch at import time (granite_vision, chart_extraction) even without
the models-local extra. To work around this, we use a local import inside
_get_converter() — the last-resort approach per python-ultimate standards
for cases where TYPE_CHECKING isn't applicable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Parser using Docling for structure-preserving extraction from PDF/DOCX."""

    supported_extensions = (".pdf", ".docx")

    def __init__(
        self,
        image_mode: str = "file",
        extract_images: bool = True,
        table_format: str = "structured",
    ) -> None:
        self.image_mode = image_mode
        self.extract_images = extract_images
        self.table_format = table_format
        self._converter = None

    def _get_converter(self):
        """Lazy-import docling to avoid torch pull at module level."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter  # noqa: PLC0415
            self._converter = DocumentConverter()
        return self._converter

    def parse(self, source_path: Path) -> ParsedDocument:
        converter = self._get_converter()
        result = converter.convert(source_path)
        doc = result.document

        elements: list[DocumentElement] = []
        if doc.pages:
            for page in doc.pages:
                for item in page.export_to_markdown().split("\n"):
                    if item.strip():
                        elements.append(
                            DocumentElement(
                                type=ElementType.TEXT,
                                content=item.strip(),
                            )
                        )

        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=doc.name or source_path.stem,
            page_count=len(doc.pages) if doc.pages else 0,
            metadata={"parser": "docling"},
        )