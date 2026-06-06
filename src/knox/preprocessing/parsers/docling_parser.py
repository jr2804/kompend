"""Docling-based PDF/DOCX parser with structure-preserving extraction."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser

logger = logging.getLogger(__name__)

try:
    from docling.datamodel.base_models import DocumentStream
    from docling.datamodel.document import ConversionResult
    from docling.document_converter import DocumentConverter
    from docling.datamodel.document import InputFormat

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


class DoclingParser(BaseParser):
    """Parser using Docling for structure-preserving extraction from PDF/DOCX."""

    supported_extensions = (".pdf", ".docx")

    def __init__(
        self,
        image_mode: str = "file",
        extract_images: bool = True,
        table_format: str = "structured",
    ) -> None:
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "docling is required for DoclingParser. "
                'Install with: uv pip install "knox[docling]"'
            )
        self.image_mode = image_mode
        self.extract_images = extract_images
        self.table_format = table_format
        self._converter = DocumentConverter()

    def parse(self, source_path: Path) -> ParsedDocument:
        result = self._converter.convert(source_path)
        doc = result.document

        elements: list[DocumentElement] = []
        image_dir = self._image_dir(source_path)

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

    def _image_dir(self, source_path: Path) -> Path | None:
        if self.extract_images and self.image_mode == "file":
            d = source_path.parent / f"{source_path.stem}_images"
            d.mkdir(exist_ok=True)
            return d
        return None
