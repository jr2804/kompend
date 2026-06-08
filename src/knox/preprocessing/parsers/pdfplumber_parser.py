"""pdfplumber-based parser for plain PDF text extraction."""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser, FileExtension


class PdfPlumberParser(BaseParser):
    """Lightweight PDF parser using pdfplumber."""

    supported_extensions = {FileExtension.PDF}

    def parse(self, source_path: Path) -> ParsedDocument:  # noqa: PLR6301
        elements: list[DocumentElement] = []
        with pdfplumber.open(source_path) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    elements.append(
                        DocumentElement(
                            type=ElementType.TEXT,
                            content=text,
                            page=page_num,
                        )
                    )
        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=source_path.stem,
            page_count=page_count,
            metadata={"parser": "pdfplumber"},
        )
