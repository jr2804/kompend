"""pdfplumber-based parser for plain PDF text extraction."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser


try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class PdfPlumberParser(BaseParser):
    """Lightweight PDF parser using pdfplumber."""

    supported_extensions = (".pdf",)

    def __init__(self) -> None:
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber is required. Install with: uv pip install pdfplumber")

    def parse(self, source_path: Path) -> ParsedDocument:
        elements: list[DocumentElement] = []
        with pdfplumber.open(source_path) as pdf:
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
            page_count=len(pdf.pages),
            metadata={"parser": "pdfplumber"},
        )
