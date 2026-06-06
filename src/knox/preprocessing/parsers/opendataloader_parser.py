"""opendataloader-pdf adapter for PDF parsing.

This is an alternative to Docling for PDF, using opendataloader's PDFLoader.
opendataloader-pdf is a heavy dependency (includes PyTorch), so this adapter
is a thin wrapper that provides the same BaseParser interface.

TODO: When opendataloader-pdf is installed, complete the conversion from
opendataloader's output format to ParsedDocument. Use docling_parser.py as
a template for the element extraction logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser

logger = logging.getLogger(__name__)

try:
    from opendataloader import PDFLoader

    OPD_AVAILABLE = True
except ImportError:
    OPD_AVAILABLE = False


class OpenDataLoaderParser(BaseParser):
    """PDF parser using opendataloader-pdf as the extraction backend.

    Requires the ``opendataloader-pdf`` package which bundles PyTorch.
    Falls back to a clear ImportError with install instructions when
    the dependency is missing.
    """

    supported_extensions = (".pdf",)

    def __init__(self) -> None:
        if not OPD_AVAILABLE:
            raise ImportError(
                "opendataloader-pdf is required for OpenDataLoaderParser. "
                "Install with: uv pip install opendataloader-pdf"
            )
        self._loader = PDFLoader()

    def parse(self, source_path: Path) -> ParsedDocument:
        """Parse a PDF using opendataloader and map to ParsedDocument.

        TODO: Full implementation. Currently a thin adapter skeleton.
        Reference ``docling_parser.py`` for the element-extraction pattern:
        iterate over pages, extract text/table/image elements, and construct
        ``DocumentElement`` instances with appropriate ``ElementType``.
        """
        # opendataloader extraction call — replace with actual API once
        # opendataloader-pdf is available and its output format is known.
        # result = self._loader.load(str(source_path))
        elements: list[DocumentElement] = []

        # Stub: return metadata-only result during adapter bring-up.
        # Replace with real page iteration and element extraction.
        logger.info("OpenDataLoaderParser.parse(%s) — stub implementation", source_path)

        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=source_path.stem,
            page_count=0,
            metadata={"parser": "opendataloader"},
        )