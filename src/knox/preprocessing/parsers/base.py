"""Abstract base for KNOX document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from knox.core.models import ParsedDocument


class FileExtension(StrEnum):
    """Canonical file extensions supported by KNOX parsers."""

    PDF = ".pdf"
    DOCX = ".docx"
    DOC = ".doc"
    EPUB = ".epub"
    HTML = ".html"
    PPTX = ".pptx"
    XLSX = ".xlsx"
    ODT = ".odt"
    RST = ".rst"
    RTF = ".rtf"
    TEX = ".tex"
    MD = ".md"
    TXT = ".txt"
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    GIF = ".gif"
    WEBP = ".webp"


class BaseParser(ABC):
    """Parser-agnostic interface for KNOX Tier 1."""

    supported_extensions: set[FileExtension] = set()

    @abstractmethod
    def parse(self, source_path: Path) -> ParsedDocument:
        """Parse a single source file into a structured document."""
        ...

    def can_parse(self, source_path: Path) -> bool:
        """Check if this parser supports the given file."""
        return source_path.suffix.lower() in self.supported_extensions


class ParserRouter:
    """Auto-selects the right parser for each file type."""

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []

    def register(self, parser: BaseParser) -> None:
        self._parsers.append(parser)

    def route(self, source_path: Path) -> BaseParser | None:
        for parser in self._parsers:
            if parser.can_parse(source_path):
                return parser
        return None

    def parse(self, source_path: Path) -> ParsedDocument:
        parser = self.route(source_path)
        if not parser:
            raise ValueError(f"No parser found for {source_path}")
        return parser.parse(source_path)
