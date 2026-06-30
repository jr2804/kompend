"""Abstract base for KNOX document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from binaryornot.check import is_binary

from knox.core.models import ParsedDocument

# Parser-layer image-resolution defaults. Lives here (the neutral parser base)
# so both individual parsers and the registry's build-options share one source.
DPI_DEFAULT: float = 200.0
IMAGE_SCALE_DEFAULT: float = DPI_DEFAULT / 72.0  # ≈ 200 DPI for image extraction


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
    LTX = ".ltx"
    MD = ".md"
    MARKDOWN = ".markdown"
    TXT = ".txt"
    CSV = ".csv"
    ADOC = ".adoc"
    EML = ".eml"
    MSG = ".msg"
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    GIF = ".gif"
    WEBP = ".webp"


class BaseParser(ABC):
    """Parser-agnostic interface for KNOX Tier 1."""

    supported_extensions: set[FileExtension] = set()

    # Informational capability flag: True when the parser already produces
    # figure descriptions (e.g. via an internal LLM). When True, figures it
    # captions carry a ``description`` and VLMEnricher skips them per-figure.
    # It is NOT a hard VLM skip — VLM still fills figures the parser missed.
    figure_enrichment: bool = False

    @abstractmethod
    def parse(self, source_file: Path) -> ParsedDocument:
        """Parse a single source file into a structured document."""
        ...

    def can_parse(self, source_file: Path) -> bool:
        """Check if this parser supports the given file."""
        return source_file.suffix.lower() in self.supported_extensions


class ParserRouter:
    """Auto-selects the right parser for each file type.

    When no registered parser matches a file's extension, a last-resort
    check using :func:`binaryornot.check.is_binary` determines whether
    the file is plain text.  If it is, the *text_fallback* parser
    (typically :class:`PandocParser`) is used.
    """

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []
        self._text_fallback: BaseParser | None = None

    def register(self, parser: BaseParser) -> None:
        self._parsers.append(parser)

    def set_text_fallback(self, parser: BaseParser) -> None:
        """Set a fallback parser for text files detected by binaryornot."""
        self._text_fallback = parser

    def route(self, source_file: Path) -> BaseParser | None:
        for parser in self._parsers:
            if parser.can_parse(source_file):
                return parser
        # No registered parser matched — try binaryornot text fallback.
        if self._text_fallback is not None and not is_binary(str(source_file)):
            return self._text_fallback
        return None

    def parse(self, source_file: Path) -> ParsedDocument:
        parser = self.route(source_file)
        if not parser:
            raise ValueError(f"No parser found for {source_file}")
        return parser.parse(source_file)
