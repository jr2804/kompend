"""Abstract base for kompend document parsers."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from kompend.core.models import ParsedDocument
from kompend.types import JSONValue

# Parser-layer image-resolution defaults. Lives here (the neutral parser base)
# so both individual parsers and the registry's build-options share one source.
DPI_DEFAULT: float = 200.0
IMAGE_SCALE_DEFAULT: float = DPI_DEFAULT / 72.0  # ≈ 200 DPI for image extraction


# PDF text extraction (broken CMaps, mojibake) can emit lone UTF-16 surrogates
# (U+D800–U+DFFF). They are invalid in UTF-8, so any strict encode downstream
# (debug dumps, cache writes, hashing) raises ``UnicodeEncodeError: surrogates
# not allowed``. Scrub them once at the router boundary — every parser output
# passes through here.
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


class FileExtension(StrEnum):
    """Canonical file extensions supported by kompend parsers."""

    PDF = ".pdf"
    DOCX = ".docx"
    DOC = ".doc"
    EPUB = ".epub"
    HTML = ".html"
    HTM = ".htm"
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
    TIFF = ".tiff"
    TIF = ".tif"
    BMP = ".bmp"

    @property
    def is_text(self) -> bool:
        """True when the format is naturally text and benefits from
        :func:`kompend.core.text.encoding.normalize_to_utf8_path` prior to
        parsing. False for binary containers (PDF/DOCX/raster/...) which
        parsers consume byte-wise.
        """
        return self in _TEXT_EXTENSIONS


#: Text-format extensions. Single source of truth for the encoding layer
#: gate (:func:`kompend.core.pipeline.ingest.parse_target_path`): a source
#: in this set is handed to the parser as UTF-8. Everything else is handed
#: over byte-wise — binary containers because they are binary, suffixes the
#: table does not know because the backend that claimed them reads the
#: bytes as they are.
_TEXT_EXTENSIONS: frozenset[FileExtension] = frozenset(
    {
        FileExtension.HTML,
        FileExtension.HTM,
        FileExtension.MD,
        FileExtension.MARKDOWN,
        FileExtension.TXT,
        FileExtension.CSV,
        FileExtension.ADOC,
        FileExtension.RST,
        FileExtension.TEX,
        FileExtension.LTX,
        FileExtension.EML,
    }
)


class BaseParser(ABC):
    """Parser-agnostic interface for kompend Tier 1."""

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
    """Dispatches each file to the one registered backend that claims it.

    Exactly one parser is registered (see :func:`build_default_router`): the
    ingestion backend named by ``[ingest] parser``. There is no ordered chain
    and no hand-off between backends — parsecraft performs its own per-page
    routing internally, so a second kompend-side level of fallback would only
    duplicate it.
    """

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []

    def __len__(self) -> int:
        return len(self._parsers)

    def register(self, parser: BaseParser) -> None:
        self._parsers.append(parser)

    def fingerprint(self) -> str:
        """Stable identity of the configured parser.

        The backend class name plus any per-backend :meth:`cache_identity`
        identify *which* backend — with which routing inputs — produced a
        document. The parse cache and chunk cache key on this so that switching
        the backend (``[ingest] parser``) or a backend's own routing
        configuration (the parsecraft judge tier) invalidates downstream caches
        instead of silently reusing output produced under the old configuration.
        """
        names = [type(parser).__name__ for parser in self._parsers]
        blob = "|".join(names)
        # A backend may carry routing inputs that change its output without
        # changing its class — the parsecraft judge tier is the first such case.
        # Duck-typed on purpose: ``ParsecraftParser`` deliberately is not a
        # ``BaseParser`` subclass, and the router only needs the structural
        # surface (``can_parse`` / ``parse``) plus this optional hook.
        identities = [identity for parser in self._parsers if (identity := _cache_identity(parser))]
        if identities:
            blob += "|" + "|".join(identities)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def route(self, source_file: Path) -> BaseParser | None:
        """Return the backend that claims *source_file*, or ``None``."""
        return next((parser for parser in self._parsers if parser.can_parse(source_file)), None)

    def parse(self, source_file: Path) -> ParsedDocument:
        """Parse *source_file* with the backend that claims it."""
        parser = self.route(source_file)
        if parser is None:
            raise ValueError(f"No parser found for {source_file}")
        return scrub_document(parser.parse(source_file))


def known_extension(suffix: str) -> FileExtension | None:
    """The :class:`FileExtension` for *suffix*, or ``None`` when the table
    does not know it.

    The table names the formats kompend knows about; a backend may know
    more (parsecraft's native-text backend reads source code: ``.c``,
    ``.h``, ``.py``). Resolving a suffix therefore never raises — an
    unknown suffix is a fact about the format, not an error — and it never
    gates parsing: the router decides which backend reads a file.
    """
    try:
        return FileExtension(suffix.lower())
    except ValueError:
        return None


def _cache_identity(parser: object) -> str:
    """Optional per-backend identity hook for :meth:`ParserRouter.fingerprint`.

    Returns the backend's own ``cache_identity()`` when it defines one, else an
    empty string so the fingerprint stays unchanged for every backend that
    carries no routing configuration of its own.
    """
    identity = getattr(parser, "cache_identity", None)
    return str(identity()) if callable(identity) else ""


def scrub_document(doc: ParsedDocument) -> ParsedDocument:
    """Scrub surrogates from all text carried by a parsed document (in place)."""
    doc.title = _scrub_text(doc.title)
    doc.author = _scrub_text(doc.author)
    doc.metadata = _scrub_metadata(doc.metadata)
    for element in doc.elements:
        content = element.content
        if isinstance(content, str):
            element.content = _scrub_text(content)
        elif isinstance(content, list):
            element.content = [[_scrub_text(cell) for cell in row] for row in content]
        element.metadata = _scrub_metadata(element.metadata)
    return doc


def _scrub_metadata(value: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Scrub strings inside a metadata mapping."""
    return {k: _scrub_value(v) for k, v in value.items()}


def _scrub_value(value: JSONValue) -> JSONValue:
    """Recursively scrub strings inside common container types."""
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub_value(v) for k, v in value.items()}
    return value


def _scrub_text(value: str) -> str:
    """Replace lone surrogate code points with U+FFFD."""
    if _SURROGATE_RE.search(value):
        return _SURROGATE_RE.sub("\ufffd", value)
    return value
