"""Format vocabulary and the intake trust boundary for kompend's one parser.

kompend ships a single ingestion backend: parsecraft, which routes every page
across its own sub-backends (docling, liteparse, pandoc, native text, OCR). There
is no kompend-side backend choice to make, so there is no registry here — only
the pieces every parse path shares:

* :class:`FileExtension` — the format table, and :func:`known_extension` to look
  a suffix up without raising;
* :func:`assert_text_source` — the trust boundary that refuses a text-classified
  file whose bytes are binary (a PDF named ``.md``);
* :func:`scrub_document` — the surrogate scrub every parser output passes
  through before it reaches a cache, a log or a hash;
* :class:`DocumentParser` — the object the pipeline holds: it decides whether the
  configured parser claims a file and applies the two guards above on the way in.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from binaryornot.check import is_binary

from kompend.core.models import ParsedDocument
from kompend.types import JSONValue

# Parser-layer image-resolution defaults. Live here (the shared parser surface)
# so both the parser and the image store read one value.
DPI_DEFAULT: float = 200.0
IMAGE_SCALE_DEFAULT: float = DPI_DEFAULT / 72.0  # ≈ 200 DPI for image extraction


# PDF text extraction (broken CMaps, mojibake) can emit lone UTF-16 surrogates
# (U+D800–U+DFFF). They are invalid in UTF-8, so any strict encode downstream
# (debug dumps, cache writes, hashing) raises ``UnicodeEncodeError: surrogates
# not allowed``. Scrub them once at the parse boundary — every parser output
# passes through here.
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


class FileExtension(StrEnum):
    """Canonical file extensions supported by kompend."""

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
    TSV = ".tsv"
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
#: table does not know because the parser that claimed them reads the
#: bytes as they are.
_TEXT_EXTENSIONS: frozenset[FileExtension] = frozenset(
    {
        FileExtension.HTML,
        FileExtension.HTM,
        FileExtension.MD,
        FileExtension.MARKDOWN,
        FileExtension.TXT,
        FileExtension.CSV,
        FileExtension.TSV,
        FileExtension.ADOC,
        FileExtension.RST,
        FileExtension.TEX,
        FileExtension.LTX,
        FileExtension.EML,
    }
)


@runtime_checkable
class ParserBackend(Protocol):
    """The structural surface :class:`DocumentParser` needs from a backend.

    Structural rather than an ABC base class: the one backend is
    :class:`~kompend.preprocessing.parsers.parsecraft.ParsecraftParser`,
    which carries parsecraft-specific constructor state the shared surface has
    no business knowing about. A future backend only has to satisfy this.
    """

    #: True when the parser already produces figure descriptions itself.
    figure_enrichment: bool

    #: Every suffix this backend claims, lowercase with the dot.
    supported_extensions: set[str]

    def can_parse(self, source_file: Path) -> bool:
        """True when this backend claims *source_file*."""
        ...

    def parse(self, source_file: Path) -> ParsedDocument:
        """Parse *source_file* into a structured document."""
        ...


class DocumentParser:
    """Holds the configured parser and applies the guards around it.

    One backend, no chain: parsecraft performs its own per-page routing, so a
    second kompend-side level of fallback would only duplicate it. This object
    exists for the two things that must happen on *every* parse regardless of the
    backend — refusing a mislabelled text file before it becomes U+FFFD soup, and
    scrubbing lone surrogates out of the result before it reaches a cache or a
    hash.
    """

    def __init__(self, parser: ParserBackend) -> None:
        self._parser = parser

    def __repr__(self) -> str:
        return f"DocumentParser({type(self._parser).__name__})"

    @property
    def parser(self) -> ParserBackend:
        """The wrapped backend."""
        return self._parser

    def backend_identity(self) -> str:
        """Identity of the parse this parser would produce.

        Two parts, both from the backend itself:

        * ``backend_identity()`` names the backends that can read here, so a
          changed backend set re-parses while a version bump does not;
        * ``cache_identity()`` names the routing inputs that change a backend's
          output without changing its class — the judge tier, the ranking
          preference, per-backend options. Those decide which backend leads each
          page, so a cached document produced under a different judge is a
          different document.

        The parse cache keys on this. A backend that defines neither hook
        returns the empty string, which leaves an existing cache key unchanged.
        """
        parts = [part for hook in ("backend_identity", "cache_identity") if (part := self._identity_of(hook))]
        return "|".join(parts)

    def route(self, source_file: Path) -> ParserBackend | None:
        """Return the parser when it claims *source_file*, else ``None``."""
        return self._parser if self._parser.can_parse(source_file) else None

    def parse(self, source_file: Path) -> ParsedDocument:
        """Parse *source_file*, guarded at the boundaries."""
        # A source the table classifies as text whose bytes sniff as binary is a
        # mislabelled file (a PDF named ``.md``): refuse it here, on the path the
        # parser is about to read, rather than let lenient decoding turn it into
        # a successful garbage parse (knox-rtmn).
        assert_text_source(source_file)
        return scrub_document(self._parser.parse(source_file))

    def _identity_of(self, hook: str) -> str:
        """Call an optional identity hook on the wrapped parser; ``""`` when absent."""
        identity = getattr(self._parser, hook, None)
        return str(identity()) if callable(identity) else ""


class BinarySourceError(ValueError):
    """A source the format table classifies as text carries binary bytes.

    Raised where intake can still refuse the file honestly
    (:meth:`DocumentParser.parse`,
    :func:`kompend.core.pipeline.ingest.parse_target_path`). A :class:`ValueError`
    subclass so the established ``except ValueError`` sites keep working, while
    callers that care about the mislabel case can catch it precisely.
    """


def assert_text_source(source_file: Path) -> None:
    """Refuse a text-classified source whose bytes are binary.

    The parser picks a format from the extension alone, so a PDF named ``.md``
    reaches the text path: bytes are decoded leniently into a U+FFFD soup and the
    parse *succeeds* with garbage content (were ``except ValueError`` not in the
    way, it would crash any pipeline). This guard closes the lie: a suffix the
    format table knows as text (:data:`FileExtension.is_text`) whose first bytes
    sniff as binary is refused before a parser can produce an untruthful result.

    Deliberately narrow:

    - Unknown suffixes pass — the backend reads more than the table lists
      (``.c``, ``.py``; see :func:`known_extension`), and the backend
      decides what its own extensions mean.
    - Binary containers pass — their parsers consume bytes as bytes.
    - Text is judged by :mod:`binaryornot`'s sniff (NUL bytes and
      control-char ratios over the file head), the documented heuristic
      for this check (issue knox-rtmn). It accepts UTF-16 text, so the
      encoding layer keeps its charset-normalizer fallback.
    """
    extension = known_extension(source_file.suffix)
    if extension is None or not extension.is_text:
        return
    if is_binary(str(source_file)):
        raise BinarySourceError(f"{source_file}: extension claims text ({extension.value}) but the bytes are binary — file mislabelled?")


def known_extension(suffix: str) -> FileExtension | None:
    """The :class:`FileExtension` for *suffix*, or ``None`` when the table
    does not know it.

    The table names the formats kompend knows about; the backend may know more
    (parsecraft's native-text backend reads source code: ``.c``, ``.h``,
    ``.py``). Resolving a suffix therefore never raises — an unknown suffix is a
    fact about the format, not an error — and it never gates parsing: the
    backend decides which formats it claims.
    """
    try:
        return FileExtension(suffix.lower())
    except ValueError:
        return None


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
