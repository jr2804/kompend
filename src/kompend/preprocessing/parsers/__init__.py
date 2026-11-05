"""kompend's document parser.

One backend: parsecraft, which routes each page across its own sub-backends
(docling, liteparse, pandoc, native text, OCR). There is no kompend-side backend
choice, so there is no registry — :func:`build_parser` builds the one parser and
:class:`DocumentParser` wraps it with the intake and scrub guards.

The parsecraft adapter is imported lazily inside :func:`build_parser`: it pulls the
parsecraft pipeline, and ``--help`` or a suffix-table query must not pay for it.
"""

from __future__ import annotations

from kompend.preprocessing.parsers.base import (
    BinarySourceError as BinarySourceError,
)
from kompend.preprocessing.parsers.base import (
    DocumentParser as DocumentParser,
)
from kompend.preprocessing.parsers.base import (
    FileExtension as FileExtension,
)
from kompend.preprocessing.parsers.base import (
    ParserBackend as ParserBackend,
)
from kompend.preprocessing.parsers.base import (
    assert_text_source as assert_text_source,
)
from kompend.preprocessing.parsers.base import (
    known_extension as known_extension,
)
from kompend.preprocessing.parsers.base import (
    scrub_document as scrub_document,
)
from kompend.preprocessing.parsers.document_parser import (
    build_parser as build_parser,
)
from kompend.preprocessing.parsers.options import (
    PARSER_PARSECRAFT as PARSER_PARSECRAFT,
)
from kompend.preprocessing.parsers.options import (
    ParserOptions as ParserOptions,
)

__all__ = [
    "PARSER_PARSECRAFT",
    "BinarySourceError",
    "DocumentParser",
    "FileExtension",
    "ParserBackend",
    "ParserOptions",
    "assert_text_source",
    "build_parser",
    "known_extension",
    "scrub_document",
]
