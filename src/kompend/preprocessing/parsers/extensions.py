"""The suffix set a vault can ingest.

Derived from the one parser rather than declared: the parse backend's own
capability table (parsecraft's ``MEDIA_TYPES``) is the authority on what it reads,
and a hand-written copy next to it would drift the moment the backend learns a
format. Only the suffix table is touched here — building it never instantiates the
backend, so this stays cheap enough to import at module scope.
"""

from __future__ import annotations

from kompend.preprocessing.parsers.base import FileExtension
from kompend.preprocessing.parsers.document_parser import build_parser


def supported_extensions() -> set[str]:
    """Every suffix this vault can ingest, as lowercase strings with the dot.

    The union of kompend's own format table and whatever the configured parser
    claims. Discovery, the status report and the parser must agree on this set —
    hence one function, not a flag on each side.
    """
    extensions = {str(extension) for extension in FileExtension}
    extensions.update(str(extension) for extension in _parser_suffixes())
    return extensions


def _parser_suffixes() -> set[str]:
    """The suffixes the configured parser claims, without paying for a backend."""
    parser = build_parser().parser
    return {str(extension) for extension in parser.supported_extensions}
