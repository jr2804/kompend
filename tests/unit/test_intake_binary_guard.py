"""Intake guard against extension/content lies (knox-rtmn).

The router picks a backend from the extension alone. A PDF renamed ``.md``
reaches the text path, is decoded leniently and parses *successfully* into
U+FFFD soup — an untruthful result the user cannot diagnose. The guard
refuses a source the format table classifies as text whose bytes sniff as
binary, at the two intake points:

- :func:`kompend.core.pipeline.ingest.parse_target_path` (ingest + tier-1
  parse, before charset normalization)
- :meth:`kompend.preprocessing.parsers.base.ParserRouter.parse`
  (``kompend parse``, attachments, MCP knowledge tool)

The guard must stay narrow: unknown suffixes (``.c``) and binary containers
(``.pdf``) pass untouched, non-UTF-8 text (Latin-1, UTF-16) still normalizes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from kompend.core.models import ParsedDocument
from kompend.core.pipeline.ingest import ingest_documents, parse_target_path
from kompend.preprocessing.parsers.base import (
    BaseParser,
    BinarySourceError,
    ParserRouter,
    assert_text_source,
)

#: A real PDF from the test corpus — synthetic "%PDF" headers are not enough:
#: binaryornot sniffs control-char ratios, and the guard must hold on real
#: container bytes.
_REAL_PDF = Path(__file__).parent.parent / "data" / "batch2" / "T-REC-P.501-202504-I!!PDF-E.pdf"


# ── ParserRouter.parse (kompend parse / attachments / MCP knowledge) ────────


class _CountingParser(BaseParser):
    """Claims everything; records whether it was reached."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def can_parse(self, source_file: Path) -> bool:
        return True

    def parse(self, source_file: Path) -> ParsedDocument:
        self.calls.append(source_file)
        return ParsedDocument(source_file=source_file, elements=[], title="")


# ── assert_text_source ────────────────────────────────────────────────────────


def test_refuses_binary_bytes_behind_a_text_extension(tmp_path: Path) -> None:
    """The PDF-named-.md lie is refused with a truthful message."""
    liar = _mislabelled_pdf(tmp_path)

    with pytest.raises(BinarySourceError, match="claims text \\(.md\\).*binary"):
        assert_text_source(liar)


def test_binary_source_error_is_a_valueerror() -> None:
    """Established ``except ValueError`` sites keep working."""
    assert issubclass(BinarySourceError, ValueError)


def test_accepts_real_text(tmp_path: Path) -> None:
    src = tmp_path / "note.md"
    src.write_text("# Note\n\nBody text.\n", encoding="utf-8")

    assert_text_source(src)  # must not raise


def test_accepts_utf16_text_behind_a_text_extension(tmp_path: Path) -> None:
    """UTF-16 carries NUL bytes; the sniff must not false-positive on it.

    The encoding layer (charset-normalizer) decodes UTF-16, so a legit
    UTF-16 ``.md`` still reaches the UTF-8 sibling path.
    """
    src = tmp_path / "utf16.md"
    src.write_text("# Heading\n\nCafé content.\n", encoding="utf-16")

    assert_text_source(src)  # must not raise


def test_accepts_non_utf8_text_behind_a_text_extension(tmp_path: Path) -> None:
    src = tmp_path / "latin.md"
    src.write_bytes("# café\n".encode("latin-1"))

    assert_text_source(src)  # must not raise


def test_passes_binary_container_byte_wise(tmp_path: Path) -> None:
    """Honest binary sources are consumed as bytes, never gated."""
    src = tmp_path / "doc.pdf"
    shutil.copyfile(_REAL_PDF, src)

    assert_text_source(src)  # must not raise


def test_passes_unknown_suffix_byte_wise(tmp_path: Path) -> None:
    """Backends read more than the table lists — the backend decides."""
    src = tmp_path / "blob.c"
    src.write_bytes(b"\x00\x01\x02\x03not really source")

    assert_text_source(src)  # must not raise


def test_router_parse_refuses_the_mislabel(tmp_path: Path) -> None:
    router, parser = _router()
    liar = _mislabelled_pdf(tmp_path)

    with pytest.raises(BinarySourceError):
        router.parse(liar)

    assert parser.calls == [], "the guard must fire before any backend sees the file"


def test_router_parse_still_parses_honest_sources(tmp_path: Path) -> None:
    router, parser = _router()
    src = tmp_path / "note.md"
    src.write_text("# Note\n", encoding="utf-8")

    doc = router.parse(src)

    assert parser.calls == [src]
    assert doc.source_file == src


def test_router_parse_keeps_binary_container_byte_wise(tmp_path: Path) -> None:
    router, parser = _router()
    src = tmp_path / "doc.pdf"
    shutil.copyfile(_REAL_PDF, src)

    router.parse(src)

    assert parser.calls == [src]


def _router() -> tuple[ParserRouter, _CountingParser]:
    parser = _CountingParser()
    router = ParserRouter()
    router.register(parser)
    return router, parser


# ── parse_target_path (ingest + tier-1 parse) ────────────────────────────────


def test_parse_target_path_refuses_the_mislabel(tmp_path: Path) -> None:
    liar = _mislabelled_pdf(tmp_path)

    with pytest.raises(BinarySourceError):
        parse_target_path(liar, tmp_path / "cache")


def test_parse_target_path_still_normalizes_honest_text(tmp_path: Path) -> None:
    src = tmp_path / "latin.md"
    src.write_bytes("# café\n".encode("latin-1"))

    target = parse_target_path(src, tmp_path / "cache")

    assert target != src
    assert target.suffix == ".md"


def test_parse_target_path_keeps_binary_container_byte_wise(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    shutil.copyfile(_REAL_PDF, src)

    assert parse_target_path(src, tmp_path / "cache") == src


def test_parse_target_path_keeps_unknown_suffix_byte_wise(tmp_path: Path) -> None:
    src = tmp_path / "code.c"
    src.write_text("int main(void) { return 0; }\n", encoding="utf-8")

    assert parse_target_path(src, tmp_path / "cache") == src


# ── End-to-end: a mislabelled source in a vault fails truthfully ────────────


def test_ingest_reports_the_mislabel_as_failed(tmp_path: Path) -> None:
    """``ingest_documents`` counts the lie as failed with a truthful reason."""
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir()
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    src = _mislabelled_pdf(base / "sources")

    with patch("kompend.core.pipeline.ingest.normalize_to_utf8_path") as mock_norm:
        result = ingest_documents(
            files=[src],
            base_dir=base,
            router=cast("ParserRouter", MagicMock()),
            reparse=True,
            config=None,
        )

    assert result.failed == 1
    assert result.count == 0
    mock_norm.assert_not_called(), "the guard fires before charset normalization"


def _mislabelled_pdf(tmp_path: Path, name: str = "liar.md") -> Path:
    target = tmp_path / name
    shutil.copyfile(_REAL_PDF, target)
    return target
