"""Verify FileExtension.is_text classification matches the encoding-layer expectation."""

from __future__ import annotations

from knox.preprocessing.parsers.base import FileExtension

_TEXT = {
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
_BINARY = set(FileExtension) - _TEXT


def test_text_set_complete() -> None:
    actual = {e for e in FileExtension if e.is_text}
    assert actual == _TEXT, (actual, _TEXT)


def test_binary_set_complete() -> None:
    actual = {e for e in FileExtension if not e.is_text}
    assert actual == _BINARY, (actual, _BINARY)


def test_pdf_is_binary() -> None:
    assert FileExtension.PDF.is_text is False


def test_md_is_text() -> None:
    assert FileExtension.MD.is_text is True
