"""Verify FileExtension.is_text classification matches the encoding-layer expectation."""

from __future__ import annotations

import pytest

from kompend.preprocessing.parsers.base import FileExtension, known_extension

_TEXT = {
    FileExtension.HTML,
    FileExtension.HTM,
    FileExtension.MD,
    FileExtension.MARKDOWN,
    FileExtension.TXT,
    FileExtension.CSV,
    FileExtension.TSV,
    FileExtension.RST,
    FileExtension.TEX,
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


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".md", FileExtension.MD),
        (".MD", FileExtension.MD),
        (".pdf", FileExtension.PDF),
    ],
)
def test_known_extension_resolves_table_suffix(suffix: str, expected: FileExtension) -> None:
    assert known_extension(suffix) is expected


@pytest.mark.parametrize("suffix", [".c", ".h", ".m", ".py", ".sh", ".cfg", ".ini", "", ".adoc", ".bmp", ".eml", ".gif", ".ltx", ".msg", ".webp"])
def test_known_extension_is_none_for_unknown_suffix(suffix: str) -> None:
    """A suffix outside the table is a fact about the format, not an error.

    The last seven were pruned from the table (knox-8tsj): no parser on
    either side honours them, so the table no longer names them.
    """
    assert known_extension(suffix) is None
