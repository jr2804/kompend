"""Tests for ``safe_stem`` — single-source-of-truth filename stem extractor.

Unlike :attr:`pathlib.Path.stem` (which mistakes any ``.xxx`` suffix for an
extension), :func:`knox.core.paths.safe_stem` only strips extensions from a
known safe set, preserving multi-dot ITU filenames like
``T-REC-P.10-201906-I!Amd1!PDF-E`` intact when no real extension follows.

Regression: the p863 wiki lint roundup (2026-09-10) revealed that
``Path("T-REC-P.10-201906-I!Amd1!PDF-E").stem == "T-REC-P"`` corrupted
``source_files`` and ``sources[].id`` for every ITU document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.core.paths import safe_stem


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Empty / whitespace
        ("", ""),
        # ITU-T multi-dot filenames (the regression)
        (
            "sources/misc/T-REC-P.10-201906-I!Amd1!PDF-E.pdf",
            "T-REC-P.10-201906-I!Amd1!PDF-E",
        ),
        (
            "sources/misc/T-REC-P.10-201906-I!Amd1!PDF-E",
            "T-REC-P.10-201906-I!Amd1!PDF-E",
        ),
        # Standard filenames with recognized extensions
        ("sources/foo.txt", "foo"),
        ("reports/2024-Q3.pdf", "2024-Q3"),
        ("data.json", "data"),
        # Multi-dot base + recognized extension
        ("a.b.c.d.pdf", "a.b.c.d"),
        # Multi-dot base + UNRECOGNIZED tail (must NOT be stripped)
        ("a.b.c.d.weird", "a.b.c.d.weird"),
        # Edge: name is exactly the extension (must return whole name, not empty)
        (".pdf", ".pdf"),
        # Edge: trailing slash
        ("foo/", "foo"),
    ],
)
def test_safe_stem_known_extensions(raw: str, expected: str) -> None:
    assert safe_stem(raw) == expected


def test_safe_stem_accepts_path_object() -> None:
    p = Path("T-REC-P.10-201906-I!Amd1!PDF-E.pdf")
    assert safe_stem(p) == "T-REC-P.10-201906-I!Amd1!PDF-E"


def test_safe_stem_differs_from_pathlib_stem_on_itu_filenames() -> None:
    """Regression: Path(...).stem chops ``T-REC-P`` from ITU filenames."""
    raw = "T-REC-P.10-201906-I!Amd1!PDF-E"
    assert Path(raw).stem == "T-REC-P"  # the bug
    assert safe_stem(raw) == raw  # the fix
