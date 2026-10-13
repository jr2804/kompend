"""Regression: ingest must not funnel binary sources through encoding normalization.

``ingest_documents`` (like ``_parse.py``) must gate
``normalize_to_utf8_path`` on ``FileExtension.is_text`` — binary
containers (PDF/DOCX/raster) are consumed byte-wise by their parser and
would raise ``ValueError("binary")`` if handed to charset detection.
See memory #802 (the compile-path gate) — ingest kept an ungated call
until ISO226.pdf exposed it (``kompend -w wiki ingest`` → ``(1 failed)``
with no diagnostics).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from kompend.core.models import ParsedDocument
from kompend.core.pipeline.ingest import ingest_documents
from kompend.core.pipeline.state import TierStateStore
from kompend.preprocessing.parsers import ParserRouter


class _FakeRouter:
    def __init__(self) -> None:
        self.parsed: list[Path] = []

    def fingerprint(self) -> str:  # noqa: D102
        return "fake-parser-chain"

    def route(self, source_file: Path) -> MagicMock:  # noqa: D102
        parser = _fake_parser()

        def _parse(target: Path) -> MagicMock:
            self.parsed.append(Path(target))
            return parser.parse.return_value

        parser.parse.side_effect = _parse
        return parser


def _fake_parser() -> MagicMock:
    doc = ParsedDocument(source_file=Path("sources"), elements=[], title="")
    parser = MagicMock()
    parser.parse.return_value = doc
    return parser


def test_binary_pdf_not_funneled_through_normalize(tmp_path: Path) -> None:
    """PDF bytes reach the parser as-is; normalize_to_utf8_path must not see them."""
    base = _seed_vault(tmp_path)
    src = base / "sources" / "doc.pdf"
    src.write_bytes(b"%PDF-1.4\x00\xff\x00binary\x00\x01")  # not valid UTF-8
    router = _FakeRouter()

    with patch("kompend.core.pipeline.ingest.normalize_to_utf8_path") as mock_norm:
        result = ingest_documents(
            files=[src],
            base_dir=base,
            router=cast("ParserRouter", router),
            no_cache=True,
            config=None,
        )

    assert result.failed == 0
    assert result.count == 1
    mock_norm.assert_not_called()
    # The parser received the original file path, not an encoding sibling.
    assert router.parsed == [src]
    # Tier-1 state recorded.
    state = TierStateStore(base / ".kompend" / "compile_state.json").load()
    assert "doc" in state.entries


def test_text_md_still_normalized(tmp_path: Path) -> None:
    """Text sources keep the normalization gate — behavior preserved."""
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    src.write_text("# Note\n", encoding="utf-8")
    router = _FakeRouter()

    with patch("kompend.core.pipeline.ingest.normalize_to_utf8_path") as mock_norm:
        mock_norm.return_value = (src, None)  # UTF-8-clean → passthrough
        result = ingest_documents(
            files=[src],
            base_dir=base,
            router=cast("ParserRouter", router),
            no_cache=True,
            config=None,
        )

    assert result.failed == 0
    mock_norm.assert_called_once_with(src, base / ".kompend" / "cache")


def _seed_vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir(parents=True)
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    return base
