"""Regression: the format table classifies sources, it never gates parsing.

``ingest_documents`` (like ``_parse.py``) must gate
``normalize_to_utf8_path`` on ``FileExtension.is_text`` — binary
containers (PDF/DOCX/raster) are consumed byte-wise by their parser and
would raise ``ValueError("binary")`` if handed to charset detection.
See memory #802 (the compile-path gate) — ingest kept an ungated call
until ISO226.pdf exposed it (``kompend -w wiki ingest`` → ``(1 failed)``
with no diagnostics).

The same gate must not reject a source whose suffix the table does not
know: backends read more than kompend's table lists (upstream issue #3 —
``.c``/``.h``/``.m``/``.py`` were dropped with ``'.c' is not a valid
FileExtension`` even though the backend parses them).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from kompend.core.models import DocumentElement, ElementType, ParsedDocument
from kompend.core.pipeline.ingest import (
    ingest_documents,
    parse_target_path,
    restore_source_path,
)
from kompend.core.pipeline.parse_cache import cache_key as document_cache_key
from kompend.core.pipeline.parse_cache import read_document
from kompend.core.pipeline.state import TierStateStore
from kompend.core.store import vault_db_path
from kompend.core.text.encoding import detect_and_decode
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


def test_code_source_reaches_parser(tmp_path: Path) -> None:
    """A suffix the format table does not know must still reach the parser.

    Upstream issue #3: ``FileExtension(".c")`` raised ``ValueError``, so the
    ingest tier dropped every code file the backend can read.
    """
    base = _seed_vault(tmp_path)
    src = base / "sources" / "ACcontextMapping.c"
    src.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    router = _FakeRouter()

    result = ingest_documents(
        files=[src],
        base_dir=base,
        router=cast("ParserRouter", router),
        no_cache=True,
        config=None,
    )

    assert result.failed == 0
    assert result.count == 1
    # The parser received the original file — no encoding sibling, no drop.
    assert router.parsed == [src]


def test_parse_target_path_keeps_unknown_suffix_byte_wise(tmp_path: Path) -> None:
    """Source code is handed to the backend as-is, not re-encoded."""
    src = tmp_path / "ACcontextMapping.c"
    src.write_text("int main(void) { return 0; }\n", encoding="utf-8")

    assert parse_target_path(src, tmp_path / "cache") == src


def test_parse_target_path_keeps_binary_container_byte_wise(tmp_path: Path) -> None:
    """Binary containers never reach charset detection."""
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4\x00\xff\x00binary\x00\x01")  # not valid UTF-8

    assert parse_target_path(src, tmp_path / "cache") == src


def test_parse_target_path_normalizes_known_text_source(tmp_path: Path) -> None:
    """A known text format that is not UTF-8 is handed over as UTF-8."""
    src = tmp_path / "note.md"
    src.write_bytes("# café\n".encode("latin-1"))
    cache = tmp_path / "cache"
    expected, _encoding = detect_and_decode(src)

    target = parse_target_path(src, cache)

    assert target != src
    assert target.parent == cache / "encoding"
    assert target.suffix == src.suffix, "backend classifies by suffix — keep it"
    assert target.read_text(encoding="utf-8") == expected


def test_non_utf8_markdown_is_ingested_not_dropped(tmp_path: Path) -> None:
    """A Latin-1 markdown source reaches the parser with a classifiable suffix.

    Regression for knox-0mtq: the normalized sibling used to be named
    ``<hash>.utf8``. Backends classify input by suffix
    (``media_type_for``), so every non-UTF-8 text source raised
    ``UnsupportedSourceError`` and was dropped.
    """
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    src.write_bytes("# café\n".encode("latin-1"))
    router = _FakeRouter()

    result = ingest_documents(
        files=[src],
        base_dir=base,
        router=cast("ParserRouter", router),
        no_cache=True,
        config=None,
    )

    assert result.failed == 0
    assert result.count == 1
    (target,) = router.parsed
    assert target != src
    assert target.suffix == ".md"
    expected, _encoding = detect_and_decode(src)
    assert target.read_text(encoding="utf-8") == expected


def test_encoding_sibling_does_not_leak_into_the_cache_entry(tmp_path: Path) -> None:
    """The cached record points at the real source, not the encoding cache."""
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    src.write_bytes("# café\n".encode("latin-1"))
    router = _FakeRouter()

    ingest_documents(
        files=[src],
        base_dir=base,
        router=cast("ParserRouter", router),
        no_cache=True,
        config=None,
    )

    content_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    data = read_document(vault_db_path(base), document_cache_key(src, content_hash))
    assert data is not None
    assert Path(data["source_file"]).as_posix() == "sources/note.md"


def test_restore_source_path_rewrites_document_and_elements() -> None:
    """Both the document and the elements that read the sibling are restored."""
    sibling = Path("/cache/encoding/abc.md")
    origin = Path("/vault/sources/note.md")
    doc = ParsedDocument(
        source_file=sibling,
        elements=[
            DocumentElement(type=ElementType.TEXT, content="a", source_file=sibling),
            DocumentElement(type=ElementType.TEXT, content="b", source_file=origin),
        ],
        title="",
    )

    restore_source_path(doc, parse_target=sibling, source_file=origin)

    assert doc.source_file == origin
    assert [el.source_file for el in doc.elements] == [origin, origin]


def test_restore_source_path_is_a_noop_without_an_encoding_sibling() -> None:
    """UTF-8 sources are parsed in place; nothing is rewritten."""
    origin = Path("/vault/sources/note.md")
    doc = ParsedDocument(
        source_file=origin,
        elements=[DocumentElement(type=ElementType.TEXT, content="a", source_file=origin)],
        title="",
    )

    restore_source_path(doc, parse_target=origin, source_file=origin)

    assert doc.source_file == origin


def _seed_vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir(parents=True)
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    return base
