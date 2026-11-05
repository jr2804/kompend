"""Tests for the single-parser surface.

kompend ships one ingestion backend (parsecraft), so there is no registry: no
``ParserName``, no dynamic backend lookup, no chain. These tests pin what
replaced it — :func:`build_parser`, :class:`DocumentParser` and
:class:`ParserOptions`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from parsecraft.cli.config import ParsecraftConfig, RoutingSettings

from kompend.core.config import resolve_config
from kompend.core.models import ParsedDocument
from kompend.core.models.elements import DocumentElement
from kompend.core.models.enums import ElementType
from kompend.core.paths import PARSECRAFT_PROJECT_FILE, VaultPaths
from kompend.preprocessing.parsers import (
    PARSER_PARSECRAFT,
    DocumentParser,
    FileExtension,
    ParserOptions,
    build_parser,
    known_extension,
    scrub_document,
)
from kompend.preprocessing.parsers.extensions import supported_extensions
from kompend.preprocessing.parsers.parsecraft_parser import ParsecraftParser
from kompend.preprocessing.parsers.parsecraft_routing import ParsecraftRoutingConfig


def test_the_one_parser_is_the_configured_one() -> None:
    """``build_parser`` returns the parsecraft adapter, guarded."""
    parser = build_parser()
    assert isinstance(parser, DocumentParser)
    assert isinstance(parser.parser, ParsecraftParser)
    assert PARSER_PARSECRAFT == "parsecraft"


def test_parser_options_defaults() -> None:
    opts = ParserOptions()
    assert opts.images_dir is None
    assert opts.image_mode == "file"
    assert opts.images_scale > 0
    # Only what is kompend's own lives here; every routing lever belongs to
    # parsecraft's config files.
    assert opts.parsecraft_judge == "none"
    assert opts.parsecraft_judge_model == ""
    assert opts.parsecraft_config_files == ()


def test_the_vault_parsecraft_file_is_part_of_the_parser_options(tmp_path: Path) -> None:
    """``from_config`` points the adapter at the vault's own parsecraft.toml.

    The vault file is a PROJECT-layer file for parsecraft's engine, so it must be
    the one that outranks the global file — passing only the global path would
    make a vault-local routing table silently inert.
    """
    options = ParserOptions.from_config(resolve_config(), base_dir=tmp_path)

    assert options.parsecraft_config_files[0] == VaultPaths(tmp_path).parsecraft_config_file
    assert options.parsecraft_config_files[1].name == PARSECRAFT_PROJECT_FILE


def test_backend_options_are_merged_over_the_derived_ones() -> None:
    """A user option wins over the adapter's derived default for the same key."""
    settings = ParsecraftConfig(routing=RoutingSettings(backend_options={"liteparse": {"extract_images": False, "extra": 1}}))
    parser = ParsecraftParser(parsecraft_config=ParsecraftRoutingConfig.from_settings(settings))

    options = parser._backend_options()  # noqa: SLF001 — the merged options are the contract

    assert options["liteparse"]["extract_images"] is False
    assert options["liteparse"]["extra"] == 1


def test_from_options_builds_the_adapter() -> None:
    """``ParsecraftParser.from_options`` is the one bridge from config to constructor."""
    parser = ParsecraftParser.from_options(
        ParserOptions(
            parsecraft_judge_model="parable-tinyjev",
            parsecraft_config=ParsecraftRoutingConfig.from_settings(ParsecraftConfig(routing=RoutingSettings(allow_ocr=False))),
        ),
    )
    assert parser._judge_model == "parable-tinyjev"  # noqa: SLF001 — the wiring is the contract


def test_supported_extensions_union_the_table_with_the_parser() -> None:
    """Discovery is the parser's own suffix set plus kompend's format table."""
    extensions = supported_extensions()
    assert {str(ext) for ext in FileExtension} <= extensions
    assert ".pdf" in extensions
    assert ".docx" in extensions
    # parsecraft reads source files as plain text through its native-text
    # backend, so a vault can ingest code.
    assert ".c" in extensions


def test_a_known_text_suffix_resolves_and_an_unknown_one_does_not() -> None:
    assert known_extension(".MD") is FileExtension.MD
    assert known_extension(".c") is None


def test_scrub_document_replaces_lone_surrogates() -> None:
    doc = scrub_document(_make_doc_with_surrogates())
    assert doc.elements[0].content == "ok \ufffd mid \ufffd end"
    assert doc.elements[0].metadata == {"caption": ["fig \ufffd"]}
    assert doc.elements[1].content == [["a\ufffd", "b"]]
    assert doc.title == "t\ufffd"
    assert doc.author == "a\ufffd"
    assert doc.metadata == {"nested": {"v": "x\ufffd"}}
    # The whole document must now be strictly UTF-8 encodable.
    json.dumps(asdict(doc), ensure_ascii=False, default=str).encode("utf-8")


def test_document_parser_scrubs_the_parser_output(tmp_path: Path) -> None:
    class _FakeParser:
        figure_enrichment = False
        supported_extensions = {".pdf"}

        def can_parse(self, source_file: Path) -> bool:
            return True

        def parse(self, source_file: Path) -> ParsedDocument:
            return _make_doc_with_surrogates()

    parser = DocumentParser(_FakeParser())
    doc = parser.parse(tmp_path / "broken.pdf")
    assert doc.elements[0].content == "ok \ufffd mid \ufffd end"


def test_document_parser_does_not_claim_a_suffix_its_parser_refuses(tmp_path: Path) -> None:
    class _FakeParser:
        figure_enrichment = False
        supported_extensions = {".pdf"}

        def can_parse(self, source_file: Path) -> bool:
            return False

        def parse(self, source_file: Path) -> ParsedDocument:
            raise AssertionError("must not parse")

    assert DocumentParser(_FakeParser()).route(tmp_path / "x.pdf") is None


def test_document_parser_refuses_a_mislabelled_text_source(tmp_path: Path) -> None:
    """A binary file named ``.md`` is refused before the parser sees it."""
    mislabelled = tmp_path / "notes.md"
    mislabelled.write_bytes(b"%PDF-1.4\x00\x01\x02\x03binary payload")

    class _FakeParser:
        figure_enrichment = False
        supported_extensions = {".pdf"}

        def can_parse(self, source_file: Path) -> bool:
            return True

        def parse(self, source_file: Path) -> ParsedDocument:
            raise AssertionError("must not parse")

    with pytest.raises(ValueError, match="claims text"):
        DocumentParser(_FakeParser()).parse(mislabelled)


# ── Surrogate scrubbing fixture ───────────────────────────────────────────


def _make_doc_with_surrogates() -> ParsedDocument:
    return ParsedDocument(
        source_file=Path("broken.pdf"),
        elements=[
            DocumentElement(
                type=ElementType.TEXT,
                content="ok \ud800 mid \udfff end",
                metadata={"caption": ["fig \udc80"]},
            ),
            DocumentElement(type=ElementType.TABLE, content=[["a\ud900", "b"]]),
        ],
        title="t\ud83d",
        author="a\udead",
        metadata={"nested": {"v": "x\udabc"}},
    )
