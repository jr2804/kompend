"""Tests for the pluggable ingestion backend registry.

Backends self-register at import time via ``register_backend``; importing the
``knox.preprocessing.parsers`` package (done by the imports below) triggers
registration before any assertion runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from knox.core.models import ParsedDocument
from knox.core.models.elements import DocumentElement
from knox.core.models.enums import ElementType
from knox.preprocessing.parsers._registry import (
    PARSER_BACKENDS,
    ParserBuildOptions,
    _normalize_backends,
    build_default_router,
    get_all_extensions,
    get_backend,
    get_registered_parsers,
    register_backend,
)
from knox.preprocessing.parsers.base import IMAGE_SCALE_DEFAULT, BaseParser, FileExtension, ParserRouter, scrub_document
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.preprocessing.parsers.liteparse_parser import LiteParseParser
from knox.preprocessing.parsers.pandoc_parser import PandocParser
from knox.preprocessing.parsers.web_parser import WebPageParser


class TestWebPageParser:
    """knox-4sq5: WebPageParser registered as 'web' backend, handles .html."""

    def test_web_backend_registered(self) -> None:
        assert "web" in get_registered_parsers()[0]["name"] or any(p["name"] == "web" for p in get_registered_parsers())

    def test_web_routes_html_before_docling(self) -> None:
        router = build_default_router(names="web,liteparse,docling")
        assert isinstance(router.route(Path("page.html")), WebPageParser)
        # PDF still goes to liteparse.
        assert isinstance(router.route(Path("paper.pdf")), LiteParseParser)

    def test_web_parser_clean_text_from_html(self, tmp_path: Path) -> None:
        """WebPageParser extracts clean text, not raw HTML tags."""
        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><body><h1>Attention Mechanisms</h1><p>Self-attention allows models to relate positions.</p></body></html>",
            encoding="utf-8",
        )

        parser = WebPageParser()
        doc = parser.parse(html_file)
        assert isinstance(doc, ParsedDocument)
        assert doc.source_file == html_file
        all_text = " ".join(e.content for e in doc.elements if isinstance(e.content, str))
        assert "Attention Mechanisms" in all_text or "attention" in all_text.lower()
        assert "<html>" not in all_text
        assert "<p>" not in all_text

    def test_web_parser_can_parse_html_only(self, tmp_path: Path) -> None:
        parser = WebPageParser()
        assert parser.can_parse(Path("page.html"))
        assert parser.can_parse(Path("page.HTML"))
        assert not parser.can_parse(Path("paper.pdf"))


def test_builtin_backends_registered() -> None:
    assert {"docling", "liteparse", "pandoc"}.issubset(PARSER_BACKENDS)


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("docling", DoclingParser),
        ("liteparse", LiteParseParser),
        ("pandoc", PandocParser),
    ],
)
def test_get_backend_factory_builds_expected_class(name: str, expected_cls: type[BaseParser]) -> None:
    parser = get_backend(name)(ParserBuildOptions())
    assert isinstance(parser, expected_cls)


def test_get_backend_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown parser backend"):
        get_backend("does-not-exist")


def test_parser_build_options_defaults() -> None:
    opts = ParserBuildOptions()
    assert opts.images_dir is None
    assert opts.image_mode == "file"
    assert opts.images_scale == IMAGE_SCALE_DEFAULT
    assert opts.pdf_config is None


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("docling", DoclingParser),
        ("liteparse", LiteParseParser),
    ],
)
def test_build_default_router_selects_primary(name: str, expected_cls: type[BaseParser]) -> None:
    router = build_default_router(names=name)
    # build_default_router registers the primary first, then the pandoc fallback.
    assert isinstance(router._parsers[0], expected_cls)
    assert any(isinstance(p, PandocParser) for p in router._parsers)


def test_build_default_router_pandoc_fallback_routes_txt() -> None:
    router = build_default_router()
    routed = router.route(Path("notes.txt"))
    assert isinstance(routed, PandocParser)


def test_build_default_router_options_forwarded(tmp_path: Path) -> None:
    opts = ParserBuildOptions(images_dir=tmp_path, image_mode="base64", images_scale=100.0)
    router = build_default_router(opts, names="docling")
    docling = router._parsers[0]
    assert isinstance(docling, DoclingParser)
    assert docling.images_dir == tmp_path
    assert docling.image_mode == "base64"
    assert docling.images_scale == 100.0


def test_register_backend_populates_metadata() -> None:
    cls = _dummy_cls()
    register_backend("_test_dummy", lambda _opts: cls(), parser_cls=cls)
    assert get_backend("_test_dummy") is not None
    names = {entry["name"] for entry in get_registered_parsers()}
    assert "_test_dummy" in names


def _dummy_cls() -> type[BaseParser]:
    class _Dummy(BaseParser):
        supported_extensions = set()  # type: ignore[assignment]
        figure_enrichment = True

        def parse(self, source_file: Path) -> ParsedDocument:
            raise NotImplementedError

    return _Dummy


def test_figure_enrichment_metadata_matches_class_attr() -> None:
    by_name = {entry["name"]: entry for entry in get_registered_parsers()}
    assert by_name["docling"]["figure_enrichment"] is DoclingParser.figure_enrichment
    assert by_name["liteparse"]["figure_enrichment"] is LiteParseParser.figure_enrichment
    assert by_name["pandoc"]["figure_enrichment"] is PandocParser.figure_enrichment


def test_get_all_extensions_covers_known_suffixes() -> None:
    extensions = get_all_extensions()
    assert ".pdf" in extensions
    assert ".docx" in extensions


def test_normalize_backends_accepts_string_and_sequence() -> None:
    assert _normalize_backends("liteparse,docling") == ["liteparse", "docling"]
    assert _normalize_backends(" docling ,, liteparse ") == ["docling", "liteparse"]
    assert _normalize_backends(["liteparse", "docling"]) == ["liteparse", "docling"]
    # duplicates removed, order preserved
    assert _normalize_backends("liteparse,liteparse,docling") == ["liteparse", "docling"]
    # blanks/empty fall back to liteparse
    assert _normalize_backends("") == ["liteparse"]
    assert _normalize_backends(",,,") == ["liteparse"]


def test_build_default_router_priority_order_routes_by_format() -> None:
    """A priority list routes each format to the first backend that supports it."""
    router = build_default_router(names="liteparse,docling")
    # liteparse is listed first and supports PDF → it wins over docling.
    assert isinstance(router.route(Path("paper.pdf")), LiteParseParser)
    # liteparse has no HTML/TEX → docling (next in the chain) handles them.
    assert isinstance(router.route(Path("page.html")), DoclingParser)
    assert isinstance(router.route(Path("report.tex")), DoclingParser)
    # neither liteparse nor docling supports DOC → pandoc (auto-appended).
    assert isinstance(router.route(Path("legacy.doc")), PandocParser)


def test_build_default_router_always_appends_pandoc() -> None:
    """Pandoc is auto-appended even when not listed, so legacy formats don't regress."""
    router = build_default_router(names="liteparse")
    assert isinstance(router.route(Path("legacy.doc")), PandocParser)
    assert any(isinstance(p, PandocParser) for p in router._parsers)


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


def test_resolve_format_unknown_text_falls_back_to_markdown() -> None:
    """PandocParser._READER_FOR is the SSOT for extension→pandoc-reader mapping.

    FileExtension is the SSOT for *which* extensions exist (enum members);
    each parser owns the metadata that's specific to it. Unknown text
    extensions fall back to FileExtension.TXT → ``"markdown"`` at parse
    time, never ``""`` (pypandoc would infer the reader from the file
    extension and crash on extensions it cannot read, e.g. ``.c``, ``.ltx``).
    """
    readers = PandocParser._READER_FOR
    assert readers[FileExtension.TXT] == "markdown"
    assert readers[FileExtension.MD] == "markdown"
    assert readers[FileExtension.TEX] == "latex"
    assert readers[FileExtension.LTX] == "latex"
    assert readers[FileExtension.DOC] == "docx"
    assert readers[FileExtension.ADOC] == "asciidoc"


def test_router_parse_scrubs_parser_output(tmp_path: Path) -> None:
    class FakeParser(BaseParser):
        def can_parse(self, source_file: Path) -> bool:
            return True

        def parse(self, source_file: Path) -> ParsedDocument:
            return _make_doc_with_surrogates()

    router = ParserRouter()
    router.register(FakeParser())
    doc = router.parse(tmp_path / "broken.pdf")
    assert doc.elements[0].content == "ok \ufffd mid \ufffd end"


# ── Surrogate scrubbing (ParserRouter.parse boundary) ─────────────────────


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
