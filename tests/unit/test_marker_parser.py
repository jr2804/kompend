"""Unit tests for the Marker parser backend.

All tests mock the heavy ``marker`` SDK calls — no real model inference runs (rule #276).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from knox.core.models.elements import ElementType
from knox.preprocessing.parsers import marker_parser as mp
from knox.preprocessing.parsers._registry import PARSER_BACKENDS
from knox.preprocessing.parsers.marker_parser import MarkerParser

SIMPLE_MARKDOWN = """# Heading One

First paragraph of text.

## Subheading

Second paragraph.

| Col A | Col B |
|-------|-------|
| 1 | 2 |
| 3 | 4 |

![Figure 1](images/figure_0_foo.png)
"""

PAGE_SEP_MARKDOWN = f"""# Page 1 Heading

Content on page 1.

{'-' * 48}

# Page 2 Heading

Content on page 2.
"""

EMPTY_MARKDOWN = ""


class _FakeMarkdownOutput:
    """Minimal stand-in for marker's MarkdownOutput."""

    def __init__(self, markdown: str = "", images: dict[str, object] | None = None) -> None:
        self.markdown = markdown
        self.images = images or {}
        self.metadata = {"pages": 1}


def _make_parser(tmp_path: Path, **kwargs: object) -> MarkerParser:
    return MarkerParser(
        image_mode="reference",
        images_dir=tmp_path / "images",
        **kwargs,
    )


@pytest.fixture
def parser(tmp_path: Path) -> MarkerParser:
    return _make_parser(tmp_path)


class TestMarkerParserRegistration:
    def test_registered_in_backend_map(self) -> None:
        assert "marker" in PARSER_BACKENDS

    def test_supported_extensions(self) -> None:
        exts = {str(e).lstrip(".").lower() for e in MarkerParser.supported_extensions}
        # PDF is the primary format; office formats also supported.
        assert "pdf" in exts
        assert "docx" in exts or "epub" in exts  # at least one office/legacy format

    def test_figure_enrichment(self) -> None:
        assert MarkerParser.figure_enrichment is False


class TestMarkerParserParse:
    def test_parse_returns_parsed_document(self, parser: MarkerParser, tmp_path: Path) -> None:
        fake_output = _FakeMarkdownOutput(SIMPLE_MARKDOWN)
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 dummy")

        with (
            patch.object(mp, "PdfConverter", return_value=lambda fp: fake_output),
            patch.object(mp, "create_model_dict", return_value={}),
            patch.object(mp, "MarkdownOutput", _FakeMarkdownOutput),
        ):
            result = parser.parse(src)

        assert result.source_file == src
        assert len(result.elements) > 0
        heading = [e for e in result.elements if e.type == ElementType.HEADING]
        assert any("Heading One" in (e.content or "") for e in heading)

    def test_parse_extracts_paragraphs(self, parser: MarkerParser, tmp_path: Path) -> None:
        fake_output = _FakeMarkdownOutput(SIMPLE_MARKDOWN)
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 dummy")

        with (
            patch.object(mp, "PdfConverter", return_value=lambda fp: fake_output),
            patch.object(mp, "create_model_dict", return_value={}),
            patch.object(mp, "MarkdownOutput", _FakeMarkdownOutput),
        ):
            result = parser.parse(src)

        paragraphs = [e for e in result.elements if e.type == ElementType.TEXT]
        texts = " ".join(str(e.content) for e in paragraphs)
        assert "First paragraph" in texts
        assert "Second paragraph" in texts

    def test_parse_extracts_table(self, parser: MarkerParser, tmp_path: Path) -> None:
        fake_output = _FakeMarkdownOutput(SIMPLE_MARKDOWN)
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 dummy")

        with (
            patch.object(mp, "PdfConverter", return_value=lambda fp: fake_output),
            patch.object(mp, "create_model_dict", return_value={}),
            patch.object(mp, "MarkdownOutput", _FakeMarkdownOutput),
        ):
            result = parser.parse(src)

        tables = [e for e in result.elements if e.type == ElementType.TABLE]
        assert len(tables) >= 1
        table = tables[0]
        assert isinstance(table.content, list)
        flat = str(table.content)
        assert "Col A" in flat or "1" in flat

    def test_parse_splits_pages(self, parser: MarkerParser, tmp_path: Path) -> None:
        fake_output = _FakeMarkdownOutput(PAGE_SEP_MARKDOWN)
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 dummy")

        with (
            patch.object(mp, "PdfConverter", return_value=lambda fp: fake_output),
            patch.object(mp, "create_model_dict", return_value={}),
            patch.object(mp, "MarkdownOutput", _FakeMarkdownOutput),
        ):
            result = parser.parse(src)

        headings = [str(e.content) for e in result.elements if e.type == ElementType.HEADING]
        assert "Page 1 Heading" in headings
        assert "Page 2 Heading" in headings

    def test_parse_empty_markdown(self, parser: MarkerParser, tmp_path: Path) -> None:
        fake_output = _FakeMarkdownOutput(EMPTY_MARKDOWN)
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 dummy")

        with (
            patch.object(mp, "PdfConverter", return_value=lambda fp: fake_output),
            patch.object(mp, "create_model_dict", return_value={}),
            patch.object(mp, "MarkdownOutput", _FakeMarkdownOutput),
        ):
            result = parser.parse(src)

        assert result.source_file == src
        assert isinstance(result.elements, list)

    def test_parse_saves_images(self, parser: MarkerParser, tmp_path: Path) -> None:
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 dummy")
        img = Image.new("RGB", (1, 1), (255, 0, 0))
        fake_output = _FakeMarkdownOutput(SIMPLE_MARKDOWN, images={"figure_0_foo.png": img})

        with (
            patch.object(mp, "PdfConverter", return_value=lambda fp: fake_output),
            patch.object(mp, "create_model_dict", return_value={}),
            patch.object(mp, "MarkdownOutput", _FakeMarkdownOutput),
        ):
            result = parser.parse(src)

        figure_elems = [e for e in result.elements if e.type == ElementType.IMAGE]
        assert len(figure_elems) >= 1
        saved = list((tmp_path / "images").glob("*.png")) + list(
            (tmp_path / "images").glob("*.jpg")
        )
        assert len(saved) >= 1

    def test_parse_assigns_page_numbers(self, parser: MarkerParser, tmp_path: Path) -> None:
        fake_output = _FakeMarkdownOutput(PAGE_SEP_MARKDOWN)
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 dummy")

        with (
            patch.object(mp, "PdfConverter", return_value=lambda fp: fake_output),
            patch.object(mp, "create_model_dict", return_value={}),
            patch.object(mp, "MarkdownOutput", _FakeMarkdownOutput),
        ):
            result = parser.parse(src)

        pages = {e.page for e in result.elements if e.page is not None}
        assert 1 in pages
        assert 2 in pages


class TestMarkerParserConfig:
    def test_converter_mode_stored(self, tmp_path: Path) -> None:
        p = MarkerParser(
            image_mode="reference",
            images_dir=tmp_path / "images",
            mode="balanced",
        )
        assert p.mode == "balanced"

    def test_default_converter_mode_is_none(self, tmp_path: Path) -> None:
        p = MarkerParser(
            image_mode="reference",
            images_dir=tmp_path / "images",
        )
        assert p.mode is None  # let marker pick its own default
