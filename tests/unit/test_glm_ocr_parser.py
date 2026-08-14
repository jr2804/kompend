"""Unit tests for the GLM-OCR parser backend.

All tests mock ``glmocr.parse`` — no real API calls (rule #276).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

try:
    from knox.core.models.enums import ElementType
    from knox.preprocessing.parsers._registry import (
        KNOWN_BACKENDS,
        PARSER_BACKENDS,
        build_default_router,
        get_registered_parsers,
    )
    from knox.preprocessing.parsers.base import FileExtension
    from knox.preprocessing.parsers.glm_ocr_parser import (
        GlmOcrParser,
        _coerce_pages,
        _markdown_table_to_grid,
    )
except ImportError:
    pytest.importorskip("glmocr")

# ── Registration ──────────────────────────────────────────────────────────


class TestGlmOcrRegistration:
    def test_glm_ocr_is_known_backend(self) -> None:
        assert "glm-ocr" in KNOWN_BACKENDS

    def test_glm_ocr_is_installed_and_registered(self) -> None:
        """Glmocr is installed in this env, so the backend must register."""
        assert "glm-ocr" in PARSER_BACKENDS

    def test_glm_ocr_in_registered_parsers(self) -> None:
        names = {p["name"] for p in get_registered_parsers()}
        assert "glm-ocr" in names

    def test_build_default_router_includes_glm_ocr(self) -> None:
        router = build_default_router(names="glm-ocr")
        parser_types = {type(p).__name__ for p in router._parsers}
        assert "GlmOcrParser" in parser_types

    def test_supported_extensions_include_images_and_pdf(self) -> None:
        exts = GlmOcrParser.supported_extensions
        assert FileExtension.PDF in exts
        assert FileExtension.PNG in exts
        assert FileExtension.JPG in exts
        assert FileExtension.TIFF in exts
        assert FileExtension.BMP in exts


class TestGlmOcrParse:
    def test_parse_empty_result_returns_empty_document(self, tmp_path: Path) -> None:
        src = tmp_path / "scan.png"
        src.write_bytes(b"\x89PNG fake")
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=None):
            doc = parser.parse(src)
        assert doc.elements == []
        assert doc.metadata["parser"] == "glm-ocr"
        assert doc.metadata["error"] == "no result"

    def test_parse_text_elements(self, tmp_path: Path) -> None:
        src = tmp_path / "page.png"
        src.write_bytes(b"\x89PNG fake")
        fake = _fake_result(
            json_result=[
                [
                    {"index": 0, "label": "title", "content": "Document Title", "bbox_2d": None},
                    {"index": 1, "label": "text", "content": "Hello world.", "bbox_2d": None},
                ]
            ],
        )
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=fake):
            doc = parser.parse(src)
        assert len(doc.elements) == 2
        assert doc.elements[0].type == ElementType.HEADING
        assert doc.elements[0].content == "Document Title"
        assert doc.elements[1].type == ElementType.TEXT
        assert doc.elements[1].content == "Hello world."
        assert doc.title == "Document Title"

    def test_parse_table_element_from_markdown(self, tmp_path: Path) -> None:
        src = tmp_path / "table.png"
        src.write_bytes(b"\x89PNG fake")
        table_md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        fake = _fake_result(
            json_result=[
                [
                    {"index": 0, "label": "table", "content": table_md, "bbox_2d": [0, 0, 100, 50]},
                ]
            ],
        )
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=fake):
            doc = parser.parse(src)
        assert len(doc.elements) == 1
        el = doc.elements[0]
        assert el.type == ElementType.TABLE
        assert el.content == [["A", "B"], ["1", "2"]]
        assert el.metadata["label"] == "table"
        assert el.metadata["bbox"] == [0, 0, 100, 50]

    def test_parse_formula_element(self, tmp_path: Path) -> None:
        src = tmp_path / "formula.png"
        src.write_bytes(b"\x89PNG fake")
        fake = _fake_result(
            json_result=[
                [
                    {"index": 0, "label": "formula", "content": "E = mc^2", "bbox_2d": None},
                ]
            ],
        )
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=fake):
            doc = parser.parse(src)
        assert len(doc.elements) == 1
        assert doc.elements[0].type == ElementType.FORMULA
        assert doc.elements[0].content == "E = mc^2"

    def test_parse_page_count_from_pages(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF fake")
        fake = _fake_result(
            json_result=[
                [{"index": 0, "label": "text", "content": "p1", "bbox_2d": None}],
                [{"index": 0, "label": "text", "content": "p2", "bbox_2d": None}],
            ],
        )
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=fake):
            doc = parser.parse(src)
        assert doc.page_count == 2
        assert doc.elements[0].page == 1
        assert doc.elements[1].page == 2

    def test_parse_unknown_label_defaults_to_text(self, tmp_path: Path) -> None:
        src = tmp_path / "weird.png"
        src.write_bytes(b"\x89PNG fake")
        fake = _fake_result(
            json_result=[
                [
                    {"index": 0, "label": "mystery_label", "content": "???"},
                ]
            ],
        )
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=fake):
            doc = parser.parse(src)
        assert len(doc.elements) == 1
        assert doc.elements[0].type == ElementType.TEXT

    def test_parse_skips_none_content(self, tmp_path: Path) -> None:
        src = tmp_path / "skip.png"
        src.write_bytes(b"\x89PNG fake")
        fake = _fake_result(
            json_result=[
                [
                    {"index": 0, "label": "text", "content": None},
                    {"index": 1, "label": "text", "content": "kept"},
                ]
            ],
        )
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=fake):
            doc = parser.parse(src)
        assert len(doc.elements) == 1
        assert doc.elements[0].content == "kept"

    def test_parse_reading_order_increments(self, tmp_path: Path) -> None:
        src = tmp_path / "order.png"
        src.write_bytes(b"\x89PNG fake")
        fake = _fake_result(
            json_result=[
                [
                    {"index": 0, "label": "text", "content": "a"},
                    {"index": 1, "label": "text", "content": "b"},
                    {"index": 2, "label": "text", "content": "c"},
                ]
            ],
        )
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=fake):
            doc = parser.parse(src)
        orders = [e.reading_order for e in doc.elements]
        assert orders == [0, 1, 2]


# ── Helpers ───────────────────────────────────────────────────────────────


class TestCoercePages:
    def test_paginated_list(self) -> None:
        result = [[{"label": "text", "content": "a"}], [{"label": "text", "content": "b"}]]
        assert len(_coerce_pages(result)) == 2

    def test_flat_list_treated_as_one_page(self) -> None:
        result = [{"label": "text", "content": "a"}, {"label": "text", "content": "b"}]
        pages = _coerce_pages(result)
        assert len(pages) == 1
        assert len(pages[0]) == 2

    def test_single_dict(self) -> None:
        result = {"label": "text", "content": "a"}
        pages = _coerce_pages(result)
        assert pages == [[result]]

    def test_json_string(self) -> None:
        result = json.dumps([[{"label": "text", "content": "a"}]])
        assert len(_coerce_pages(result)) == 1

    def test_empty(self) -> None:
        assert _coerce_pages([]) == []


class TestMarkdownTableToGrid:
    def test_simple_table(self) -> None:
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        grid = _markdown_table_to_grid(md)
        assert grid == [["A", "B"], ["1", "2"]]

    def test_skips_separator(self) -> None:
        md = "| --- | :---: |\n| 1 | 2 |\n"
        grid = _markdown_table_to_grid(md)
        assert grid == [["1", "2"]]

    def test_empty_returns_empty(self) -> None:
        assert _markdown_table_to_grid("") == []


# ── Fallback to markdown ─────────────────────────────────────────────────


class TestMarkdownFallback:
    def test_falls_back_to_markdown_when_no_json(self, tmp_path: Path) -> None:
        src = tmp_path / "fallback.png"
        src.write_bytes(b"\x89PNG fake")
        fake = _fake_result(
            json_result=[],
            markdown_result="# Title\n\nParagraph one.\n\nParagraph two.",
        )
        parser = GlmOcrParser()
        with patch("knox.preprocessing.parsers.glm_ocr_parser.glm_parse", return_value=fake):
            doc = parser.parse(src)
        # 3 blocks: heading + 2 paragraphs.
        assert len(doc.elements) == 3
        assert doc.elements[0].type == ElementType.HEADING
        assert doc.title == "Title"


# ── Parsing (mocked) ──────────────────────────────────────────────────────


def _fake_result(
    json_result: list | None = None,
    markdown_result: str = "",
    image_files: dict | None = None,
    original_images: list[str] | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace mimicking PipelineResult."""
    return SimpleNamespace(
        json_result=json_result or [],
        markdown_result=markdown_result,
        image_files=image_files or {},
        original_images=original_images or [],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
