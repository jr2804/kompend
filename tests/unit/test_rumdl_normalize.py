"""Parse-time markdown normalisation (rumdl over element contents)."""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.core.models import DocumentElement, ElementType
from knox.core.text.rumdl import (
    _split_batch,
    format_markdown_text,
    normalize_element_markdown,
    rumdl_available,
)

requires_rumdl = pytest.mark.skipif(not rumdl_available(), reason="rumdl binary not on PATH")


class TestSplitBatch:
    def test_splits_on_markers(self) -> None:
        formatted = "<!--knox-part-0-->\n\nalpha\n\n<!--knox-part-1-->\n\nbeta\n"
        assert _split_batch(formatted, expected=2) == ["\nalpha\n", "\nbeta"]

    def test_mismatched_count_returns_none(self) -> None:
        formatted = "<!--knox-part-0-->\n\nalpha\n"
        assert _split_batch(formatted, expected=2) is None

    def test_no_markers_returns_none_when_expected(self) -> None:
        assert _split_batch("plain text", expected=1) is None


class TestFormatMarkdownText:
    @requires_rumdl
    def test_collapses_repeated_blank_lines(self) -> None:
        out = format_markdown_text("# Title\n\n\n\nbody   \n")
        assert out is not None
        assert "\n\n\n" not in out
        assert "body   \n" not in out  # trailing whitespace removed

    def test_empty_input_returns_none(self) -> None:
        assert format_markdown_text("   \n  ") is None


class TestNormalizeElementMarkdown:
    @requires_rumdl
    def test_strips_wrapper_whitespace_and_keeps_metadata(self) -> None:
        heading = DocumentElement(type=ElementType.HEADING, content="  Messy  ", heading_level=2)
        item = DocumentElement(type=ElementType.LIST_ITEM, content="   spaced item   ")
        code = DocumentElement(type=ElementType.CODE, content="x = 1", metadata={"language": "python"})
        changed = normalize_element_markdown([heading, item, code])
        assert changed >= 1
        assert heading.content == "Messy"
        assert item.content == "spaced item"
        # code body is opaque to rumdl — unchanged, metadata intact
        assert code.content == "x = 1"
        assert code.metadata["language"] == "python"

    @requires_rumdl
    def test_skips_non_markdown_elements(self) -> None:
        table = DocumentElement(type=ElementType.TABLE, content=[["A"], ["1"]])
        image = DocumentElement(type=ElementType.IMAGE, content="data:image/png;base64,AAA")
        changed = normalize_element_markdown([table, image])
        assert changed == 0
        assert table.content == [["A"], ["1"]]
        assert image.content == "data:image/png;base64,AAA"

    def test_no_targets_returns_zero(self) -> None:
        assert normalize_element_markdown([]) == 0

    @requires_rumdl
    def test_reading_order_and_page_are_untouched(self) -> None:
        element = DocumentElement(
            type=ElementType.HEADING,
            content="  Title  ",
            heading_level=1,
            page=7,
            reading_order=3,
        )
        normalize_element_markdown([element])
        assert element.page == 7
        assert element.reading_order == 3
        assert element.type is ElementType.HEADING


class TestParseTierWiring:
    """The parse tier must call the normaliser so cached content is canonical."""

    def test_parse_source_normalises_elements(self) -> None:
        source = Path("src/knox/core/pipeline/stages/_parse.py").read_text(encoding="utf-8")
        assert "normalize_element_markdown(" in source
        # must be applied after relativisation, before the cache write
        assert source.index("normalize_element_markdown(") < source.index("_save_cache(pipeline, source_file, parsed)")

    def test_ingest_normalises_elements(self) -> None:
        source = Path("src/knox/core/pipeline/ingest.py").read_text(encoding="utf-8")
        assert "normalize_element_markdown(" in source

    def test_attachments_normalise_elements(self) -> None:
        source = Path("src/knox/core/pipeline/attachments.py").read_text(encoding="utf-8")
        assert "normalize_element_markdown(" in source
