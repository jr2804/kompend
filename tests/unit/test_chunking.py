"""Tests for the KNOX HybridChunker."""

from pathlib import Path

from knox.chunking.hybrid import HybridChunker
from knox.core.config._core import DEFAULTS
from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.llm.client import ContextWindow


def _make_doc(elements: list[DocumentElement], title: str = "test") -> ParsedDocument:
    return ParsedDocument(
        source_path=Path(f"{title}.pdf"),
        elements=elements,
        title=title,
    )


class TestHybridChunker:
    """Test suite for HybridChunker."""

    def test_empty_document(self) -> None:
        doc = _make_doc([])
        result = HybridChunker(max_tokens=512).chunk(doc)
        assert result == []

    def test_single_text(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="Hello world."),
            ]
        )
        result = HybridChunker(max_tokens=512).chunk(doc)
        assert len(result) >= 1
        assert ElementType.TEXT in result[0].element_types

    def test_table_is_atomic(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Data", heading_level=1),
                DocumentElement(
                    type=ElementType.TABLE,
                    content=[["A", "B"], ["1", "2"]],
                ),
            ]
        )
        result = HybridChunker(max_tokens=512).chunk(doc)
        table_chunks = [c for c in result if ElementType.TABLE in c.element_types]
        assert len(table_chunks) == 1, "TABLE must never be split"
        assert "A | B" in table_chunks[0].text

    def test_code_is_atomic(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.CODE, content="print('hello')"),
            ]
        )
        result = HybridChunker(max_tokens=256).chunk(doc)
        code_chunks = [c for c in result if ElementType.CODE in c.element_types]
        assert len(code_chunks) == 1

    def test_image_is_atomic(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="See figure below."),
                DocumentElement(type=ElementType.IMAGE, content=b"fake_png_data"),
            ]
        )
        result = HybridChunker(max_tokens=512).chunk(doc)
        img_chunks = [c for c in result if ElementType.IMAGE in c.element_types]
        assert len(img_chunks) == 1

    def test_heading_context_preserved(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Results", heading_level=1),
                DocumentElement(type=ElementType.TEXT, content="Significant findings."),
            ]
        )
        result = HybridChunker(max_tokens=512).chunk(doc)
        assert any("Results" in c.heading_context or "Results" in c.text for c in result)

    def test_long_text_is_chunked(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content=("word " * 500)),
            ]
        )
        result = HybridChunker(max_tokens=100).chunk(doc)
        assert len(result) > 1, "Long text should produce multiple chunks"

    def test_mixed_content(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Intro", heading_level=1),
                DocumentElement(type=ElementType.TEXT, content="Some intro text."),
                DocumentElement(type=ElementType.TABLE, content=[["X"]]),
                DocumentElement(type=ElementType.HEADING, content="Conclusion", heading_level=1),
                DocumentElement(type=ElementType.TEXT, content="Final remarks."),
            ]
        )
        result = HybridChunker(max_tokens=256).chunk(doc)
        assert len(result) >= 3  # text_run + table + text_run
        types_seen = set()
        for c in result:
            types_seen.update(c.element_types)
        assert ElementType.TEXT in types_seen or ElementType.HEADING in types_seen
        # TABLE must be present as atomic
        assert any(ElementType.TABLE in c.element_types for c in result)

    def test_reserve_ratio_effective(self) -> None:
        """20% reserve reduces effective max_tokens."""
        chunker = HybridChunker(max_tokens=1000, reserve_ratio=0.2)
        assert chunker._effective_max == 800
        chunker2 = HybridChunker(max_tokens=1000, reserve_ratio=0.0)
        assert chunker2._effective_max == 1000

    def test_chunk_ids_assigned(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="First."),
                DocumentElement(type=ElementType.TEXT, content="Second."),
                DocumentElement(type=ElementType.CODE, content="x=1"),
            ]
        )
        result = HybridChunker(max_tokens=512).chunk(doc)
        for c in result:
            assert c.id.startswith("test_chunk_"), f"Bad ID: {c.id}"

    def test_heading_context_with_brackets_in_text(self) -> None:
        """Adversarial text: literal '[' and ']' in content must not be parsed as heading markers."""
        doc = _make_doc(
            [
                DocumentElement(
                    type=ElementType.HEADING,
                    content="Config [key] = value",
                    heading_level=1,
                ),
                DocumentElement(
                    type=ElementType.TEXT,
                    content="See array [1, 2, 3] for details.",
                ),
            ]
        )
        result = HybridChunker(max_tokens=512).chunk(doc)
        # The heading context should reflect the real heading, not any bracket substring.
        for c in result:
            assert "key" not in c.heading_context or "Config" in c.heading_context
        assert any("Config [key] = value" in c.heading_context for c in result if c.heading_context)

    def test_config_chunk_size_flows_to_effective_max(self) -> None:
        """The config chunk_size (passed as max_tokens) must reach _effective_max."""
        chunker = HybridChunker(max_tokens=1024, reserve_ratio=0.2)
        assert chunker._effective_max == 819

    def test_default_chunk_size_is_1024(self) -> None:
        """DEFAULTS chunk_size must be 1024 (not the old 512)."""
        assert DEFAULTS["compile"]["chunk_size"] == 1024

    def test_context_window_adapts_effective_max(self) -> None:
        """When context_window is provided, _effective_max adapts to model capacity."""
        cw = ContextWindow(max_input=8192, max_output=4096, source="test")
        chunker = HybridChunker(max_tokens=1024, context_window=cw, reserve_ratio=0.2)
        available = int(8192 * 0.8)
        expected = max(2048, min(available, 131072))
        assert chunker._effective_max == expected

    def test_context_window_clamped_to_min(self) -> None:
        """Tiny context window is clamped to chunk_size_min."""
        cw = ContextWindow(max_input=512, max_output=256, source="test")
        chunker = HybridChunker(context_window=cw, chunk_size_min=2048)
        assert chunker._effective_max == 2048

    def test_context_window_clamped_to_max(self) -> None:
        """Huge context window is clamped to chunk_size_max."""
        cw = ContextWindow(max_input=2_000_000, max_output=2_000_000, source="test")
        chunker = HybridChunker(context_window=cw, chunk_size_max=131072)
        assert chunker._effective_max == 131072

    def test_no_context_window_uses_max_tokens(self) -> None:
        """Without context_window, falls back to max_tokens path."""
        chunker = HybridChunker(max_tokens=2048, reserve_ratio=0.25)
        assert chunker._effective_max == int(2048 * 0.75)
