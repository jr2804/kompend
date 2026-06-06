"""Tests for the KNOX HybridChunker."""

from pathlib import Path

from knox.chunking.hybrid import HybridChunker
from knox.core.models import DocumentElement, ElementType, ParsedDocument


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
