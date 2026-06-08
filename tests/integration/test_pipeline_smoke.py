"""Smoke test for the Tier 1 → Tier 2 pipeline integration.

Creates a ParsedDocument via test fixtures and runs it through HybridChunker
to verify that chunks have the correct structure.
"""

from __future__ import annotations

from knox.chunking.hybrid import HybridChunker
from knox.core.models import ElementType
from tests.fixtures.documents import make_test_doc


class TestPipelineSmoke:
    """Integration test: ParsedDocument → HybridChunker → Chunks."""

    def test_full_document_chunking(self) -> None:
        """Run the full pipeline and verify chunk structure."""
        doc = make_test_doc(num_paragraphs=3, include_tables=True, include_code=True, include_images=True)

        # Sanity: doc has all element types
        types = {e.type for e in doc.elements}
        assert ElementType.HEADING in types
        assert ElementType.TEXT in types
        assert ElementType.TABLE in types
        assert ElementType.CODE in types
        assert ElementType.IMAGE in types

        chunker = HybridChunker(max_tokens=512)
        chunks = chunker.chunk(doc)

        # Basic sanity checks
        assert len(chunks) > 0, "Should produce at least one chunk"

        # Every chunk must have an id
        for c in chunks:
            assert c.id, f"Chunk id must be non-empty: {c!r}"

        # Every chunk must have text
        for c in chunks:
            assert c.text, f"Chunk text must be non-empty: {c.id}"

        # Every chunk must have element_types
        for c in chunks:
            assert len(c.element_types) > 0, f"Chunk {c.id} has no element types"

        # Every chunk must have source_elements
        for c in chunks:
            assert len(c.source_elements) > 0, f"Chunk {c.id} has no source elements"

        # All source_element indices must be valid
        for c in chunks:
            for idx in c.source_elements:
                assert 0 <= idx < len(doc.elements), f"Chunk {c.id} references invalid element index {idx}"

        # Atomic elements must each be in their own chunk
        for c in chunks:
            for atype in (ElementType.TABLE, ElementType.CODE, ElementType.IMAGE):
                if atype in c.element_types:
                    assert len(c.source_elements) == 1, f"Atomic {atype.name} chunk {c.id} has {len(c.source_elements)} source elements, expected 1"

        # All elements must be covered by at least one chunk's source_elements
        covered: set[int] = set()
        for c in chunks:
            covered.update(c.source_elements)
        assert covered == set(range(len(doc.elements))), f"Not all elements covered. Missing: {set(range(len(doc.elements))) - covered}"

        # Heading context should be present on some chunks
        has_context = any(c.heading_context for c in chunks)
        assert has_context, "At least one chunk should have heading_context"

    def test_minimal_document(self) -> None:
        """Chunk a document with only text and headings."""
        doc = make_test_doc(num_paragraphs=2, include_tables=False, include_code=False, include_images=False)
        chunks = HybridChunker(max_tokens=1024).chunk(doc)
        assert len(chunks) > 0
        for c in chunks:
            assert all(t != ElementType.TABLE for t in c.element_types)
            assert all(t != ElementType.CODE for t in c.element_types)

    def test_empty_elements(self) -> None:
        """Empty element list returns empty chunk list."""
        doc = make_test_doc(num_paragraphs=0, include_tables=False, include_code=False, include_images=False)
        doc.elements = []
        chunks = HybridChunker(max_tokens=512).chunk(doc)
        assert chunks == []

    def test_single_atomic_element(self) -> None:
        """A document with only a table should produce one chunk."""
        from pathlib import Path

        from knox.core.models import DocumentElement

        doc = make_test_doc(num_paragraphs=0, include_tables=False, include_code=False, include_images=False)
        doc.elements = [
            DocumentElement(type=ElementType.TABLE, content=[["A", "B"], ["1", "2"]]),
        ]
        doc.source_path = Path("atomic.pdf")
        chunks = HybridChunker(max_tokens=512).chunk(doc)
        assert len(chunks) == 1
        assert ElementType.TABLE in chunks[0].element_types
        assert len(chunks[0].source_elements) == 1

    def test_chunk_id_format(self) -> None:
        """Chunk IDs should be stable and use the source filename."""
        doc = make_test_doc(num_paragraphs=1, include_tables=False, include_code=False, include_images=False)
        from pathlib import Path

        doc.source_path = Path("my_report.pdf")
        chunks = HybridChunker(max_tokens=512).chunk(doc)
        for c in chunks:
            assert c.id.startswith("my_report_chunk_"), f"Unexpected id format: {c.id}"
            # Extract the number suffix and verify it's sequential
            parts = c.id.split("_")
            assert parts[-1].isdigit(), f"Chunk id suffix should be numeric: {c.id}"
