"""Smoke test for the Tier 1 → Tier 2 pipeline integration.

Creates a ParsedDocument via test fixtures and runs it through HierarchicalChunker
to verify that the chunk tree has the correct structure.
"""

from __future__ import annotations

from pathlib import Path

from knox.chunking.hierarchical import HierarchicalChunker
from knox.core.models import ChunkTree, DocumentElement, ElementType
from tests.fixtures.documents import make_test_doc


class TestPipelineSmoke:
    """Integration test: ParsedDocument → HierarchicalChunker → ChunkTree."""

    def test_full_document_chunking(self) -> None:
        doc = make_test_doc(num_paragraphs=3, include_tables=True, include_code=True, include_images=True)

        types = {e.type for e in doc.elements}
        assert ElementType.HEADING in types
        assert ElementType.TEXT in types
        assert ElementType.TABLE in types
        assert ElementType.CODE in types
        assert ElementType.IMAGE in types

        chunker = HierarchicalChunker(max_tokens=512)
        tree = chunker.chunk(doc)
        chunks = _leaf_chunks(tree)

        assert len(chunks) > 0, "Should produce at least one chunk"

        for c in chunks:
            assert c.id, f"Chunk id must be non-empty: {c!r}"
            assert c.text, f"Chunk text must be non-empty: {c.id}"
            assert len(c.element_types) > 0, f"Chunk {c.id} has no element types"
            assert len(c.source_elements) > 0, f"Chunk {c.id} has no source elements"
            for idx in c.source_elements:
                assert 0 <= idx < len(doc.elements), f"Chunk {c.id} references invalid element index {idx}"

        for c in chunks:
            for atype in (ElementType.TABLE, ElementType.CODE, ElementType.IMAGE):
                if atype in c.element_types:
                    assert len(c.source_elements) == 1, f"Atomic {atype.name} chunk {c.id} has {len(c.source_elements)} source elements, expected 1"

        covered: set[int] = set()
        for c in chunks:
            covered.update(c.source_elements)
        expected = {i for i, e in enumerate(doc.elements) if e.type not in (ElementType.HEADING, ElementType.METADATA)}
        assert covered == expected, f"Not all elements covered. Missing: {expected - covered}"

        has_context = any(c.heading_context for c in chunks)
        assert has_context, "At least one chunk should have heading_context"

    def test_minimal_document(self) -> None:
        doc = make_test_doc(num_paragraphs=2, include_tables=False, include_code=False, include_images=False)
        tree = HierarchicalChunker(max_tokens=1024).chunk(doc)
        chunks = _leaf_chunks(tree)
        assert len(chunks) > 0
        for c in chunks:
            assert ElementType.TABLE not in c.element_types
            assert ElementType.CODE not in c.element_types

    def test_empty_elements(self) -> None:
        doc = make_test_doc(num_paragraphs=0, include_tables=False, include_code=False, include_images=False)
        doc.elements = []
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        assert _leaf_chunks(tree) == []

    def test_single_atomic_element(self) -> None:
        doc = make_test_doc(num_paragraphs=0, include_tables=False, include_code=False, include_images=False)
        doc.elements = [
            DocumentElement(type=ElementType.TABLE, content=[["A", "B"], ["1", "2"]]),
        ]
        doc.source_file = Path("atomic.pdf")
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        chunks = _leaf_chunks(tree)
        assert len(chunks) == 1
        assert ElementType.TABLE in chunks[0].element_types
        assert len(chunks[0].source_elements) == 1

    def test_chunk_id_format(self) -> None:
        """Chunk IDs should use the source filename stem."""
        doc = make_test_doc(num_paragraphs=1, include_tables=False, include_code=False, include_images=False)
        doc.source_file = Path("my_report.pdf")
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        for n in tree.leaves():
            if n.chunk is None:
                continue
            assert n.id.startswith("my_report_"), f"Unexpected id format: {n.id}"


def _leaf_chunks(tree: ChunkTree) -> list:
    """Extract Chunk objects from leaf nodes."""
    return [n.chunk for n in tree.leaves() if n.chunk is not None]
