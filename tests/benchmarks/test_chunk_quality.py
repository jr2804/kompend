"""Chunk quality benchmark.

Takes a mock ParsedDocument (from test fixtures), runs HierarchicalChunker, and measures:
  - Token efficiency (total chunk tokens / raw document tokens)
  - Atomic preservation (tables/code blocks atomically preserved)
  - Chunk count distribution

Reports metrics, doesn't fail on specific values (except basic sanity).
"""

from __future__ import annotations

import time

import pytest

from knox.chunking import HierarchicalChunker
from knox.core.models import Chunk, ElementType, ParsedDocument
from tests.fixtures.documents import make_test_doc

pytestmark = pytest.mark.benchmark


class TestChunkQualityBenchmark:
    """Quality benchmarks for HierarchicalChunker."""

    @pytest.fixture(scope="class")
    def large_doc(self) -> ParsedDocument:
        """A large, realistic test document."""
        return make_test_doc(
            num_paragraphs=8,
            include_tables=True,
            include_code=True,
            include_images=True,
        )

    @pytest.fixture(scope="class")
    def text_only_doc(self) -> ParsedDocument:
        """A document without atomic elements."""
        return make_test_doc(
            num_paragraphs=5,
            include_tables=False,
            include_code=False,
            include_images=False,
        )

    # ── Chunking speed ─────────────────────────────────────────────────

    def test_benchmark_chunking_speed(self, large_doc: ParsedDocument) -> None:
        """Measure chunking speed across multiple configurations."""
        configs = [
            ("max_tokens=256", 256),
            ("max_tokens=512", 512),
            ("max_tokens=1024", 1024),
        ]
        print("\n")
        for label, max_tok in configs:
            durations: list[float] = []
            for _ in range(5):
                t0 = time.perf_counter()
                _leaf_chunks(large_doc, max_tok)
                durations.append(time.perf_counter() - t0)
            avg = sum(durations) / len(durations)
            print(f"  ⏱  HierarchicalChunker({label}) — avg: {avg * 1000:.1f}ms over 5 runs")

        # Just verify it doesn't crash
        result = _leaf_chunks(large_doc, 512)
        assert len(result) > 0

    # ── Token efficiency ───────────────────────────────────────────────

    def test_token_efficiency(self, large_doc: ParsedDocument) -> None:
        """Measure total chunk tokens vs raw document tokens."""
        # Count raw tokens in the document
        raw_text = "\n\n".join(
            str(el.content)
            if isinstance(el.content, str)
            else (
                "\n".join(" | ".join(str(c) for c in row) for row in el.content)
                if isinstance(el.content, (list, tuple))
                else f"[binary: {len(el.content)} bytes]"
            )
            for el in large_doc.elements
        )
        raw_tokens = self._approx_tokens(raw_text)

        chunks = _leaf_chunks(large_doc, 512)
        chunk_tokens = sum(self._approx_tokens(c.text) for c in chunks)

        efficiency = chunk_tokens / raw_tokens if raw_tokens > 0 else 1.0
        overhead = (chunk_tokens - raw_tokens) / raw_tokens * 100 if raw_tokens > 0 else 0.0

        print("\n  📊 Token efficiency metrics:")
        print(f"     Raw document tokens (approx): {raw_tokens}")
        print(f"     Total chunk tokens (approx):  {chunk_tokens}")
        print(f"     Efficiency ratio:             {efficiency:.2%}")
        print(f"     Overhead:                     {overhead:.1f}%")
        print(f"     Number of chunks:             {len(chunks)}")

        # Sanity: efficiency should be reasonable (some overhead expected)
        assert efficiency >= 0.5, f"Token efficiency suspiciously low: {efficiency:.2%}"
        assert efficiency <= 3.0, f"Token efficiency suspiciously high: {efficiency:.2%} (may indicate duplication)"

    def test_token_efficiency_text_only(self, text_only_doc: ParsedDocument) -> None:
        """Token efficiency for text-only documents."""
        raw_text = "\n\n".join(str(el.content) for el in text_only_doc.elements if isinstance(el.content, str))
        raw_tokens = self._approx_tokens(raw_text)

        chunks = _leaf_chunks(text_only_doc, 512)
        chunk_tokens = sum(self._approx_tokens(c.text) for c in chunks)
        efficiency = chunk_tokens / raw_tokens if raw_tokens > 0 else 1.0

        print("\n  📊 Token efficiency (text-only):")
        print(f"     Raw tokens:   {raw_tokens}")
        print(f"     Chunk tokens: {chunk_tokens}")
        print(f"     Ratio:        {efficiency:.2%}")
        print(f"     Chunks:       {len(chunks)}")

    # ── Atomic preservation ────────────────────────────────────────────

    def test_table_atomic_preservation(self, large_doc: ParsedDocument) -> None:
        """Verify tables are never split across chunks."""
        chunks = _leaf_chunks(large_doc, 256)
        table_chunks = [c for c in chunks if ElementType.TABLE in c.element_types]
        print(f"\n  🏛  Table chunks: {len(table_chunks)}")
        for c in table_chunks:
            assert len(c.source_elements) == 1, f"Table chunk {c.id} spans {len(c.source_elements)} elements, expected 1"
            print(f"     {c.id}: text length={len(c.text)}, source=[{c.source_elements}]")

    def test_code_atomic_preservation(self, large_doc: ParsedDocument) -> None:
        """Verify code blocks are never split across chunks."""
        chunks = _leaf_chunks(large_doc, 256)
        code_chunks = [c for c in chunks if ElementType.CODE in c.element_types]
        print(f"\n  💻 Code chunks: {len(code_chunks)}")
        for c in code_chunks:
            assert len(c.source_elements) == 1, f"Code chunk {c.id} spans {len(c.source_elements)} elements, expected 1"

    def test_image_atomic_preservation(self, large_doc: ParsedDocument) -> None:
        """Verify images are never split across chunks."""
        chunks = _leaf_chunks(large_doc, 256)
        img_chunks = [c for c in chunks if ElementType.IMAGE in c.element_types]
        print(f"\n  🖼  Image chunks: {len(img_chunks)}")
        for c in img_chunks:
            assert len(c.source_elements) == 1, f"Image chunk {c.id} spans {len(c.source_elements)} elements, expected 1"

    # ── Chunk count distribution ───────────────────────────────────────

    def test_chunk_count_distribution(self, large_doc: ParsedDocument) -> None:
        """Measure chunk counts at different max_tokens settings."""
        results: list[tuple[int, int, int]] = []
        for max_tok in [128, 256, 512, 1024]:
            chunks = _leaf_chunks(large_doc, max_tok)
            chunk_sizes = [len(c.text) for c in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            results.append((max_tok, len(chunks), int(avg_size)))

        print("\n  📊 Chunk count distribution:")
        print(f"     {'max_tokens':>12} | {'chunks':>7} | {'avg_size':>9}")
        print(f"     {'-' * 12}-+-{'-' * 7}-+-{'-' * 9}")
        for max_tok, count, avg_sz in results:
            print(f"     {max_tok:>12} | {count:>7} | {avg_sz:>9}")

        # Sanity: more max_tokens should mean fewer chunks
        counts = [c for _, c, _ in results]
        assert counts == sorted(counts, reverse=True), f"Chunk counts should decrease with increasing max_tokens: {counts}"

    # ── Heading context ────────────────────────────────────────────────

    def test_heading_context_quality(self, large_doc: ParsedDocument) -> None:
        """Measure how often heading context is populated."""
        chunks = _leaf_chunks(large_doc, 256)
        with_context = sum(1 for c in chunks if c.heading_context)
        ratio = with_context / len(chunks) if chunks else 0.0

        print("\n  📑 Heading context coverage:")
        print(f"     Chunks with context: {with_context}/{len(chunks)} ({ratio:.1%})")
        print(f"     Sample contexts: {[c.heading_context for c in chunks[:5] if c.heading_context]}")

        # At least some chunks should have heading context
        assert with_context > 0, "No chunks have heading_context"

    # ── Fallback mode ──────────────────────────────────────────────────

    def test_fallback_chunking(self, large_doc: ParsedDocument) -> None:
        """Verify chunking still works — chonkie is a hard dep."""
        chunks = _leaf_chunks(large_doc, 512)
        assert len(chunks) > 0, "Chunking should produce chunks"
        print(f"\n  🔧 Fallback chunking produced {len(chunks)} chunks")

    def _approx_tokens(self, text: str) -> int:
        """Rough token count (4 chars per token approximation)."""
        return len(text) // 4


def _leaf_chunks(doc: ParsedDocument, max_tokens: int) -> list[Chunk]:
    """Chunk a document and extract leaf-node Chunk objects."""
    tree = HierarchicalChunker(max_tokens=max_tokens).chunk(doc)
    return [n.chunk for n in tree.leaves() if n.chunk]
