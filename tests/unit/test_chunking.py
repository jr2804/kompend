"""Tests for the KNOX HierarchicalChunker."""

from pathlib import Path

from knox.chunking.hierarchical import HierarchicalChunker
from knox.core.models import (
    ChunkNode,
    ChunkStrategy,
    ChunkTree,
    DocumentElement,
    ElementType,
    ParsedDocument,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)
from knox.llm.context_window import ContextSource, ContextWindow


def _make_doc(elements: list[DocumentElement], title: str = "test") -> ParsedDocument:
    return ParsedDocument(
        source_file=Path(f"{title}.pdf"),
        elements=elements,
        title=title,
    )


def _text_elements(paragraphs: list[str]) -> list[DocumentElement]:
    return [DocumentElement(type=ElementType.TEXT, content=p) for p in paragraphs]


def _text_leaves(tree: ChunkTree) -> list[ChunkNode]:
    """Leaf nodes carrying a TEXT chunk."""
    return [n for n in tree.leaves() if n.chunk is not None and ElementType.TEXT in n.chunk.element_types]


def _structured_leaves(tree: ChunkTree) -> list[ChunkNode]:
    """Leaf nodes carrying structured data."""
    return [n for n in tree.leaves() if n.structured is not None]


def _sentence_ends(text: str) -> bool:
    stripped = text.rstrip()
    return not stripped or stripped[-1] in ".!?\n"


class TestHierarchicalChunker:
    """Test suite for HierarchicalChunker."""

    # ── basic structure ─────────────────────────────────────────────────

    def test_empty_document(self) -> None:
        doc = _make_doc([])
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        assert isinstance(tree, ChunkTree)
        assert len(tree.nodes) == 1  # just root
        assert tree.strategy == ChunkStrategy.SINGLE_PASS
        root = tree.nodes[tree.root_id]
        assert root.children_ids == []
        assert root.chunk is None
        assert root.structured is None

    def test_single_text(self) -> None:
        doc = _make_doc([DocumentElement(type=ElementType.TEXT, content="Hello world.")])
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        leaves = _text_leaves(tree)
        assert len(leaves) == 1
        assert "Hello world." in leaves[0].chunk.text  # ty: ignore

    def test_source_file_propagated(self) -> None:
        doc = _make_doc([DocumentElement(type=ElementType.TEXT, content="x")])
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        assert tree.source_file == Path("test.pdf")

    # ── atomic elements ─────────────────────────────────────────────────

    def test_table_is_atomic(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Data", heading_level=1),
                DocumentElement(type=ElementType.TABLE, content=[["A", "B"], ["1", "2"]]),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        tables = _structured_leaves(tree)
        assert len(tables) == 1, "TABLE must be a single atomic leaf"
        assert isinstance(tables[0].structured, StructuredTable)
        assert tables[0].structured.columns == ["A", "B"]
        assert tables[0].structured.rows == [{"A": "1", "B": "2"}]

    def test_code_is_atomic(self) -> None:
        doc = _make_doc([DocumentElement(type=ElementType.CODE, content="print('hello')")])
        tree = HierarchicalChunker(max_tokens=256).chunk(doc)
        code_leaves = [n for n in tree.leaves() if n.chunk is not None and ElementType.CODE in n.chunk.element_types]
        assert len(code_leaves) == 1

    def test_image_is_atomic(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="See figure below."),
                DocumentElement(type=ElementType.IMAGE, content=b"fake_png_data"),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        imgs = [n for n in _structured_leaves(tree) if isinstance(n.structured, StructuredFigure)]
        assert len(imgs) == 1

    def test_formula_is_atomic(self) -> None:
        doc = _make_doc([DocumentElement(type=ElementType.FORMULA, content=r"E = mc^2")])
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        eqs = [n for n in _structured_leaves(tree) if isinstance(n.structured, StructuredEquation)]
        assert len(eqs) == 1
        assert eqs[0].structured.latex == r"E = mc^2"  # ty: ignore

    # ── heading hierarchy ───────────────────────────────────────────────

    def test_heading_creates_internal_node(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Results", heading_level=1),
                DocumentElement(type=ElementType.TEXT, content="Significant findings."),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        root = tree.nodes[tree.root_id]
        assert len(root.children_ids) == 1
        heading_node = tree.nodes[root.children_ids[0]]
        assert heading_node.chunk is None
        assert heading_node.structured is None
        assert len(heading_node.children_ids) == 1

    def test_heading_path_on_leaves(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Chapter 1", heading_level=1),
                DocumentElement(type=ElementType.HEADING, content="Section 1.1", heading_level=2),
                DocumentElement(type=ElementType.TEXT, content="Deep content."),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        leaves = _text_leaves(tree)
        assert len(leaves) == 1
        assert leaves[0].heading_path == ("Chapter 1", "Section 1.1")
        assert leaves[0].chunk.heading_context == "Chapter 1 / Section 1.1"  # ty: ignore

    def test_nested_heading_hierarchy(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="A", heading_level=1),
                DocumentElement(type=ElementType.TEXT, content="Under A"),
                DocumentElement(type=ElementType.HEADING, content="B", heading_level=1),
                DocumentElement(type=ElementType.HEADING, content="B1", heading_level=2),
                DocumentElement(type=ElementType.TEXT, content="Under B1"),
                DocumentElement(type=ElementType.HEADING, content="C", heading_level=1),
                DocumentElement(type=ElementType.TEXT, content="Under C"),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        root = tree.nodes[tree.root_id]
        assert len(root.children_ids) == 3
        a_node = tree.nodes[root.children_ids[0]]
        b_node = tree.nodes[root.children_ids[1]]
        c_node = tree.nodes[root.children_ids[2]]
        assert len(a_node.children_ids) == 1
        assert len(b_node.children_ids) == 1
        assert len(c_node.children_ids) == 1
        b1 = tree.nodes[b_node.children_ids[0]]
        assert len(b1.children_ids) == 1

    def test_heading_closes_deeper_levels(self) -> None:
        """A heading at level 1 closes open level-2+ headings."""
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="A", heading_level=1),
                DocumentElement(type=ElementType.HEADING, content="A1", heading_level=2),
                DocumentElement(type=ElementType.TEXT, content="under A1"),
                DocumentElement(type=ElementType.HEADING, content="B", heading_level=1),
                DocumentElement(type=ElementType.TEXT, content="under B"),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        root = tree.nodes[tree.root_id]
        # A and B are both depth-1 children of root
        assert len(root.children_ids) == 2
        b_node = tree.nodes[root.children_ids[1]]
        assert b_node.depth == 1
        # B's text leaf is at depth 2, not depth 3
        b_text = tree.nodes[b_node.children_ids[0]]
        assert b_text.depth == 2

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
        tree = HierarchicalChunker(max_tokens=256).chunk(doc)
        assert len(_text_leaves(tree)) == 2
        assert len(_structured_leaves(tree)) == 1

    def test_metadata_elements_skipped(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.METADATA, content="key=val"),
                DocumentElement(type=ElementType.TEXT, content="Real content."),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        leaves = _text_leaves(tree)
        assert len(leaves) == 1
        assert "Real content." in leaves[0].chunk.text  # ty: ignore

    # ── strategy selection ──────────────────────────────────────────────

    def test_strategy_single_pass(self) -> None:
        doc = _make_doc([DocumentElement(type=ElementType.TEXT, content="Small doc.")])
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        assert tree.strategy == ChunkStrategy.SINGLE_PASS

    def test_strategy_two_level(self) -> None:
        """Doc producing <= 10 chunks → TWO_LEVEL."""
        # effective_max=100 (reserve=0). 500 tokens → ceil(500/100)=5 chunks <= 10
        text = "word " * 500  # ~625 tokens → 7 chunks
        doc = _make_doc([DocumentElement(type=ElementType.TEXT, content=text)])
        tree = HierarchicalChunker(max_tokens=100, reserve_tokens=0).chunk(doc)
        assert tree.strategy == ChunkStrategy.TWO_LEVEL

    def test_strategy_multi_resolution(self) -> None:
        """Doc producing > 10 chunks → MULTI_RESOLUTION."""
        # effective_max=100. 1500 tokens → ceil(1500/100)=15 chunks > 10
        text = "word " * 1200  # ~1500 tokens → 15 chunks
        doc = _make_doc([DocumentElement(type=ElementType.TEXT, content=text)])
        tree = HierarchicalChunker(max_tokens=100, reserve_tokens=0).chunk(doc)
        assert tree.strategy == ChunkStrategy.MULTI_RESOLUTION

    # ── token computation ───────────────────────────────────────────────

    def test_subtree_token_computation(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="S", heading_level=1),
                DocumentElement(type=ElementType.TEXT, content="First part here."),
                DocumentElement(type=ElementType.TEXT, content="Second part here."),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        root = tree.nodes[tree.root_id]
        section = tree.nodes[root.children_ids[0]]
        expected = sum(tree.nodes[c].token_count for c in section.children_ids)
        assert section.token_count == expected
        assert root.token_count == section.token_count

    # ── effective_max / context window ──────────────────────────────────

    def test_reserve_tokens_effective(self) -> None:
        chunker = HierarchicalChunker(max_tokens=1000, reserve_tokens=200)
        assert chunker._effective_max == 800
        chunker2 = HierarchicalChunker(max_tokens=1000, reserve_tokens=0)
        assert chunker2._effective_max == 1000

    def test_chunk_ids_assigned(self) -> None:
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="First."),
                DocumentElement(type=ElementType.TEXT, content="Second."),
                DocumentElement(type=ElementType.CODE, content="x=1"),
            ]
        )
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        for node in tree.leaves():
            assert node.id.startswith("test_"), f"Bad ID: {node.id}"

    def test_heading_context_with_brackets_in_text(self) -> None:
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
        tree = HierarchicalChunker(max_tokens=512).chunk(doc)
        leaves = _text_leaves(tree)
        assert len(leaves) == 1
        assert "Config [key] = value" in leaves[0].heading_path

    def test_config_chunk_size_flows_to_effective_max(self) -> None:
        chunker = HierarchicalChunker(max_tokens=1024, reserve_tokens=205)
        assert chunker._effective_max == 819

    def test_context_window_adapts_effective_max(self) -> None:
        cw = ContextWindow(max_input=8192, max_output=4096, source=ContextSource.FALLBACK)
        chunker = HierarchicalChunker(max_tokens=1024, context_window=cw, reserve_tokens=1639)
        assert chunker._effective_max == 8192 - 1639

    def test_context_window_tiny_model(self) -> None:
        cw = ContextWindow(max_input=512, max_output=256, source=ContextSource.FALLBACK)
        chunker = HierarchicalChunker(context_window=cw, chunk_size_min=2048)
        assert chunker._effective_max == 2048

    def test_context_window_huge_model(self) -> None:
        cw = ContextWindow(max_input=2_000_000, max_output=2_000_000, source=ContextSource.FALLBACK)
        chunker = HierarchicalChunker(context_window=cw, chunk_size_max=131072)
        assert chunker._effective_max == 131072

    def test_no_context_window_uses_max_tokens(self) -> None:
        chunker = HierarchicalChunker(max_tokens=2048, reserve_tokens=512)
        assert chunker._effective_max == 2048 - 512

    # ── text splitting ──────────────────────────────────────────────────

    def test_long_text_is_chunked(self) -> None:
        doc = _make_doc([DocumentElement(type=ElementType.TEXT, content="word " * 500)])
        tree = HierarchicalChunker(max_tokens=100).chunk(doc)
        assert len(_text_leaves(tree)) > 1, "Long text should produce multiple leaves"

    def test_splits_at_paragraph_boundary(self) -> None:
        paragraphs = [
            "First paragraph has two clear sentences. Here is the second one.",
            "Second paragraph is longer and should stay together if possible. It has more words that we can count.",
        ]
        doc = _make_doc(_text_elements(paragraphs))
        tree = HierarchicalChunker(max_tokens=24, reserve_tokens=0).chunk(doc)
        leaves = _text_leaves(tree)
        assert len(leaves) >= 1
        assert any(_sentence_ends(n.chunk.text) for n in leaves if n.chunk is not None)  # type: ignore[union-attr]

    def test_no_mid_sentence_cut_when_chunk_fits(self) -> None:
        sentences = [
            "Short sentence one.",
            "Short sentence two.",
            "Short sentence three is the longest of them all.",
        ]
        doc = _make_doc(_text_elements(sentences))
        tree = HierarchicalChunker(max_tokens=12, reserve_tokens=0).chunk(doc)
        leaves = _text_leaves(tree)
        for n in leaves:
            assert _sentence_ends(n.chunk.text)  # ty: ignore

    def test_long_paragraph_split_at_sentence_boundary(self) -> None:
        text = "Sentence one is moderately long. " * 6
        doc = _make_doc([DocumentElement(type=ElementType.TEXT, content=text)])
        tree = HierarchicalChunker(max_tokens=16, reserve_tokens=0).chunk(doc)
        leaves = _text_leaves(tree)
        for n in leaves:
            assert _sentence_ends(n.chunk.text)  # ty: ignore


class TestContextWindowClamp:
    """Chunk size must be clamped to [chunk_size_min, chunk_size_max] (knox-7sp)."""

    def test_large_context_clamped_to_max(self) -> None:
        cw = ContextWindow(max_input=1_000_000, max_output=8192, source=ContextSource.MODELS_DEV)
        chunker = HierarchicalChunker(context_window=cw, chunk_size_min=2048, chunk_size_max=131072)
        assert chunker._effective_max == 131072

    def test_small_context_respects_min(self) -> None:
        cw = ContextWindow(max_input=8192, max_output=4096, source=ContextSource.MODELS_DEV)
        # Default reserve_tokens=8192 zeros out an 8K window → floored to chunk_size_min.
        chunker = HierarchicalChunker(context_window=cw, chunk_size_min=2048, chunk_size_max=131072)
        assert chunker._effective_max == 2048

    def test_mid_range_context_unclamped(self) -> None:
        cw = ContextWindow(max_input=131072, max_output=8192, source=ContextSource.MODELS_DEV)
        # Default reserve_tokens=8192 → effective=122880, sits between min and max rails.
        chunker = HierarchicalChunker(context_window=cw, chunk_size_min=2048, chunk_size_max=131072)
        assert chunker._effective_max == 131072 - 8192

    def test_fallback_uses_max_tokens_unclamped(self) -> None:
        chunker = HierarchicalChunker(max_tokens=1024, reserve_tokens=205)
        assert chunker._effective_max == 819
