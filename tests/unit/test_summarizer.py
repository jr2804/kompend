"""Tests for NodeSummarizer (summarizer.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from knox.chunking.summarizer import NodeSummarizer
from knox.core.models import (
    Chunk,
    ChunkNode,
    ChunkStrategy,
    ChunkTree,
    ElementType,
    StructuredTable,
)


def _make_chunk(cid: str, text: str, source: str = "doc.md") -> Chunk:
    """Build a Chunk with required fields."""
    return Chunk(
        id=cid,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[],
        source_file=Path(source),
        token_count=max(1, len(text) // 4),
    )


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_tree() -> ChunkTree:
    """Build a 5-node tree: root → section (leaf-a, leaf-b) + section-c."""
    root = ChunkNode(id="doc_root", depth=0)
    section1 = ChunkNode(
        id="doc_h1",
        parent_id="doc_root",
        depth=1,
        heading_path=("Introduction",),
    )
    leaf_a = ChunkNode(
        id="doc_t1",
        parent_id="doc_h1",
        depth=2,
        heading_path=("Introduction",),
        chunk=_make_chunk("doc_t1", "The SNR threshold is 20 dB for intelligibility."),
        token_count=10,
    )
    leaf_b = ChunkNode(
        id="doc_t2",
        parent_id="doc_h1",
        depth=2,
        heading_path=("Introduction",),
        chunk=_make_chunk("doc_t2", "Results show improvement at higher levels."),
        token_count=8,
    )
    section2 = ChunkNode(
        id="doc_h2",
        parent_id="doc_root",
        depth=1,
        heading_path=("Methodology",),
    )
    leaf_c = ChunkNode(
        id="doc_t3",
        parent_id="doc_h2",
        depth=2,
        heading_path=("Methodology",),
        chunk=_make_chunk("doc_t3", "We used a standard test setup with calibration."),
        token_count=10,
    )

    root.children_ids = ["doc_h1", "doc_h2"]
    section1.children_ids = ["doc_t1", "doc_t2"]
    section2.children_ids = ["doc_t3"]

    nodes = {n.id: n for n in [root, section1, leaf_a, leaf_b, section2, leaf_c]}
    return ChunkTree(
        root_id="doc_root",
        nodes=nodes,
        source_file=Path("doc.md"),
        strategy=ChunkStrategy.TWO_LEVEL,
    )


@pytest.fixture
def single_pass_tree() -> ChunkTree:
    """A SINGLE_PASS tree — should be skipped by summarizer."""
    root = ChunkNode(id="root", depth=0)
    leaf = ChunkNode(
        id="leaf",
        parent_id="root",
        depth=1,
        chunk=_make_chunk("leaf", "text"),
        token_count=1,
    )
    root.children_ids = ["leaf"]
    return ChunkTree(
        root_id="root",
        nodes={"root": root, "leaf": leaf},
        source_file=Path("d.md"),
        strategy=ChunkStrategy.SINGLE_PASS,
    )


@pytest.fixture
def single_child_tree() -> ChunkTree:
    """Tree with single-child internal nodes."""
    root = ChunkNode(id="r", depth=0)
    wrapper = ChunkNode(id="w", parent_id="r", depth=1, heading_path=("Wrapper",))
    leaf = ChunkNode(
        id="l",
        parent_id="w",
        depth=2,
        heading_path=("Wrapper",),
        chunk=_make_chunk("l", "content"),
        token_count=2,
    )
    root.children_ids = ["w"]
    wrapper.children_ids = ["l"]
    return ChunkTree(
        root_id="r",
        nodes={"r": root, "w": wrapper, "l": leaf},
        source_file=Path("d.md"),
        strategy=ChunkStrategy.TWO_LEVEL,
    )


# ── tests ────────────────────────────────────────────────────────────────


class TestSummarizerSkip:
    def test_single_pass_skipped(self, single_pass_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(llm_client=None)
        result = summarizer.summarize(single_pass_tree)
        root = result.nodes["root"]
        assert root.summary == ""

    def test_single_child_skipped_by_default(self, single_child_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(llm_client=None, skip_single_child=True)
        summarizer.summarize(single_child_tree)
        wrapper = single_child_tree.nodes["w"]
        # Single-child wrapper gets the child's content propagated
        assert wrapper.summary == ""

    def test_single_child_summarized_when_disabled(self, single_child_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(llm_client=None, skip_single_child=False)
        summarizer.summarize(single_child_tree)
        wrapper = single_child_tree.nodes["w"]
        # Extractive fallback: first sentence from single child
        assert wrapper.summary != ""


class TestSummarizerExtractiveFallback:
    def test_internal_nodes_get_summaries(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(llm_client=None)
        summarizer.summarize(sample_tree)

        section1 = sample_tree.nodes["doc_h1"]
        assert section1.summary != ""
        assert "SNR" in section1.summary or "threshold" in section1.summary.lower()

    def test_root_gets_summary(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(llm_client=None)
        summarizer.summarize(sample_tree)

        root = sample_tree.nodes["doc_root"]
        assert root.summary != ""

    def test_leaf_nodes_unchanged(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(llm_client=None)
        summarizer.summarize(sample_tree)

        leaf = sample_tree.nodes["doc_t1"]
        assert leaf.summary == ""
        assert leaf.chunk is not None

    def test_extractive_max_sentences(self, sample_tree: ChunkTree) -> None:
        """Extractive fallback takes at most 5 sentences."""
        summarizer = NodeSummarizer(llm_client=None)
        summarizer.summarize(sample_tree)

        root = sample_tree.nodes["doc_root"]
        # Root summarizes both sections — each contributes a sentence
        assert len(root.summary.split(". ")) <= 6


class TestSummarizerLLM:
    def test_llm_called_for_internal_nodes(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.default_model = "test-model"
        mock_llm.complete.return_value = MagicMock(content="LLM summary.")

        summarizer = NodeSummarizer(llm_client=mock_llm)
        summarizer.summarize(sample_tree)

        section1 = sample_tree.nodes["doc_h1"]
        assert section1.summary == "LLM summary."
        assert mock_llm.complete.called

    def test_llm_template_rendered(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.default_model = "test-model"
        mock_llm.complete.return_value = MagicMock(content="Summary.")

        summarizer = NodeSummarizer(llm_client=mock_llm)
        summarizer.summarize(sample_tree)

        # Check the LLM was called with a rendered prompt
        call_args = mock_llm.complete.call_args
        messages = call_args[0][0]
        prompt = messages[0]["content"]
        assert "SNR" in prompt or "threshold" in prompt.lower()

    def test_returns_same_tree_object(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(llm_client=None)
        result = summarizer.summarize(sample_tree)
        assert result is sample_tree


class TestSummarizerCache:
    def test_cache_hit_skips_llm(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.default_model = "test-model"
        mock_llm.complete.return_value = MagicMock(content="Fresh summary.")

        mock_cache = MagicMock()
        mock_cache.get.return_value = "Cached summary."

        summarizer = NodeSummarizer(llm_client=mock_llm, cache_store=mock_cache)
        summarizer.summarize(sample_tree)

        mock_llm.complete.assert_not_called()

    def test_cache_miss_calls_llm(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.default_model = "test-model"
        mock_llm.complete.return_value = MagicMock(content="Fresh summary.")

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        summarizer = NodeSummarizer(llm_client=mock_llm, cache_store=mock_cache)
        summarizer.summarize(sample_tree)

        assert mock_llm.complete.called
        assert mock_cache.set.called

    def test_cache_includes_model_in_deps(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.default_model = "my-model"
        mock_llm.complete.return_value = MagicMock(content="Summary.")

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        summarizer = NodeSummarizer(llm_client=mock_llm, cache_store=mock_cache)
        summarizer.summarize(sample_tree)

        for call in mock_cache.set.call_args_list:
            deps = call[0][1]
            assert deps["model"] == "my-model"


class TestSummarizerStructuredLeaf:
    def test_structured_leaf_content(self) -> None:
        """Test that structured leaves contribute content to summaries."""
        root = ChunkNode(id="r", depth=0)
        section = ChunkNode(id="s", parent_id="r", depth=1, heading_path=("Data",))
        table_leaf = ChunkNode(
            id="t",
            parent_id="s",
            depth=2,
            heading_path=("Data",),
            structured=StructuredTable(
                columns=["A", "B"],
                rows=[{"A": "1", "B": "2"}],
                caption="Results table",
            ),
            chunk=_make_chunk("t", "Table: Results table\nA | B\n1 | 2"),
            token_count=10,
        )
        root.children_ids = ["s"]
        section.children_ids = ["t"]

        tree = ChunkTree(
            root_id="r",
            nodes={"r": root, "s": section, "t": table_leaf},
            source_file=Path("d.md"),
            strategy=ChunkStrategy.TWO_LEVEL,
        )

        summarizer = NodeSummarizer(llm_client=None, skip_single_child=False)
        summarizer.summarize(tree)

        # Section should have a summary derived from the table content
        section_node = tree.nodes["s"]
        assert section_node.summary != ""
