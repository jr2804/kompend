"""Tests for the ContextualEnricher (contextual retrieval chunking)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kompend.chunking.contextual import ContextualEnricher
from kompend.core.models import Chunk, ChunkNode, ChunkTree, DocumentElement, ElementType, ParsedDocument


class TestCollectLeaves:
    """_collect_leaves finds all leaf nodes in the tree."""

    def test_single_leaf(self) -> None:
        leaf = _make_leaf("leaf1")
        tree = _make_tree(leaf)
        leaves = ContextualEnricher._collect_leaves(tree)
        assert [n.id for n in leaves] == ["leaf1"]

    def test_multiple_leaves(self) -> None:
        tree = _make_tree(_make_leaf("a"), _make_leaf("b"), _make_leaf("c"))
        leaves = ContextualEnricher._collect_leaves(tree)
        assert {n.id for n in leaves} == {"a", "b", "c"}

    def test_internal_node_not_included(self) -> None:
        leaf = _make_leaf("leaf1")
        internal = ChunkNode(id="internal", children_ids=["leaf1"])
        root = ChunkNode(id="root", children_ids=["internal"])
        tree = ChunkTree(
            root_id="root",
            nodes={"root": root, "internal": internal, "leaf1": leaf},
        )
        leaves = ContextualEnricher._collect_leaves(tree)
        assert [n.id for n in leaves] == ["leaf1"]

    def test_empty_tree_root_only(self) -> None:
        root = ChunkNode(id="root")
        tree = ChunkTree(root_id="root", nodes={"root": root})
        leaves = ContextualEnricher._collect_leaves(tree)
        assert leaves == [root]


class TestSummarizeDocument:
    """_summarize_document extracts a heading-based summary."""

    def test_uses_headings_when_present(self) -> None:
        doc = ParsedDocument(
            source_file=Path("doc.md"),
            elements=[
                DocumentElement(type=ElementType.HEADING, content="Introduction"),
                DocumentElement(type=ElementType.HEADING, content="Methods"),
            ],
            title="My Paper",
        )
        summary = ContextualEnricher(agent=None)._summarize_document(doc)
        assert "Introduction" in summary
        assert "Methods" in summary

    def test_falls_back_to_title(self) -> None:
        doc = _make_doc(title="Fallback Title")
        summary = ContextualEnricher(agent=None)._summarize_document(doc)
        assert summary == "Fallback Title"

    def test_falls_back_to_stem_when_no_title(self) -> None:
        doc = ParsedDocument(source_file=Path("my_document.md"), elements=[], title="")
        summary = ContextualEnricher(agent=None)._summarize_document(doc)
        assert summary == "my_document"


class TestEnrichTreeNoClient:
    """Without an LLM client, enrich_tree returns tree unchanged."""

    def test_returns_same_tree(self) -> None:
        enricher = ContextualEnricher(agent=None)
        tree = _make_tree(_make_leaf("x"))
        doc = _make_doc()
        result = enricher.enrich_tree(tree, doc)
        assert result is tree

    def test_leaves_unchanged(self) -> None:
        enricher = ContextualEnricher(agent=None)
        leaf = _make_leaf("x")
        tree = _make_tree(leaf)
        enricher.enrich_tree(tree, _make_doc())
        assert leaf.contextual_context == ""


class TestEnrichTreeWithClient:
    """With an LLM client, leaf nodes get contextual_context set."""

    def test_context_set_on_leaves(self) -> None:
        enricher = self._make_enricher("This chunk is about X.")
        tree = _make_tree(_make_leaf("leaf1"))
        enricher.enrich_tree(tree, _make_doc())
        assert tree.nodes["leaf1"].contextual_context == "This chunk is about X."

    def test_all_leaves_enriched(self) -> None:
        enricher = self._make_enricher("some context")
        tree = _make_tree(_make_leaf("a"), _make_leaf("b"), _make_leaf("c"))
        enricher.enrich_tree(tree, _make_doc())
        for node_id in ("a", "b", "c"):
            assert tree.nodes[node_id].contextual_context != ""

    def test_llm_failure_raises(self) -> None:
        """Policy (.agents/CODE.md, "LLM calls"): a failed enrichment burns."""
        mock = MagicMock()
        mock.run.side_effect = RuntimeError("LLM unavailable")
        enricher = ContextualEnricher(agent=mock)
        leaf = _make_leaf("x", heading=("Introduction",))
        tree = _make_tree(leaf)

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            enricher.enrich_tree(tree, _make_doc("My Paper"))

    def test_empty_llm_response_raises(self) -> None:
        """An empty model response is a failed enrichment, not a fallback."""
        enricher = self._make_enricher("")  # empty response
        leaf = _make_leaf("y", heading=("Results",))
        tree = _make_tree(leaf)

        with pytest.raises(ValueError, match="empty response"):
            enricher.enrich_tree(tree, _make_doc("Doc"))

    def test_full_chunk_text_reaches_the_model(self) -> None:
        """The prompt carries the whole chunk — no excerpt crop (policy)."""
        long_text = "word " * 200  # ~1000 chars, far beyond any display excerpt
        mock = MagicMock()
        mock.run.return_value = "context"
        enricher = ContextualEnricher(agent=mock)
        leaf = _make_leaf("z", text=long_text)

        enricher.enrich_tree(_make_tree(leaf), _make_doc("Doc"))

        rendered = mock.run.call_args.args[0].text
        assert long_text.strip() in rendered

    def test_all_headings_reach_the_prompt(self) -> None:
        """The document outline is not cropped to a display count (policy)."""
        doc = ParsedDocument(
            source_file=Path("doc.md"),
            elements=[DocumentElement(type=ElementType.HEADING, content=f"Heading {index}") for index in range(8)],
            title="Many Headings",
        )
        summary = ContextualEnricher(agent=None)._summarize_document(doc)

        for index in range(8):
            assert f"Heading {index}" in summary

    def _make_enricher(self, response: str = "context paragraph") -> ContextualEnricher:
        mock = MagicMock()
        mock.run.return_value = response
        return ContextualEnricher(agent=mock)


def _make_doc(title: str = "Test Doc") -> ParsedDocument:

    return ParsedDocument(
        source_file=Path("test.md"),
        elements=[],
        title=title,
    )


def _make_leaf(node_id: str, text: str = "some text", heading: tuple[str, ...] = ("Section A",)) -> ChunkNode:
    chunk = Chunk(
        id=node_id,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[0],
    )
    return ChunkNode(
        id=node_id,
        heading_chain=heading,
        chunk=chunk,
    )


def _make_tree(*leaves: ChunkNode) -> ChunkTree:
    root = ChunkNode(id="root", children_ids=[leaf.id for leaf in leaves])
    nodes: dict[str, ChunkNode] = {"root": root}
    for leaf in leaves:
        nodes[leaf.id] = leaf
    return ChunkTree(root_id="root", nodes=nodes)
