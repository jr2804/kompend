"""Tests for ``BaseCompiler.merge_by_concept_key`` — deterministic duplicate merge.

Regression: p863 wiki roundup (2026-09-10) showed three near-duplicate
"Absolute Category Rating" pages survived all dedup layers. Root cause:
the embedding-cosine merge in :meth:`BaseCompiler.merge_duplicate_nodes`
silently returned input unchanged when the embedding provider failed,
and the deterministic ``normalize_concept_key`` merge only ran at the
extraction stage, never at the node stage. The new
``merge_by_concept_key`` runs BEFORE the embedding pass and force-merges
on key collision regardless of provider availability.
"""

from __future__ import annotations

from kompend.compilation._two_phase_core import upsert_node
from kompend.compilation.base import BaseCompiler
from kompend.core.models import WikiNode
from kompend.core.models.enums import NodeType


class TestMergeByConceptKey:
    def test_collapses_two_parens_variants(self) -> None:
        """'Absolute Category Rating' and 'Absolute Category Rating (ACR)'
        share a concept key — must collapse.
        """
        a = _node("Absolute Category Rating", "body A")
        b = _node("Absolute Category Rating (ACR)", "body B")
        merged = BaseCompiler.merge_by_concept_key([a, b])
        assert len(merged) == 1
        # Longest title wins (preserves descriptive form).
        assert merged[0].title == "Absolute Category Rating (ACR)"
        # Content sections unioned (section-aware dedup in _merge_node).
        assert "body A" in merged[0].content or "body B" in merged[0].content

    def test_keeps_distinct_concepts(self) -> None:
        """Different keys stay separate."""
        a = _node("Absolute Category Rating", "A")
        b = _node("Degradation Category Rating", "B")
        c = _node("Comparison Category Rating", "C")
        merged = BaseCompiler.merge_by_concept_key([a, b, c])
        assert len(merged) == 3
        titles = {n.title for n in merged}
        assert titles == {"Absolute Category Rating", "Degradation Category Rating", "Comparison Category Rating"}

    def test_preserves_order_of_unmerged_nodes(self) -> None:
        """Nodes not involved in merges keep their relative order."""
        a = _node("Alpha", "A")
        b = _node("Beta (B)", "B1")
        b2 = _node("Beta", "B2")  # same key as b → collides
        c = _node("Gamma", "C")
        merged = BaseCompiler.merge_by_concept_key([a, b, b2, c])
        assert [n.title for n in merged] == ["Alpha", "Beta (B)", "Gamma"]
        assert len(merged) == 3

    def test_passes_through_non_concept_nodes(self) -> None:
        """Entities / summaries / overviews are NOT merged here."""
        e = WikiNode(
            id="entity-x",
            title="Entity X",
            content="x",
            node_type=NodeType.ENTITY,
            source_chunks=[],
            metadata={},
        )
        s = WikiNode(
            id="summary-y",
            title="Summary Y",
            content="y",
            node_type=NodeType.SUMMARY,
            source_chunks=[],
            metadata={},
        )
        merged = BaseCompiler.merge_by_concept_key([e, s])
        assert merged == [e, s]

    def test_does_not_mutate_input_list(self) -> None:
        """Original list is left alone (merge returns a new list)."""
        a = _node("X", "A")
        b = _node("X (extra)", "B")
        original = [a, b]
        merged = BaseCompiler.merge_by_concept_key(original)
        assert merged is not original
        assert len(original) == 2  # input untouched

    def test_handles_empty_input(self) -> None:
        assert BaseCompiler.merge_by_concept_key([]) == []

    def test_handles_single_node(self) -> None:
        a = _node("Only", "x")
        merged = BaseCompiler.merge_by_concept_key([a])
        assert merged == [a]


class TestUpsertNodeHardensSourceChunks:
    """Regression: p863 wiki had ``source_chunks=None`` on one duplicate node."""

    def test_source_chunks_none_coerced_to_empty_list(self) -> None:
        node = WikiNode(
            id="x",
            title="X",
            content="x",
            node_type=NodeType.CONCEPT,
            source_chunks=None,  # type: ignore[arg-type]
            metadata={},
        )
        nodes: list[WikiNode] = []
        slug_idx: dict[str, int] = {}
        upsert_node(node, nodes, slug_idx)
        assert nodes[0].source_chunks == []


def _node(title: str, content: str = "body", chunks: list[str] | None = None) -> WikiNode:
    return WikiNode(
        id=title.lower().replace(" ", "-"),
        title=title,
        content=content,
        node_type=NodeType.CONCEPT,
        source_chunks=chunks or [],
        metadata={},
    )
