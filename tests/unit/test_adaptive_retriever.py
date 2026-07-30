"""Tests for AdaptiveRetriever — vector-only retrieval with adaptive modes.

The retriever is purely vector-backed (token/FTS paths were removed in the
PydanticAI migration). Tests inject an in-memory ``_FakeVectorStore`` so the
vector path returns hits without requiring a real embedding index.
"""

from __future__ import annotations

from typing import Any

from knox.core.models import CompiledWiki, WikiNode
from knox.search.adaptive_retriever import AdaptiveRetriever


class _FakeVectorStore:
    """In-memory stand-in for ZvecWikiStore. Returns canned hits keyed by node id."""

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self._hits = hits or []
        self.available = True

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:  # noqa: ARG002
        return self._hits[:top_k]


class TestAdaptiveRetriever:
    def test_search_empty_wiki(self) -> None:
        wiki = _make_wiki([])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever.search("test query")
        assert results == []

    def test_search_single_node(self) -> None:
        node = _make_wiki_node("n1", content="signal to noise ratio is important", depth=1, parent_id="root")
        wiki = _make_wiki([node])
        fake = _FakeVectorStore(hits=[{"id": "n1", "score": 0.9}])
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        results = retriever.search("signal", top_k=5)
        assert len(results) == 1
        assert results[0].article_id == "n1"

    def test_search_top_k_limits(self) -> None:
        nodes = [_make_wiki_node(f"n{i}", content=f"unique content about topic {i}") for i in range(10)]
        wiki = _make_wiki(nodes)
        # Vector store returns all 10; search must clamp to top_k.
        fake = _FakeVectorStore(hits=[{"id": f"n{i}", "score": 0.9 - i * 0.01} for i in range(10)])
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        results = retriever.search("topic", top_k=3)
        assert len(results) <= 3

    def test_search_zoom_in_mode(self) -> None:
        nodes = [
            _make_wiki_node("root", depth=0),
            _make_wiki_node("s1", content="section one about filtering", depth=1, parent_id="root"),
            _make_wiki_node("s2", content="section two about noise", depth=1, parent_id="root"),
        ]
        wiki = _make_wiki(nodes)
        fake = _FakeVectorStore(hits=[{"id": "s1", "score": 0.9}])
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        results = retriever.search("filtering", mode="zoom_in", top_k=5)
        assert len(results) > 0

    def test_search_zoom_out_mode(self) -> None:
        nodes = [
            _make_wiki_node("root", depth=0),
            _make_wiki_node("s1", content="section one", depth=1, parent_id="root"),
            _make_wiki_node("s2", content="section two", depth=1, parent_id="root"),
            _make_wiki_node("s3", content="section three", depth=1, parent_id="root"),
            _make_wiki_node("s4", content="section four", depth=1, parent_id="root"),
        ]
        wiki = _make_wiki(nodes)
        # zoom_out collapses leaf hits to their parent sections. The parent
        # "root" must appear in candidates so _find_node can resolve it.
        hits = [{"id": "root", "score": 0.95}]
        hits += [{"id": f"s{i}", "score": 0.9} for i in range(1, 5)]
        fake = _FakeVectorStore(hits=hits)
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        results = retriever.search("section", mode="zoom_out", top_k=5)
        assert len(results) > 0

    def test_search_targeted_mode(self) -> None:
        structured = _make_wiki_node("structured", content="table data", node_type="concept", depth=1, parent_id="root")
        structured.structured_sections["table"] = {"columns": ["a", "b"]}
        plain = _make_wiki_node("plain", content="just text", depth=1, parent_id="root")
        wiki = _make_wiki([structured, plain])
        fake = _FakeVectorStore(hits=[{"id": "structured", "score": 0.9}])
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        results = retriever.search("table", mode="targeted", top_k=5)
        assert len(results) > 0

    def test_search_unknown_mode_fallback(self) -> None:
        node = _make_wiki_node("n1", content="test content", depth=1, parent_id="root")
        wiki = _make_wiki([node])
        fake = _FakeVectorStore(hits=[{"id": "n1", "score": 0.9}])
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        results = retriever.search("test", mode="unknown_mode", top_k=5)
        assert len(results) == 1

    def test_collect_candidates_returns_nodes(self) -> None:
        node = _make_wiki_node("n1", content="signal processing", depth=1, parent_id="root")
        wiki = _make_wiki([node])
        fake = _FakeVectorStore(hits=[{"id": "n1", "score": 0.9}])
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        candidates = retriever._collect_candidates("signal", top_k=10)
        assert len(candidates) == 1
        assert candidates[0].id == "n1"

    def test_collect_candidates_empty_query(self) -> None:
        node = _make_wiki_node("n1", content="test")
        wiki = _make_wiki([node])
        # Empty query still goes through vector search; with no hits, no candidates.
        fake = _FakeVectorStore(hits=[])
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        candidates = retriever._collect_candidates("", top_k=10)
        assert len(candidates) == 0

    def test_wiki_vector_results_no_store(self) -> None:
        node = _make_wiki_node("n1", content="test")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever._wiki_vector_results("test", top_k=5)
        assert results == []

    def test_wiki_vector_results_with_store(self) -> None:
        node = _make_wiki_node("n1", content="test content", depth=1, parent_id="root")
        wiki = _make_wiki([node])
        fake = _FakeVectorStore(hits=[{"id": "n1", "score": 0.9}])
        retriever = AdaptiveRetriever(
            wiki,
            vector_store=fake,  # ty: ignore
            max_input_tokens=128_000,
        )
        results = retriever._wiki_vector_results("test", top_k=5)
        assert len(results) == 1
        assert results[0].article_id == "n1"


def _make_wiki_node(
    node_id: str,
    content: str = "test content about signal processing",
    depth: int = 0,
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    node_type: str = "concept",
) -> WikiNode:
    return WikiNode(
        id=node_id,
        title=node_id,
        content=content,
        node_type=node_type,
        depth=depth,
        parent_id=parent_id,
        children_ids=children_ids or [],
        metadata={"source": "test.pdf"},
    )


def _make_wiki(nodes: list[WikiNode]) -> CompiledWiki:
    return CompiledWiki(nodes=nodes, edges=[], toc=[])
