from __future__ import annotations

from knox.core.models import CompiledWiki, WikiNode
from knox.search.adaptive_retriever import AdaptiveRetriever


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


class TestAdaptiveRetriever:
    def test_search_empty_wiki(self) -> None:
        wiki = _make_wiki([])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever.search("test query")
        assert results == []

    def test_search_single_node(self) -> None:
        node = _make_wiki_node("n1", content="signal to noise ratio is important")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever.search("signal", top_k=5)
        assert len(results) == 1
        assert results[0].article_id == "n1"

    def test_search_top_k_limits(self) -> None:
        nodes = [_make_wiki_node(f"n{i}", content=f"unique content about topic {i}") for i in range(10)]
        wiki = _make_wiki(nodes)
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever.search("topic", top_k=3)
        assert len(results) <= 3

    def test_search_zoom_in_mode(self) -> None:
        nodes = [
            _make_wiki_node("root", depth=0),
            _make_wiki_node("s1", content="section one about filtering", depth=1, parent_id="root"),
            _make_wiki_node("s2", content="section two about noise", depth=1, parent_id="root"),
        ]
        wiki = _make_wiki(nodes)
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever.search("filtering", mode="zoom_in", top_k=5)
        assert len(results) > 0

    def test_search_zoom_out_mode(self) -> None:
        nodes = [
            _make_wiki_node("root", depth=0),
            _make_wiki_node("s1", content="section one", depth=1, parent_id="root"),
            _make_wiki_node("s2", content="section two", depth=1, parent_id="root"),
        ]
        wiki = _make_wiki(nodes)
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever.search("section", mode="zoom_out", top_k=5)
        assert len(results) > 0

    def test_search_targeted_mode(self) -> None:
        structured = _make_wiki_node("structured", content="table data", node_type="concept")
        structured.structured_sections["table"] = {"columns": ["a", "b"]}
        plain = _make_wiki_node("plain", content="just text")
        wiki = _make_wiki([structured, plain])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever.search("table", mode="targeted", top_k=5)
        assert len(results) > 0

    def test_search_unknown_mode_fallback(self) -> None:
        node = _make_wiki_node("n1", content="test content")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever.search("test", mode="unknown_mode", top_k=5)
        assert len(results) == 1

    def test_collect_candidates_returns_nodes(self) -> None:
        node = _make_wiki_node("n1", content="signal processing")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        candidates = retriever._collect_candidates("signal", top_k=10)
        assert len(candidates) > 0
        assert candidates[0].id == "n1"

    def test_collect_candidates_empty_query(self) -> None:
        node = _make_wiki_node("n1", content="test")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        candidates = retriever._collect_candidates("", top_k=10)
        assert len(candidates) == 0

    def test_wiki_token_results(self) -> None:
        node = _make_wiki_node("n1", content="unique keyword match")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever._wiki_token_results("unique keyword", top_k=5)
        assert len(results) == 1
        assert results[0].article_id == "n1"

    def test_wiki_token_results_no_match(self) -> None:
        node = _make_wiki_node("n1", content="irrelevant text")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever._wiki_token_results("zzzzz", top_k=5)
        assert len(results) == 0

    def test_wiki_fts_results_no_index(self) -> None:
        node = _make_wiki_node("n1", content="test")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever._wiki_fts_results("test", top_k=5)
        assert results == []

    def test_wiki_vector_results_no_store(self) -> None:
        node = _make_wiki_node("n1", content="test")
        wiki = _make_wiki([node])
        retriever = AdaptiveRetriever(wiki, max_input_tokens=128_000)
        results = retriever._wiki_vector_results("test", top_k=5)
        assert results == []
