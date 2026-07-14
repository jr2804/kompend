"""Tests for the search retriever and snippet extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from knox.agents.query import QueryAgent
from knox.core.models import CompiledWiki, WikiNode
from knox.llm.context_window import ContextSource, ContextWindow
from knox.search.retriever import HybridRetriever, SearchResult, TokenOverlapRetriever, _extract_snippet


class TestExtractSnippet:
    """Snippet extraction picks the window richest in query tokens."""

    def test_finds_query_rich_window(self) -> None:
        # Place the query-rich region after the 60-word fixed window so the
        # algorithm must move best_start forward rather than stay at zero.
        content = "filler " * 70 + "the quick brown fox jumps over the lazy dog. " + "trailing " * 20
        query_tokens = {"fox", "dog", "jumps"}
        snippet = _extract_snippet(content, query_tokens, max_chars=200)
        assert "fox" in snippet.lower()
        assert "dog" in snippet.lower()
        assert not snippet.startswith("filler")
        assert snippet.endswith("...")

    def test_fallback_when_no_query_tokens_match(self) -> None:
        content = "Alpha beta gamma delta epsilon."
        snippet = _extract_snippet(content, {"xyz", "zzz"}, max_chars=20)
        assert snippet.startswith("Alpha")
        assert snippet.endswith("...")

    def test_respects_max_chars(self) -> None:
        content = "word " * 200
        snippet = _extract_snippet(content, {"word"}, max_chars=50)
        assert len(snippet) <= 55  # allow for trailing ellipsis and rsplit
        assert "word" in snippet.lower()


class _FakeVectorStore:
    """In-memory stand-in for ZvecWikiStore in retriever tests."""

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self._hits = hits or []
        self.available = True

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:  # noqa: ARG002
        return self._hits[:top_k]


class TestHybridRetriever:
    """HybridRetriever fuses multiple signals via reciprocal rank fusion."""

    def test_fusion_rrf_combines_ranked_lists(self) -> None:
        wiki = self._make_wiki()
        retriever = HybridRetriever(wiki)
        list_a = [SearchResult(article_id="alpha", title="Alpha", snippet="snip", score=1.0)]
        list_b = [SearchResult(article_id="beta", title="Beta", snippet="snip", score=1.0)]

        fused = retriever._fuse_results(list_a, list_b, top_k=5)

        ids = [r.article_id for r in fused]
        assert "alpha" in ids
        assert "beta" in ids
        # Both appear exactly once with equal RRF score when each is ranked first
        # in exactly one list and the second list contributes nothing.
        alpha = next(r for r in fused if r.article_id == "alpha")
        beta = next(r for r in fused if r.article_id == "beta")
        assert alpha.score == beta.score

    def test_search_wiki_falls_back_to_token_overlap(self) -> None:
        wiki = self._make_wiki()
        retriever = HybridRetriever(wiki)
        results = retriever.search_wiki("alpha", top_k=5)
        assert any(r.article_id == "alpha" for r in results)

    def test_search_wiki_uses_vector_store(self) -> None:
        wiki = self._make_wiki()
        fake = _FakeVectorStore(hits=[{"id": "beta", "score": 0.99, "source_file": "beta.pdf"}])
        retriever = HybridRetriever(wiki, vector_store=fake)
        results = retriever.search_wiki("beta", top_k=5)
        beta = next((r for r in results if r.article_id == "beta"), None)
        assert beta is not None
        assert beta.source_file == "beta.pdf"
        assert "beta" in beta.snippet.lower()

    def test_search_hybrid_includes_raw_sources(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        (sources_dir / "apple.md").write_text("This file is about apples and orchards.")
        (sources_dir / "banana.md").write_text("Bananas are yellow tropical fruit.")

        wiki = self._make_wiki()
        retriever = HybridRetriever(wiki, sources_dir=sources_dir)
        # Use "bananas" to match the tokenized form of "Bananas"
        results = retriever.search("bananas tropical", top_k=5)

        raw = next((r for r in results if r.article_id == "banana"), None)
        assert raw is not None
        assert raw.source_file == str(sources_dir / "banana.md")

    def test_search_wiki_empty_query_returns_empty(self) -> None:
        wiki = self._make_wiki()
        retriever = HybridRetriever(wiki)
        assert retriever.search_wiki("a", top_k=5) == []

    def test_search_hybrid_returns_results_from_each_subsystem(self, tmp_path: Path) -> None:
        """Verify hybrid search fuses results from FTS, vector, token, and BM25 raw."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        (sources_dir / "apple.md").write_text("This file is about apples and orchards.")
        (sources_dir / "banana.md").write_text("Bananas are yellow tropical fruit.")

        wiki = self._make_wiki_with_edges()
        fake_vector = _FakeVectorStore(hits=[{"id": "beta", "score": 0.95, "source_file": "beta.pdf"}])
        retriever = HybridRetriever(wiki, vector_store=fake_vector, sources_dir=sources_dir)

        # Search for something that matches raw BM25 ("bananas") and wiki token/vector ("beta")
        results = retriever.search_hybrid("beta bananas", top_k=5)

        # Debug: print results
        print("RESULTS:", [(r.article_id, r.score) for r in results])

        # Should contain wiki result (beta)
        assert any(r.article_id == "beta" for r in results)
        # Should contain raw BM25 result (banana)
        assert any(r.article_id == "banana" for r in results)

    def test_graph_expansion_adds_related_nodes(self, tmp_path: Path) -> None:
        """Verify graph expansion follows edges and adds related nodes with decayed score."""
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        wiki = self._make_wiki_with_edges()
        retriever = HybridRetriever(wiki, sources_dir=sources_dir)

        # Search for "alpha" which should match the alpha node
        results = retriever.search_hybrid("alpha", top_k=5, expand_graph=True)

        # Should contain the original result
        assert any(r.article_id == "alpha" for r in results)
        # Should contain the expanded result (gamma) via the edge alpha -> gamma
        assert any(r.article_id == "gamma" for r in results)

        # Verify decayed score
        alpha_result = next(r for r in results if r.article_id == "alpha")
        gamma_result = next(r for r in results if r.article_id == "gamma")
        assert gamma_result.score <= alpha_result.score * 0.5

    def _make_wiki(self) -> CompiledWiki:
        return CompiledWiki(
            nodes=[
                WikiNode(id="alpha", title="Alpha Page", content="alpha content here", node_type="section"),
                WikiNode(id="beta", title="Beta Page", content="beta content here", node_type="section"),
            ],
            edges=[],
            toc=[],
        )

    def _make_wiki_with_edges(self) -> CompiledWiki:
        return CompiledWiki(
            nodes=[
                WikiNode(id="alpha", title="Alpha Page", content="alpha content here", node_type="section"),
                WikiNode(id="beta", title="Beta Page", content="beta content here", node_type="section"),
                WikiNode(id="gamma", title="Gamma Page", content="gamma related content", node_type="section"),
            ],
            edges=[
                {"source": "alpha", "target": "gamma", "type": "related_to"},
                {"source": "beta", "target": "alpha", "type": "depends_on"},
            ],
            toc=[],
        )


class _FakeLLMClient:
    """Captures complete() kwargs for synthesis tests."""

    def __init__(self, max_output: int = 4096) -> None:
        self.default_model = "test-model"
        self.context_window = ContextWindow(max_input=8192, max_output=max_output, source=ContextSource.FALLBACK)
        self.last_kwargs: dict[str, Any] = {}
        self.last_messages: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> MagicMock:  # noqa: ANN401
        self.last_messages = messages
        self.last_kwargs = kwargs
        return MagicMock(
            content="synthesized answer",
            json_data=None,
        )


class TestSynthesizeVerbosity:
    """Query synthesis routes through the PydanticAI QueryAgent."""

    def test_routes_to_query_agent(self) -> None:
        result, agent = self._synthesize("standard")
        assert result.answer == "answer (standard)"
        agent.run.assert_called_once()

    def test_passes_verbosity_in_deps(self) -> None:
        _, agent = self._synthesize("brief")
        deps = agent.run.call_args.args[0]
        assert deps.verbosity == "brief"
        assert deps.question == "q"

    def test_brief_verbosity_renders_brief_instruction(self) -> None:
        agent = QueryAgent(litellm_model="model")
        deps = MagicMock()
        deps.question = "q"
        deps.context = "[1] Some context"
        deps.verbosity = "brief"
        deps.source_id = "query"
        system = agent.render_system_prompt(deps)
        assert "concise" in system.lower()

    def test_comprehensive_verbosity_renders_comprehensive_instruction(self) -> None:
        agent = QueryAgent(litellm_model="model")
        deps = MagicMock()
        deps.question = "q"
        deps.context = "[1] Some context"
        deps.verbosity = "comprehensive"
        deps.source_id = "query"
        system = agent.render_system_prompt(deps)
        assert "comprehensive" in system.lower()

    def test_standard_verbosity_renders_balanced_instruction(self) -> None:
        agent = QueryAgent(litellm_model="model")
        deps = MagicMock()
        deps.question = "q"
        deps.context = "[1] Some context"
        deps.verbosity = "standard"
        deps.source_id = "query"
        system = agent.render_system_prompt(deps)
        assert "balanced" in system.lower()

    def _synthesize(self, verbosity: str, max_tokens: int | None = 100) -> tuple[Any, Any]:
        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        fake = _FakeLLMClient()
        retriever = TokenOverlapRetriever(wiki)
        mock_agent = MagicMock()
        mock_agent.run = MagicMock(return_value=f"answer ({verbosity})")
        with patch("knox.search.retriever.QueryAgent", return_value=mock_agent):
            result = retriever.synthesize("q", [], model=fake.default_model, verbosity=verbosity, max_tokens=max_tokens)
        return result, mock_agent
