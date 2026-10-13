"""Tests for the EnhancedRetriever pipeline."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.search import CragGradeAgent, HydeAgent, QueryExpansionAgent
from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.templates import get_template_loader
from kompend.search.pipeline import EnhancedRetriever
from kompend.search.retriever import SearchResult, SynthesisResult


class TestFuseMultiQuery:
    """_fuse_multi_query merges ranked lists via RRF."""

    def test_single_list_returned_unchanged(self) -> None:
        results = [_search_result("a"), _search_result("b")]
        fused = EnhancedRetriever._fuse_multi_query([results], top_k=5)
        ids = [r.article_id for r in fused]
        assert ids == ["a", "b"]

    def test_two_lists_merged_with_rrf(self) -> None:
        list_a = [_search_result("a"), _search_result("b")]
        list_b = [_search_result("b"), _search_result("c")]
        fused = EnhancedRetriever._fuse_multi_query([list_a, list_b], top_k=5)
        ids = [r.article_id for r in fused]
        # "b" appears in both lists at good rank — it should rank highest
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_top_k_respected(self) -> None:
        list_a = [_search_result(f"x{i}") for i in range(10)]
        fused = EnhancedRetriever._fuse_multi_query([list_a], top_k=3)
        assert len(fused) == 3

    def test_scores_are_positive(self) -> None:
        list_a = [_search_result("a"), _search_result("b")]
        fused = EnhancedRetriever._fuse_multi_query([list_a], top_k=5)
        for r in fused:
            assert r.score > 0


class TestSearchPipeline:
    """EnhancedRetriever.search runs the pipeline."""

    def test_no_features_delegates_to_hybrid(self) -> None:
        wiki = _make_wiki()
        retriever = EnhancedRetriever(wiki)
        results = retriever.search("alpha", top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_k_respected(self) -> None:
        wiki = _make_wiki()
        retriever = EnhancedRetriever(wiki)
        results = retriever.search("alpha beta", top_k=1)
        assert len(results) <= 1

    def test_returns_empty_on_no_results(self) -> None:
        wiki = _make_wiki()
        retriever = EnhancedRetriever(wiki)
        # Mock inner retriever to return nothing
        retriever._retriever.search_hybrid = MagicMock(return_value=[])
        results = retriever.search("something completely absent", top_k=5)
        assert results == []

    def test_crag_drops_low_relevance(self) -> None:
        wiki = _make_wiki()
        mock_agent = MagicMock()
        mock_agent.run.return_value = MagicMock(relevance=0.0)
        retriever = EnhancedRetriever(
            wiki,
            crag_enabled=True,
            crag_threshold=0.5,
            crag_agent=mock_agent,
        )
        candidates = [_search_result("a"), _search_result("b")]
        # When all are dropped CRAG falls back to original
        result = retriever._crag_filter("query", candidates)
        # Should return original set (safety fallback)
        assert len(result) == 2

    def test_crag_keeps_high_relevance(self) -> None:
        wiki = _make_wiki()
        mock_agent = MagicMock()
        mock_agent.run.return_value = MagicMock(relevance=0.9)
        retriever = EnhancedRetriever(
            wiki,
            crag_enabled=True,
            crag_threshold=0.5,
            crag_agent=mock_agent,
        )
        candidates = [_search_result("a"), _search_result("b")]
        result = retriever._crag_filter("query", candidates)
        assert len(result) == 2


class TestSynthesize:
    """synthesize returns the synthesis result."""

    def test_synthesize_returns_result(self) -> None:
        wiki = _make_wiki()
        retriever = EnhancedRetriever(wiki)

        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content="synthesized answer",
            json_data=None,
        )
        retriever._synthesis_model = "test"

        results = [_search_result("alpha")]
        # Patch inner synthesizer to avoid full pipeline
        with patch("kompend.search.pipeline.synthesize_results") as mock_synth:
            mock_synth.return_value = SynthesisResult(
                answer="the answer",
                citations=[{"article": "alpha", "article_id": None, "source": None, "snippet": "", "score": 1.0}],
            )
            synthesis = retriever.synthesize("q", results)

        assert synthesis.answer == "the answer"


class TestSearchAgentPrompts:
    """Regression: the caller pre-renders the prompt; the agent must not re-render it.

    These agents previously declared the variable-bearing ``search/*.md.j2``
    template as both user and system template, so the base rendered it with no
    variables and raised — swallowed by the pipeline's fallback, silently
    disabling the feature.
    """

    @pytest.mark.parametrize("agent_cls", [QueryExpansionAgent, CragGradeAgent, HydeAgent])
    def test_agent_returns_prerendered_prompt(self, agent_cls: type) -> None:
        agent = agent_cls(model_string="ollama/gpt-oss:20b")
        deps = PipelineAgentDeps(text="PROMPT BODY", source_id="test", timeout=120)
        assert agent.render_user_prompt(deps) == "PROMPT BODY"
        assert agent.render_system_prompt(deps) == ""

    def test_search_prompt_templates_render(self) -> None:
        loader = get_template_loader()
        expanded = loader.render("search/expand_query.md.j2", query="voice activity detection")
        assert "voice activity detection" in expanded

        graded = loader.render("search/crag_grade.md.j2", query="the query", title="Title", snippet="Snippet")
        assert "the query" in graded
        assert "Title" in graded
        assert "Snippet" in graded

        hyde = loader.render("search/hyde_generate.md.j2", query="what is VAD")
        assert "what is VAD" in hyde


class TestRetrievalEnhancementWarnings:
    """A retrieval-enhancement stage that cannot run must warn, not fail silently."""

    def test_expansion_failure_warns_and_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("boom")
        retriever = EnhancedRetriever(_make_wiki(), expansion_agent=agent, expansion_enabled=True)
        with caplog.at_level(logging.WARNING):
            assert retriever._expand_queries("q") == ["q"]
        assert _warned(caplog, "Query expansion")

    def test_hyde_failure_warns_and_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("boom")
        retriever = EnhancedRetriever(_make_wiki(), hyde_agent=agent, hyde_enabled=True)
        with caplog.at_level(logging.WARNING):
            assert retriever._generate_hypothetical("q") == "q"
        assert _warned(caplog, "HyDE generation")

    def test_crag_failure_warns_and_returns_neutral(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("boom")
        retriever = EnhancedRetriever(_make_wiki(), crag_agent=agent, crag_enabled=True)
        with caplog.at_level(logging.WARNING):
            assert retriever._grade_relevance("q", _search_result("a")) == 0.5
        assert _warned(caplog, "CRAG grading")


def _warned(caplog: pytest.LogCaptureFixture, needle: str) -> bool:
    return any(record.levelno == logging.WARNING and needle in record.getMessage() for record in caplog.records)


def _make_wiki() -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(id="alpha", title="Alpha Page", content="alpha content here", node_type="section"),
            WikiNode(id="beta", title="Beta Page", content="beta content here", node_type="section"),
        ],
        edges=[],
        toc=[],
    )


def _search_result(article_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(article_id=article_id, title=article_id.title(), snippet="snip", score=score)
