"""The optional cross-encoder rerank stage.

The stage is best-effort enrichment: a missing model, a missing dependency
or a failing reranker leaves the candidates untouched. These tests cover the
mechanics and the degradation — never a model download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import kompend.rag
from kompend.core.config._configs import RerankerBackend
from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import NodeType
from kompend.rag.rerank import FlashRankReranker, RerankerUnavailable
from kompend.search.pipeline import EnhancedRetriever
from kompend.search.retriever import SearchResult


class _StubReranker:
    """Scores by the length of the passage — deterministic, no model."""

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [float(len(passage)) for passage in passages]


class _FakeRanker:
    """Stands in for flashrank's ranker — returns a fixed ranking."""

    def __init__(self, ranked: list[dict[str, object]]) -> None:
        self._ranked = ranked

    def rerank(self, request: object) -> list[dict[str, object]]:
        return self._ranked


def test_flashrank_reranker_reports_a_missing_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vault without the 'rerank' extra gets a clear error, not a crash."""

    def _missing(*args: object, **kwargs: object) -> object:
        raise ImportError("No module named 'flashrank'")

    _stub_loader(monkeypatch, _missing)
    with pytest.raises(RerankerUnavailable):
        _build_reranker(tmp_path)


def test_flashrank_reranker_reports_a_model_that_will_not_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that cannot be downloaded is a stage failure, not a crash."""

    def _broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("model download failed")

    _stub_loader(monkeypatch, _broken)
    with pytest.raises(RerankerUnavailable):
        _build_reranker(tmp_path)


def test_scores_come_back_in_the_input_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reranker's scores line up with the passages it was given.

    The model returns the passages in its own order; the stage maps them back
    onto the input.
    """
    # This one builds a real flashrank request, so it needs the extra.
    pytest.importorskip("flashrank", reason="the 'rerank' extra is not installed")

    def _loader(*args: object, **kwargs: object) -> _FakeRanker:
        return _FakeRanker([{"id": "2", "score": 0.9}, {"id": "0", "score": 0.1}])

    _stub_loader(monkeypatch, _loader)
    scores = _build_reranker(tmp_path).rerank("q", ["a", "bb", "ccc"])
    assert scores == [0.1, 0.0, 0.9]


def _build_reranker(tmp_path: Path) -> FlashRankReranker:
    """A reranker whose model is already stubbed out — never a download."""
    return FlashRankReranker(cache_dir=tmp_path)


def _stub_loader(monkeypatch: pytest.MonkeyPatch, loader: object) -> None:
    """Replace the lazy ``kompend.rag.load_ranker`` without importing flashrank.

    The attribute is served by ``lazy_loader``, so *reading* it imports the
    backend module — which imports flashrank, and fails when the 'rerank'
    extra is not installed. Patching the module dict shadows the lazy lookup
    without triggering the import, so these tests behave the same with and
    without the extra.
    """
    monkeypatch.setitem(kompend.rag.__dict__, "load_ranker", loader)


def test_the_backend_module_imports_when_the_extra_is_installed() -> None:
    """The lazy backend module must import cleanly when the extra is present.

    Without the extra it is never imported — that is what the lazy boundary
    buys — so this is the one test that catches a typo in the module itself.
    """
    pytest.importorskip("flashrank", reason="the 'rerank' extra is not installed")
    assert callable(kompend.rag.load_ranker)
    assert callable(kompend.rag.build_request)


def test_cross_encoder_stage_reorders_the_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cross-encoder's ranking wins over the retrieval order."""
    retriever = EnhancedRetriever(
        _wiki(),
        reranker_enabled=True,
        reranker_backend=RerankerBackend.FLASHRANK,
        reranker_top_k=2,
    )
    monkeypatch.setattr(retriever, "_ensure_reranker", _StubReranker)
    candidates = [_result("a", "x"), _result("b", "xxx"), _result("c", "xx")]
    reranked = retriever._neural_rerank("query", candidates)
    assert [r.article_id for r in reranked] == ["b", "c"]
    assert len(reranked) == 2


def test_cross_encoder_stage_keeps_the_page_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage that rebuilds a result forwards the body and the figures."""
    retriever = EnhancedRetriever(
        _wiki(),
        reranker_enabled=True,
        reranker_backend=RerankerBackend.FLASHRANK,
    )
    monkeypatch.setattr(retriever, "_ensure_reranker", _StubReranker)
    reranked = retriever._neural_rerank("query", [_result("a", "the body")])
    assert reranked[0].content == "the body"


def test_unavailable_reranker_keeps_the_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reranker that cannot run leaves the candidates as they were."""
    retriever = EnhancedRetriever(
        _wiki(),
        reranker_enabled=True,
        reranker_backend=RerankerBackend.FLASHRANK,
    )

    def _unavailable() -> object:
        raise RerankerUnavailable("no model")

    monkeypatch.setattr(retriever, "_ensure_reranker", _unavailable)
    candidates = [_result("a"), _result("b")]
    assert retriever._neural_rerank("query", candidates) == candidates


def test_the_none_backend_skips_the_stage() -> None:
    """``backend = none`` leaves the candidates untouched."""
    retriever = EnhancedRetriever(
        _wiki(),
        reranker_enabled=True,
        reranker_backend=RerankerBackend.NONE,
    )
    candidates = [_result("a")]
    assert retriever._neural_rerank("query", candidates) == candidates


def _result(article_id: str, content: str = "body") -> SearchResult:
    return SearchResult(
        article_id=article_id,
        title=f"title {article_id}",
        snippet=f"snippet {article_id}",
        score=0.5,
        content=content,
    )


def _wiki() -> CompiledWiki:
    return CompiledWiki(
        nodes=[WikiNode(id="n1", title="t", content="c", node_type=NodeType.SECTION)],
        edges=[],
        toc=[],
    )
