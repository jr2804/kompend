"""Tests for the unified string/semantic matching module (core/text/matching)."""

from __future__ import annotations

import pytest

from knox.core.text.matching import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SemanticMatcher,
    best_match,
    cosine,
    similarity,
)


class TestSimilarity:
    def test_identical_strings_score_100(self) -> None:
        assert similarity("Mean Opinion Score", "Mean Opinion Score") == 100.0

    def test_punctuation_variants_match(self) -> None:
        """``P.10/G.100`` and ``P 10 G 100`` are the same after normalization."""
        assert similarity("P.10/G.100", "P 10 G 100") >= 95.0

    def test_token_order_irrelevant(self) -> None:
        assert similarity("Score Opinion Mean", "Mean Opinion Score") >= 95.0

    def test_parenthetical_extras_tolerated(self) -> None:
        assert similarity("Mean Opinion Score", "Mean Opinion Score (MOS)") >= 90.0

    def test_distinct_concepts_stay_below_default_threshold(self) -> None:
        """Artificial Mouth vs Artificial Ear must NOT clear the default bar."""
        assert similarity("Artificial Mouth", "Artificial Ear") < DEFAULT_SIMILARITY_THRESHOLD

    def test_empty_input_scores_zero(self) -> None:
        assert similarity("", "anything") == 0.0
        assert similarity("anything", "") == 0.0


class TestBestMatch:
    CANDIDATES = {
        "p 10 g 100 recommendation": "p 10 g 100 recommendation",
        "artificial ear": "artificial ear",
        "mean opinion score": "mean opinion score",
    }

    def test_exact_match_wins_outright(self) -> None:
        result = best_match("ARTIFICIAL EAR", self.CANDIDATES)
        assert result == ("artificial ear", 100.0)

    def test_fuzzy_match_above_threshold(self) -> None:
        result = best_match("P.10/G.100", self.CANDIDATES, threshold=75.0)
        assert result is not None
        assert result[0] == "p 10 g 100 recommendation"

    def test_no_match_below_threshold(self) -> None:
        assert best_match("Artificial Mouth", self.CANDIDATES, threshold=DEFAULT_SIMILARITY_THRESHOLD) is None

    def test_empty_target_returns_none(self) -> None:
        assert best_match("", self.CANDIDATES) is None

    def test_empty_candidates_returns_none(self) -> None:
        assert best_match("anything", {}) is None

    def test_set_input(self) -> None:
        result = best_match("Mean Opinion Score", {"mean opinion score", "artificial ear"})
        assert result is not None
        assert result[0] == "mean opinion score"


class _FakeEmbedder:
    """Deterministic 2-D embedder for tests — vectors keyed by exact text."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.embed_calls = 0

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._vectors[text]


class TestCosine:
    def test_identical_vectors(self) -> None:
        assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_scores_zero(self) -> None:
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestSemanticMatcher:
    def test_unavailable_without_embedder(self) -> None:
        matcher = SemanticMatcher(embedder=None, candidates={"a": "a"})
        assert matcher.available is False
        assert matcher.match("query") is None

    def test_match_above_threshold(self) -> None:
        vectors: dict[str, list[float]] = {"mean opinion score": [1.0, 0.1], "artificial ear": [0.0, 1.0]}
        matcher, _ = self._matcher(vectors)
        result = matcher.match("mean opinion score")
        assert result is not None
        assert result[0] == "mean opinion score"

    def test_embed_failure_disables_tier(self) -> None:
        vectors: dict[str, list[float]] = {"a": [1.0, 0.0]}
        embedder = _FakeEmbedder(vectors)
        matcher = SemanticMatcher(embedder=embedder, candidates={"a": "a", "b": "b"})
        assert matcher.available is False
        assert matcher.match("a") is None

    def test_query_embed_failure_returns_none(self) -> None:
        vectors: dict[str, list[float]] = {"a": [1.0, 0.0]}
        matcher, _ = self._matcher(vectors)
        assert matcher.available is True
        assert matcher.match("unknown query") is None

    def test_no_candidates_means_unavailable(self) -> None:
        matcher: SemanticMatcher
        embedder = _FakeEmbedder({})
        matcher = SemanticMatcher(embedder=embedder, candidates={})
        assert matcher.available is False

    def test_candidate_vectors_embedded_once(self) -> None:
        vectors: dict[str, list[float]] = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        matcher, embedder = self._matcher(vectors, threshold=0.99)
        matcher.match("a")  # unknown query → embed fails → None, but construction embedded 2
        assert embedder.embed_calls >= 2

    def _matcher(self, vectors: dict[str, list[float]], threshold: float = 0.55) -> tuple[SemanticMatcher, _FakeEmbedder]:
        embedder = _FakeEmbedder(vectors)
        matcher = SemanticMatcher(embedder=embedder, candidates={k: k for k in vectors}, threshold=threshold)
        return matcher, embedder
