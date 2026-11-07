"""Tests for the concept-merge guard and the centroid linkage it replaced.

Regression: the p863-rev-eng wiki roundup (2026-11-06) compiled 413
concept candidates down to 31 pages. One transitive union-find chain
absorbed 98 sources under the canonical name "Appendix III - Prediction
of acoustically recorded narrowband speech", swallowing POLQA, MOS,
PESQ, temporal alignment, ACR, idealization and resampling with it.
Same-domain text sits above the cosine threshold for clearly distinct
concepts (measured with the configured embedder: cos(PESQ, POLQA)=0.886,
cos(temporal alignment, resampling)=0.876), and a transitive closure
chained them into one cluster no matter how far apart the endpoints sat.

Two things are pinned here:
- the grouping is centroid linkage, not transitive closure, so a pair that
  is only indirectly connected never lands in the same cluster;
- a merge that would swallow the pool, or a cluster spanning dozens of
  sources, is refused and logged rather than applied silently.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from kompend.compilation.flows.assembly import merge_similar_concepts


def _concept(name: str, *sources: str, definition: str = "") -> dict[str, Any]:
    """One concept candidate, as the extraction stage hands them over."""
    return {
        "name": name,
        "sources": list(sources),
        "definitions": [definition] if definition else [],
    }


class _FakeProvider:
    """An embedding provider that returns a vector per text by lookup.

    The vectors are handed in as text → vector, so a test can place two
    concepts at a chosen distance without needing a model.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> list[float]:
        return self._vectors[text]


@pytest.fixture
def _patched_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fake provider wherever the merge looks it up."""
    monkeypatch.setattr(
        "kompend.compilation.flows.assembly.make_provider",
        lambda _model: _FakeProvider(_VECTORS),
    )


#: The text each concept is embedded as ("<name>. <definition>").
_VECTORS: dict[str, list[float]] = {}


class TestCentroidLinkage:
    """A pair that is only indirectly connected must not merge."""

    def test_a_chain_does_not_collapse_the_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three concepts in a chain stay three under centroid linkage.

        A is close to B, B is close to C, A is far from C. Transitive
        closure merged all three; centroid linkage keeps A and C apart,
        because C's nearest centroid is B's.
        """
        vectors = {
            "A": [1.0, 0.0],
            "B": [0.9, 0.1],
            "C": [0.8, 0.2],
        }
        monkeypatch.setattr(
            "kompend.compilation.flows.assembly.make_provider",
            lambda _model: _FakeProvider(vectors),
        )
        concepts = [
            _concept("A", "s1"),
            _concept("B", "s2"),
            _concept("C", "s3"),
        ]

        # cos(A,B)=0.9939 and cos(B,C)=0.9910 both clear 0.99, so transitive
        # closure would chain all three into one cluster. cos(A,C)=0.9702 does
        # not, and the A+B centroid has moved to [0.95, 0.05] by the time C
        # arrives, leaving cos(C, centroid)=0.9818 — below the threshold.
        merged = merge_similar_concepts(
            concepts,
            "fake/model",
            threshold=0.99,
            string_threshold=0.7,
            jaccard_min_token_len=1,
        )

        assert len(merged) == 2, "the chain must not collapse into one concept"
        names = {c["name"] for c in merged}
        assert "A" in names
        assert "C" in names

    def test_a_genuinely_close_pair_still_merges(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two concepts above the threshold are merged, longest name wins."""
        vectors = {
            "Perceptual Evaluation of Speech Quality": [1.0, 0.0],
            "PESQ": [0.99, 0.01],
        }
        monkeypatch.setattr(
            "kompend.compilation.flows.assembly.make_provider",
            lambda _model: _FakeProvider(vectors),
        )
        concepts = [
            _concept("PESQ", "s1"),
            _concept("Perceptual Evaluation of Speech Quality", "s2"),
        ]

        merged = merge_similar_concepts(
            concepts,
            "fake/model",
            threshold=0.9,
            string_threshold=0.7,
            jaccard_min_token_len=1,
        )

        assert len(merged) == 1
        assert merged[0]["name"] == "Perceptual Evaluation of Speech Quality"
        assert sorted(merged[0]["sources"]) == ["s1", "s2"]


class TestReductionGuard:
    """A merge that would swallow the pool is refused, not applied."""

    def test_a_collapse_of_the_whole_pool_is_refused(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """Every concept identical → one cluster → refused at the guard."""
        vectors = {f"C{i}": [1.0, 0.0] for i in range(10)}
        monkeypatch.setattr(
            "kompend.compilation.flows.assembly.make_provider",
            lambda _model: _FakeProvider(vectors),
        )
        concepts = [_concept(f"C{i}", f"s{i}") for i in range(10)]

        with caplog.at_level(logging.WARNING):
            merged = merge_similar_concepts(
                concepts,
                "fake/model",
                threshold=0.9,
                string_threshold=0.7,
                jaccard_min_token_len=1,
                max_reduction=0.5,
            )

        assert len(merged) == 10, "the concepts must stay separate"
        assert any("refused" in record.message for record in caplog.records)

    def test_a_merge_within_budget_still_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One pair merged out of four is a 25% reduction — inside the budget."""
        vectors = {
            "A": [1.0, 0.0],
            "B": [0.99, 0.01],
            "C": [0.0, 1.0],
            "D": [-0.99, 0.01],
        }
        monkeypatch.setattr(
            "kompend.compilation.flows.assembly.make_provider",
            lambda _model: _FakeProvider(vectors),
        )
        concepts = [
            _concept("A", "s1"),
            _concept("B", "s2"),
            _concept("C", "s3"),
            _concept("D", "s4"),
        ]

        merged = merge_similar_concepts(
            concepts,
            "fake/model",
            threshold=0.9,
            string_threshold=0.7,
            jaccard_min_token_len=1,
            max_reduction=0.5,
        )

        # A and B merge (cos=1.0); C and D do not (cos=0.01). One concept
        # removed out of four is a 25% reduction — inside the 50% budget.
        assert len(merged) == 3
        merged_names = {c["name"] for c in merged}
        assert "C" in merged_names
        assert "D" in merged_names
        assert any(len(c["sources"]) == 2 for c in merged)

    def test_the_guard_is_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the guard the merge applies — the default is permissive."""
        vectors = {f"C{i}": [1.0, 0.0] for i in range(5)}
        monkeypatch.setattr(
            "kompend.compilation.flows.assembly.make_provider",
            lambda _model: _FakeProvider(vectors),
        )
        concepts = [_concept(f"C{i}", f"s{i}") for i in range(5)]

        merged = merge_similar_concepts(concepts, "fake/model", threshold=0.9, string_threshold=0.7, jaccard_min_token_len=1)

        assert len(merged) == 1


class TestSourceGuard:
    """A cluster spanning dozens of sources is refused."""

    def test_a_cluster_holding_too_many_sources_is_refused(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """Four sources merged into one, with a cap of two, is refused."""
        vectors = {
            "A": [1.0, 0.0],
            "B": [0.99, 0.01],
            "C": [0.98, 0.02],
            "D": [0.97, 0.03],
        }
        monkeypatch.setattr(
            "kompend.compilation.flows.assembly.make_provider",
            lambda _model: _FakeProvider(vectors),
        )
        concepts = [
            _concept("A", "s1"),
            _concept("B", "s2"),
            _concept("C", "s3"),
            _concept("D", "s4"),
        ]

        with caplog.at_level(logging.WARNING):
            merged = merge_similar_concepts(
                concepts,
                "fake/model",
                threshold=0.9,
                string_threshold=0.7,
                jaccard_min_token_len=1,
                max_sources=2,
            )

        assert len(merged) == 4, "the concepts must stay separate"
        assert any("refused" in record.message for record in caplog.records)


class TestStringFallback:
    """The offline path keeps working and honours the same guards."""

    def test_the_string_fallback_merges_close_names(self) -> None:
        """With no embedding model the rapidfuzz path merges close names."""
        concepts = [
            _concept("Narrowband speech", "s1"),
            _concept("Narrowband Speechn", "s2"),
            _concept("Perceptual Evaluation", "s3"),
        ]

        merged = merge_similar_concepts(
            concepts,
            "hash",
            threshold=0.82,
            string_threshold=0.7,
            jaccard_min_token_len=1,
        )

        assert len(merged) == 2
        assert any(len(c["sources"]) == 2 for c in merged)

    def test_the_string_fallback_honours_the_guard(self, caplog: pytest.LogCaptureFixture) -> None:
        """The guards apply to the offline path too, not only the embedding one."""
        concepts = [
            _concept("Narrowband speech", "s1"),
            _concept("Narrowband Speechn", "s2"),
        ]

        with caplog.at_level(logging.WARNING):
            merged = merge_similar_concepts(
                concepts,
                "hash",
                threshold=0.82,
                string_threshold=0.7,
                jaccard_min_token_len=1,
                max_sources=1,
            )

        assert len(merged) == 2
        assert any("refused" in record.message for record in caplog.records)
