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
import math
from typing import Any

import pytest

from kompend.compilation.flows.assembly import merge_similar_concepts

#: The text each concept is embedded as ("<name>. <definition>").
_VECTORS: dict[str, list[float]] = {}


class _FakeProvider:
    """An embedding provider that returns a vector per text by lookup.

    The vectors are handed in as text → vector, so a test can place two
    concepts at a chosen distance without needing a model.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> list[float]:
        return self._vectors[text]


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
        """Every concept identical → one cluster → the wholesale merge is refused.

        The refusal no longer means "nothing happens": the cluster is merged
        pairwise up to the pool's own budget, so ten identical concepts
        become five pairs — half the pool removed, which is exactly what
        ``max_reduction=0.5`` allows, and no concept holds more than two
        sources.
        """
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

        assert len(merged) == 5, "only the budget's share of the pool may go"
        assert all(len(concept["sources"]) == 2 for concept in merged)
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
        """Four sources merged into one, with a cap of two, is refused.

        The refusal is of the wholesale merge: the cluster is merged
        pairwise instead, and every merged concept stays within the cap of
        two sources.
        """
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
                similarity_margin=0.95,
            )

        assert len(merged) == 2, "the four concepts must not collapse into one"
        assert all(len(concept["sources"]) <= 2 for concept in merged)
        assert any("refused" in record.message for record in caplog.records)


class TestRefusalLogging:
    """A refusal logs the shape of the redundancy, not only the verdict."""

    def test_a_refusal_logs_the_cluster_size_distribution(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """The refusal names the largest clusters and the sources they hold.

        The verdict alone ("230 concepts / 5 sources") says nothing about
        what the merge wanted to do; the distribution shows the next run the
        shape of the redundancy around it.
        """
        vectors = {
            "A0": [1.0, 0.0],
            "A1": [0.99, 0.01],
            "A2": [0.98, 0.02],
            "A3": [0.97, 0.03],
            "B0": [0.0, 1.0],
            "B1": [0.01, 0.99],
            "B2": [0.02, 0.98],
            "C0": [-1.0, 0.0],
            "C1": [-0.99, 0.01],
        }
        monkeypatch.setattr(
            "kompend.compilation.flows.assembly.make_provider",
            lambda _model: _FakeProvider(vectors),
        )
        concepts = [_concept(name, f"s{index}") for index, name in enumerate(vectors)]

        with caplog.at_level(logging.WARNING):
            merge_similar_concepts(
                concepts,
                "fake/model",
                threshold=0.9,
                string_threshold=0.7,
                jaccard_min_token_len=1,
                max_reduction=0.25,
            )

        refusals = [record.message for record in caplog.records if "refused" in record.message]
        assert refusals, "the guard must log the refusal"
        assert any(
            "4 concepts/4 sources" in message and "3 concepts/3 sources" in message for message in refusals
        ), f"the refusal must carry the cluster-size distribution: {refusals}"
        assert any("pool 9 concepts in 3 clusters" in message for message in refusals)


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


class TestOversizedCluster:
    """A cluster too big for the guards merges its top pairs, not the whole."""

    def test_a_big_cluster_collapses_duplicates_and_keeps_sub_topics(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """The bc35f97e shape: four topics, three revisions each, one cluster.

        The corpus is several revisions of the same documentation, so the
        extraction finds every topic three times and the cluster holds
        twelve of the twenty concepts. Merging it whole would swallow the
        pool, so the guard refuses — but the refusal must not leave twelve
        near-duplicates behind: the top-K merge takes the revisions of each
        topic, the four topics stay four concepts, and the pool shrinks to a
        corpus-relative count.
        """
        vectors = _revisions_and_fillers()
        monkeypatch.setattr(
            "kompend.compilation.flows.assembly.make_provider",
            lambda _model: _FakeProvider(vectors),
        )
        concepts = [_concept(name, f"src-{name}") for name in vectors]

        with caplog.at_level(logging.WARNING):
            merged = merge_similar_concepts(
                concepts,
                "fake/model",
                threshold=0.82,
                string_threshold=0.7,
                jaccard_min_token_len=1,
                max_reduction=0.5,
                similarity_margin=0.95,
            )

        # Twelve concepts became four (one per topic) plus the eight
        # untouched fillers: the revisions collapsed, nothing else did.
        assert len(merged) == 12
        topics = [concept for concept in merged if len(concept["sources"]) > 1]
        assert len(topics) == 4, "each topic must collapse into exactly one concept"
        for topic in topics:
            assert len(topic["sources"]) == 3, "the three revisions of a topic merge"
            assert len({source.split("-", 1)[1].split(" r")[0] for source in topic["sources"]}) == 1, "a merged concept holds one topic only"
        assert any("refused" in record.message for record in caplog.records)


def _concept(name: str, *sources: str, definition: str = "") -> dict[str, Any]:
    """One concept candidate, as the extraction stage hands them over."""
    return {
        "name": name,
        "sources": list(sources),
        "definitions": [definition] if definition else [],
    }


def _revisions_and_fillers() -> dict[str, list[float]]:
    """Four topics (three revisions each) and eight concepts far from them.

    The four topics sit at a similarity of 0.85 from each other — above the
    clustering threshold, so they land in one cluster, and far below the
    revisions' 0.999, so the margin keeps them apart. The fillers sit in
    their own clusters and give the pool the size the budget is derived
    from.
    """
    bases = {
        "alpha": [1.0, 0.0, 0.0, 0.0],
        "beta": [0.85, 0.5268, 0.0, 0.0],
        "gamma": [0.85, 0.2416, 0.4681, 0.0],
        "delta": [0.85, 0.2416, 0.1477, 0.4442],
    }
    vectors: dict[str, list[float]] = {}
    for topic, base in bases.items():
        for revision in range(3):
            name = f"{topic} r{revision}"
            vectors[name] = _revision(base, revision) if revision else _unit(base)
    fillers = {
        "filler a": [0.0, 1.0, 0.0, 0.0],
        "filler b": [0.0, 0.0, 1.0, 0.0],
        "filler c": [0.0, 0.0, 0.0, 1.0],
        "filler d": [0.0, -1.0, 0.0, 0.0],
        "filler e": [0.0, 0.0, -1.0, 0.0],
        "filler f": [0.0, 0.0, 0.0, -1.0],
        "filler g": [0.0, 0.7071, 0.7071, 0.0],
        "filler h": [0.0, 0.7071, -0.7071, 0.0],
    }
    for name, vec in fillers.items():
        vectors[name] = _unit(vec)
    return vectors


def _revision(base: list[float], axis: int) -> list[float]:
    """A near-copy of *base*: the same topic, nudged in its own direction."""
    delta = [0.0] * len(base)
    delta[axis % len(base)] = 0.02
    return _unit([component + shift for component, shift in zip(base, delta, strict=True)])


def _unit(vec: list[float]) -> list[float]:
    """A unit vector in the fake embedding space."""
    norm = math.sqrt(sum(component * component for component in vec))
    return [component / norm for component in vec]


@pytest.fixture
def _patched_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fake provider wherever the merge looks it up."""
    monkeypatch.setattr(
        "kompend.compilation.flows.assembly.make_provider",
        lambda _model: _FakeProvider(_VECTORS),
    )
