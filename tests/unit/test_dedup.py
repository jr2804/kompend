"""Tests for content-level duplicate detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from knox.preprocessing.dedup.base import DedupVerdict, classify_score
from knox.preprocessing.dedup.embedding_matcher import EmbeddingMatcher
from knox.preprocessing.dedup.minhash_matcher import MinhashMatcher
from knox.preprocessing.dedup.orchestrator import DedupOrchestrator
from knox.preprocessing.dedup.store import DedupStore
from knox.rag.zvec_store import PydanticAIEmbeddingProvider
from tests.conftest import DeterministicTestEmbeddingProvider

# ---------------------------------------------------------------------------
# classify_score
# ---------------------------------------------------------------------------


class TestClassifyScore:
    """Tests for the verdict classifier."""

    def test_identical_at_threshold(self) -> None:
        assert classify_score(0.95, 0.95, 0.80) == DedupVerdict.IDENTICAL

    def test_identical_above_threshold(self) -> None:
        assert classify_score(0.98, 0.95, 0.80) == DedupVerdict.IDENTICAL

    def test_similar_at_threshold(self) -> None:
        assert classify_score(0.80, 0.95, 0.80) == DedupVerdict.SIMILAR

    def test_similar_between_thresholds(self) -> None:
        assert classify_score(0.85, 0.95, 0.80) == DedupVerdict.SIMILAR

    def test_different_below_similar(self) -> None:
        assert classify_score(0.70, 0.95, 0.80) == DedupVerdict.DIFFERENT

    def test_different_zero(self) -> None:
        assert classify_score(0.0, 0.95, 0.80) == DedupVerdict.DIFFERENT

    def test_custom_thresholds(self) -> None:
        assert classify_score(0.90, 0.90, 0.50) == DedupVerdict.IDENTICAL
        assert classify_score(0.60, 0.90, 0.50) == DedupVerdict.SIMILAR
        assert classify_score(0.30, 0.90, 0.50) == DedupVerdict.DIFFERENT


# ---------------------------------------------------------------------------
# ContentStore (base)
# ---------------------------------------------------------------------------


class TestContentStore:
    """Tests for the base ContentStore persistence layer."""

    def test_load_missing_file(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.load()
        assert len(store) == 0

    def test_save_and_load(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.load()
        store["doc-1"] = {"sha256": "abc"}
        store.save()

        store2 = _make_store(tmp_path)
        store2.load()
        assert store2.get("doc-1") == {"sha256": "abc"}
        assert store2.is_fresh()

    def test_freshness_stamp_mismatch(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.load()
        store["doc-1"] = {"sha256": "abc"}
        store.save()

        # Different config → different stamp → not fresh
        store2 = _make_store(tmp_path, config={"dedup.matchers": "embedding"})
        store2.load()
        assert not store2.is_fresh()

    def test_corrupt_file_starts_fresh(self, tmp_path: Path) -> None:
        db_file = tmp_path / "store.json"
        db_file.write_text("not json", encoding="utf-8")
        store = DedupStore(db_file)
        store.load()
        assert len(store) == 0
        assert store._data == {}


# ---------------------------------------------------------------------------
# EmbeddingMatcher
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding_matcher(**kwargs: object) -> EmbeddingMatcher:
    """Create an EmbeddingMatcher with a deterministic test provider."""
    model_name: str = kwargs.get("model_name", "all-MiniLM-L6-v2")  # type: ignore[assignment]
    dim: int = kwargs.get("dim", 384)  # type: ignore[assignment]
    chunk_size: int = kwargs.get("chunk_size", 512)  # type: ignore[assignment]
    provider = DeterministicTestEmbeddingProvider(dim=dim)
    return EmbeddingMatcher(
        model_name=model_name,
        dim=dim,
        provider=provider,
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# EmbeddingMatcher
# ---------------------------------------------------------------------------


class TestEmbeddingMatcher:
    """Tests for the embedding-based matcher."""

    def test_name(self) -> None:
        assert _make_embedding_matcher().name == "embedding"

    def test_identical_texts_max_score(self) -> None:
        matcher = _make_embedding_matcher()
        text = "Machine learning is a subset of artificial intelligence."
        fp_a = matcher.fingerprint(text)
        fp_b = matcher.fingerprint(text)
        score = matcher.score(fp_a, fp_b)
        assert score > 0.99

    def test_different_texts_low_score(self) -> None:
        matcher = _make_embedding_matcher()
        fp_a = matcher.fingerprint("Machine learning is a subset of artificial intelligence.")
        fp_b = matcher.fingerprint("Quantum computing exploits superposition and entanglement.")
        score = matcher.score(fp_a, fp_b)
        assert score < 0.6

    def test_empty_text(self) -> None:
        matcher = _make_embedding_matcher()
        fp = matcher.fingerprint("")
        assert all(v == 0.0 for v in fp)

    def test_serialize_roundtrip(self) -> None:
        matcher = _make_embedding_matcher()
        text = "This is a test document for serialization roundtrip."
        fp = matcher.fingerprint(text)
        data = matcher.serialize(fp)
        restored = matcher.deserialize(data)
        score = matcher.score(fp, restored)
        assert score > 0.99

    @pytest.mark.skip(reason="requires a real embedding model for semantic mean-pooling quality")
    def test_long_document_mean_pooled(self) -> None:
        """Long documents should produce stable mean-pooled vectors."""
        matcher = _make_embedding_matcher(chunk_size=5)  # small chunk to force splitting
        short_text = "word " * 10
        long_text = "word " * 100
        fp_short = matcher.fingerprint(short_text)
        fp_long = matcher.fingerprint(long_text)
        score = matcher.score(fp_short, fp_long)
        # Both are all "word" repeated — should be very similar
        # Tiny chunks introduce noise, so threshold is relaxed
        assert score > 0.75


# ---------------------------------------------------------------------------
# MinhashMatcher
# ---------------------------------------------------------------------------


class TestMinhashMatcher:
    """Tests for the MinHash-based matcher."""

    def test_name(self) -> None:
        assert MinhashMatcher().name == "minhash"

    def test_identical_texts_max_score(self) -> None:
        matcher = MinhashMatcher()
        text = "Machine learning is a subset of artificial intelligence."
        fp_a = matcher.fingerprint(text)
        fp_b = matcher.fingerprint(text)
        score = matcher.score(fp_a, fp_b)
        assert score > 0.99

    def test_similar_texts_high_score(self) -> None:
        matcher = MinhashMatcher(ngram=2)
        fp_a = matcher.fingerprint("deep learning neural networks natural language")
        fp_b = matcher.fingerprint("deep learning neural networks image recognition")
        score = matcher.score(fp_a, fp_b)
        # 3 of 5 n-grams overlap (60%) → score should be > 0.35
        assert score > 0.35

    def test_different_texts_low_score(self) -> None:
        matcher = MinhashMatcher()
        fp_a = matcher.fingerprint("Machine learning is a subset of artificial intelligence.")
        fp_b = matcher.fingerprint("Quantum computing exploits superposition and entanglement.")
        score = matcher.score(fp_a, fp_b)
        assert score < 0.3

    def test_empty_text(self) -> None:
        matcher = MinhashMatcher()
        fp = matcher.fingerprint("")
        # An empty MinHash has all hashvalues at max
        assert fp.jaccard(fp) > 0.99

    def test_serialize_roundtrip(self) -> None:
        matcher = MinhashMatcher()
        text = "This is a test document for serialization roundtrip."
        fp = matcher.fingerprint(text)
        data = matcher.serialize(fp)
        restored = matcher.deserialize(data)
        score = matcher.score(fp, restored)
        assert score > 0.99

    def test_lsh_insert_and_query(self) -> None:
        matcher = MinhashMatcher(lsh_threshold=0.3)
        matcher.init_lsh()

        fp_a = matcher.fingerprint("machine learning neural networks")
        fp_b = matcher.fingerprint("machine learning deep learning")
        fp_c = matcher.fingerprint("quantum computing error correction")

        matcher.lsh_insert("doc-a", fp_a)
        matcher.lsh_insert("doc-b", fp_b)
        matcher.lsh_insert("doc-c", fp_c)

        candidates = matcher.lsh_query(fp_a)
        # LSH is probabilistic; doc-a should at least find itself
        assert "doc-a" in candidates


# ---------------------------------------------------------------------------
# DedupOrchestrator
# ---------------------------------------------------------------------------


class TestDedupOrchestrator:
    """Tests for the full dedup orchestrator."""

    def test_disabled_returns_different(self, tmp_path: Path) -> None:
        orch = DedupOrchestrator(tmp_path, config={"dedup.enabled": False})
        results = orch.process_batch([("doc-1", "some text")])
        assert results["doc-1"].verdict == DedupVerdict.DIFFERENT

    def test_litellm_prefix_translated_to_pydantic_ai_format(self, tmp_path: Path) -> None:
        """Regression: ``litellm:`` prefix must be translated to pydantic-ai's
        ``provider:model`` format so the Embedder can resolve the provider.
        """
        orch = DedupOrchestrator(
            tmp_path,
            config={
                "dedup.enabled": True,
                "dedup.matchers": "embedding",
                "dedup.model": "litellm:ollama/granite-embedding:278m",
                "compile.embedding.dim": 384,
            },
        )
        matchers, _ = orch._build_matchers()  # noqa: SLF001 — introspect build
        emb_matcher = next(m for m in matchers if m.name == "embedding")
        assert isinstance(emb_matcher._provider, PydanticAIEmbeddingProvider)  # noqa: SLF001
        # Prefix must be translated to pydantic-ai format
        assert emb_matcher._provider._model_string == "ollama:granite-embedding:278m"  # noqa: SLF001

    def test_identical_docs_detected(self, tmp_path: Path) -> None:
        orch = DedupOrchestrator(
            tmp_path,
            config={
                "dedup.enabled": True,
                "dedup.matchers": "minhash",
                "dedup.threshold_identical": 0.95,
            },
        )
        text = "This is a paper about machine learning and neural networks."
        results = orch.process_batch([("doc-a", text), ("doc-b", text)])
        assert results["doc-a"].verdict == DedupVerdict.IDENTICAL
        assert results["doc-b"].verdict == DedupVerdict.IDENTICAL

    def test_different_docs_not_flagged(self, tmp_path: Path) -> None:
        orch = DedupOrchestrator(
            tmp_path,
            config={
                "dedup.enabled": True,
                "dedup.matchers": "minhash",
                "dedup.threshold_identical": 0.95,
                "dedup.threshold_similar": 0.80,
            },
        )
        results = orch.process_batch(
            [
                ("doc-a", "Machine learning is a subset of AI."),
                ("doc-b", "Quantum computing exploits superposition."),
            ]
        )
        assert results["doc-a"].verdict == DedupVerdict.DIFFERENT
        assert results["doc-b"].verdict == DedupVerdict.DIFFERENT

    def test_closest_match_tracked(self, tmp_path: Path) -> None:
        orch = DedupOrchestrator(
            tmp_path,
            config={
                "dedup.enabled": True,
                "dedup.matchers": "minhash",
                "dedup.threshold_identical": 0.95,
                "dedup.threshold_similar": 0.80,
            },
        )
        text = "Machine learning and AI research advances."
        results = orch.process_batch(
            [
                ("doc-a", text),
                ("doc-b", text),
                ("doc-c", "Quantum computing is completely different."),
            ]
        )
        assert results["doc-a"].closest_match == "doc-b"
        assert results["doc-a"].closest_score > 0.95

    def test_persistent_store_survives_across_batches(self, tmp_path: Path) -> None:
        orch = DedupOrchestrator(
            tmp_path,
            config={
                "dedup.enabled": True,
                "dedup.matchers": "minhash",
                "dedup.threshold_identical": 0.95,
            },
        )
        orch.process_batch([("doc-1", "Paper about machine learning.")])

        # Second batch gets a near-duplicate — new orchestrator loads old store
        orch2 = DedupOrchestrator(
            tmp_path,
            config={
                "dedup.enabled": True,
                "dedup.matchers": "minhash",
                "dedup.threshold_identical": 0.95,
            },
        )
        results = orch2.process_batch([("doc-2", "Paper about machine learning.")])
        assert results["doc-2"].verdict == DedupVerdict.IDENTICAL
        assert results["doc-2"].closest_match == "doc-1"

    def test_combined_matchers_max_score_used(self, tmp_path: Path) -> None:
        """When both matchers are active, max of both scores is used."""
        orch = DedupOrchestrator(
            tmp_path,
            config={
                "dedup.enabled": True,
                "dedup.matchers": "embedding,minhash",
                "dedup.threshold_identical": 0.95,
            },
        )
        text = "Machine learning neural networks deep learning AI."
        results = orch.process_batch([("doc-a", text), ("doc-b", text)])
        r = results["doc-a"]
        assert r.verdict == DedupVerdict.IDENTICAL
        # Both scores should be stored
        assert r.embedding_score is not None
        assert r.minhash_score is not None

    def test_store_persistence(self, tmp_path: Path) -> None:
        store = DedupStore(tmp_path / "fingerprints.json")
        store.load()
        assert store.slugs == []

        store["doc-1"] = {"closest_match": None, "closest_score": 0.0}
        store.save()

        store2 = DedupStore(tmp_path / "fingerprints.json")
        store2.load()
        assert "doc-1" in store2
        assert store2.get("doc-1", {}).get("closest_score") == 0.0

    # ── fingerprint() + compare() incremental API ───────────────────

    def test_fingerprint_then_compare_matches_process_batch(self, tmp_path: Path) -> None:
        """Incremental fingerprint+compare must yield same verdicts as process_batch."""
        config = {
            "dedup.enabled": True,
            "dedup.matchers": "minhash",
            "dedup.threshold_identical": 0.95,
        }
        text = "This is a paper about machine learning and neural networks."

        # Path A: incremental
        orch_inc = DedupOrchestrator(tmp_path / "inc", config=config)
        fps = {}
        fps["doc-a"] = orch_inc.fingerprint("doc-a", text)
        fps["doc-b"] = orch_inc.fingerprint("doc-b", text)
        results_inc = orch_inc.compare(fps)

        # Path B: convenience
        orch_batch = DedupOrchestrator(tmp_path / "batch", config=config)
        results_batch = orch_batch.process_batch([("doc-a", text), ("doc-b", text)])

        assert results_inc["doc-a"].verdict == results_batch["doc-a"].verdict
        assert results_inc["doc-a"].closest_score == results_batch["doc-a"].closest_score

    def test_fingerprint_returns_empty_when_disabled(self, tmp_path: Path) -> None:
        """fingerprint() returns empty dict when dedup is disabled."""
        orch = DedupOrchestrator(tmp_path, config={"dedup.enabled": False})
        assert orch.fingerprint("doc-1", "some text") == {}

    def test_embedding_error_is_fatal(self, tmp_path: Path) -> None:
        """Embedding errors must propagate — not be swallowed silently.

        Regression for the provider-prefix bug and the try/except that
        hid configuration errors.
        """
        orch = DedupOrchestrator(
            tmp_path,
            config={
                "dedup.enabled": True,
                "dedup.matchers": "embedding",
                "dedup.model": "litellm:ollama/granite-embedding:278m",
                "compile.embedding.dim": 384,
            },
        )

        # Simulate a provider error (as would happen with a misconfigured model)
        with (
            patch.object(
                EmbeddingMatcher,
                "fingerprint",
                side_effect=RuntimeError("BadRequestError: LLM Provider NOT provided"),
            ),
            pytest.raises(RuntimeError, match="BadRequestError"),
        ):
            orch.fingerprint("doc-1", "some text")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, config: dict | None = None) -> DedupStore:
    return DedupStore(
        tmp_path / "dedup" / "fingerprints.json",
        config=config or {},
    )
