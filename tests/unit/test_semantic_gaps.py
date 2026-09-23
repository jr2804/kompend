"""Tests for the semantic-gap embedding-vector cache keying.

Covers ``_load_vector_cache`` / ``_save_vector_cache`` in
``knox.lint.semantic_gaps`` directly — no embedding provider is involved.
The cache must be keyed by embedding model, so a model change is a full
miss (stale cross-dimension vectors must never be compared).

Also covers the contextual-context consumer: a chunk's
``contextual_context`` is prepended to the embedded text (shared helper
``knox.core.text.embedding_text.embed_text``), and the cache key hashes
that *composed* text so enriched and un-enriched vectors never collide.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from knox.core.text.embedding_text import embed_text
from knox.lint import semantic_gaps
from knox.lint.semantic_gaps import _load_vector_cache, _save_vector_cache
from knox.rag.zvec_store import EmbeddingProvider


class TestVectorCacheKeying:
    def test_same_model_roundtrip_is_a_hit(self, tmp_path: Path) -> None:
        path = _cache_file(tmp_path)
        vectors = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
        _save_vector_cache(path, vectors, "model-x")
        assert _load_vector_cache(path, "model-x") == vectors

    def test_different_model_is_a_full_miss(self, tmp_path: Path) -> None:
        path = _cache_file(tmp_path)
        _save_vector_cache(path, {"a": [1.0, 2.0]}, "model-x")
        assert _load_vector_cache(path, "model-y") == {}

    def test_same_model_ignores_dim_field(self, tmp_path: Path) -> None:
        """``dim`` is metadata; the model is the load-time invalidation key.

        A same-model payload with an inconsistent ``dim`` is still returned —
        a genuine dimension change surfaces later as ``cosine``'s
        length-mismatch ``ValueError``, never as a silent wrong score.
        """
        path = _cache_file(tmp_path)
        path.write_text(
            json.dumps({"model": "model-x", "dim": 99, "vectors": {"a": [1.0, 2.0]}}),
            encoding="utf-8",
        )
        assert _load_vector_cache(path, "model-x") == {"a": [1.0, 2.0]}

    def test_legacy_model_less_shape_is_a_miss(self, tmp_path: Path) -> None:
        path = _cache_file(tmp_path)
        path.write_text(json.dumps({"a": [1.0, 2.0]}), encoding="utf-8")
        assert _load_vector_cache(path, "model-x") == {}

    def test_corrupt_json_is_a_miss(self, tmp_path: Path) -> None:
        path = _cache_file(tmp_path)
        path.write_text("{not json", encoding="utf-8")
        assert _load_vector_cache(path, "model-x") == {}

    def test_missing_file_is_a_miss(self, tmp_path: Path) -> None:
        assert _load_vector_cache(_cache_file(tmp_path), "model-x") == {}

    def test_saved_file_uses_model_dim_vectors_shape(self, tmp_path: Path) -> None:
        path = _cache_file(tmp_path)
        _save_vector_cache(path, {"a": [1.0, 2.0, 3.0]}, "model-x")
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "model": "model-x",
            "dim": 3,
            "vectors": {"a": [1.0, 2.0, 3.0]},
        }


class _RecordingProvider(EmbeddingProvider):
    """Deterministic provider that records every embedded string."""

    def __init__(self, dim: int = 3) -> None:
        self.calls: list[str] = []
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]


class TestGapCacheKeyIncludesContext:
    def test_enriched_chunk_hashes_the_composed_text(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _RecordingProvider()
        monkeypatch.setattr(semantic_gaps, "make_provider", lambda _model: provider)

        _write_chunk_cache(tmp_path, text="chunk body", context="situating context")
        semantic_gaps.verify_topics(["topic"], tmp_path, "model-x")

        expected_key = hashlib.sha256(embed_text("chunk body", "situating context").encode("utf-8")).hexdigest()[:16]
        raw_key = hashlib.sha256(b"chunk body").hexdigest()[:16]
        vectors = _gap_vectors(tmp_path)
        assert expected_key in vectors
        assert raw_key not in vectors

    def test_enriched_and_unenriched_keys_differ(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _RecordingProvider()
        monkeypatch.setattr(semantic_gaps, "make_provider", lambda _model: provider)

        enriched_dir = tmp_path / "enriched"
        _write_chunk_cache(enriched_dir, text="chunk body", context="ctx")
        semantic_gaps.verify_topics(["topic"], enriched_dir, "model-x")

        plain_dir = tmp_path / "plain"
        _write_chunk_cache(plain_dir, text="chunk body", context="")
        semantic_gaps.verify_topics(["topic"], plain_dir, "model-x")

        assert set(_gap_vectors(enriched_dir)) != set(_gap_vectors(plain_dir))


def _cache_file(tmp_path: Path) -> Path:
    return tmp_path / "gap_chunk_vectors.json"


def _write_chunk_cache(base_dir: Path, *, text: str, context: str) -> None:
    chunk_dir = base_dir / ".knox" / "cache" / "chunk"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "value": {
            "source_file": "doc.md",
            "nodes": {"n1": {"chunk": {"text": text}, "contextual_context": context}},
        }
    }
    (chunk_dir / "doc.json").write_text(json.dumps(payload), encoding="utf-8")


def _gap_vectors(base_dir: Path) -> dict[str, list[float]]:
    path = base_dir / ".knox" / "cache" / "gap_chunk_vectors.json"
    return json.loads(path.read_text(encoding="utf-8"))["vectors"]
