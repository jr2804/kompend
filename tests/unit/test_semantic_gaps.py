"""Tests for the semantic-gap embedding-vector cache keying.

Covers ``_load_vector_cache`` / ``_save_vector_cache`` in
``knox.lint.semantic_gaps`` directly — no embedding provider is involved.
The cache must be keyed by embedding model, so a model change is a full
miss (stale cross-dimension vectors must never be compared).
"""

from __future__ import annotations

import json
from pathlib import Path

from knox.lint.semantic_gaps import _load_vector_cache, _save_vector_cache


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


def _cache_file(tmp_path: Path) -> Path:
    return tmp_path / "gap_chunk_vectors.json"
