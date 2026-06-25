"""Tests for the semantic cache module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from knox.search.cache import CacheEntry, SemanticCache, _cosine_similarity


class TestCosine:
    """_cosine_similarity returns the correct value."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_non_unit_vectors_normalised(self) -> None:
        a = [3.0, 4.0]
        b = [3.0, 4.0]
        assert _cosine_similarity(a, b) == pytest.approx(1.0)


class TestSemanticCacheNoProvider:
    """Without an embedding provider the cache always misses."""

    def test_lookup_returns_none(self) -> None:
        cache = SemanticCache(threshold=0.9, max_entries=10)
        assert cache.lookup("any query") is None

    def test_store_is_noop(self) -> None:
        cache = SemanticCache(threshold=0.9, max_entries=10)
        cache.store("q", "answer")
        assert cache.size == 0

    def test_clear_is_safe(self) -> None:
        cache = SemanticCache()
        cache.clear()
        assert cache.size == 0


class TestSemanticCacheWithProvider:
    """With a mock embedding provider the cache finds similar queries."""

    def test_exact_hit_above_threshold(self) -> None:
        cache = self._make_cache(threshold=0.9)
        cache.store("hello world", "my answer", [{"article": "a", "source": "s"}])
        result = cache.lookup("hello world")
        assert result is not None
        assert result.answer == "my answer"

    def test_miss_below_threshold(self) -> None:
        cache = self._make_cache(threshold=0.9)
        # "a" has embed [1.0, 1.0]; "long query here" has embed [15.0, 1.0]
        # cosine between those is not close to 1.0
        cache.store("a", "answer")
        result = cache.lookup("long query here")
        # similarity: dot(1,15)+dot(1,1) = 16, |a|=sqrt(2), |b|=sqrt(226)
        # sim ≈ 16/(1.414*15.03) ≈ 0.752 — below 0.9 threshold
        assert result is None

    def test_hit_count_increments(self) -> None:
        cache = self._make_cache(threshold=0.5)
        cache.store("test", "answer")
        cache.lookup("test")
        cache.lookup("test")
        entry = cache._entries[0]
        assert entry.hit_count == 3  # 1 initial + 2 lookups

    def test_eviction_at_capacity(self) -> None:
        cache = self._make_cache(threshold=0.9)
        cache._max_entries = 3
        for i in range(4):
            cache._entries.append(CacheEntry(query=f"q{i}", answer=f"a{i}", timestamp=float(i)))
        # store() triggers eviction when at capacity
        cache.store("new", "answer")
        assert cache.size <= 4  # evicted oldest, added new

    def test_clear_empties_cache(self) -> None:
        cache = self._make_cache()
        cache.store("q", "a")
        cache.clear()
        assert cache.size == 0

    def _make_cache(self, threshold: float = 0.9) -> SemanticCache:
        """Return a SemanticCache wired with a simple identity-like mock provider."""
        cache = SemanticCache(threshold=threshold, max_entries=5)
        # Inject a fake provider that returns the query length as a 2-vector
        provider = MagicMock()
        provider.embed.side_effect = lambda q: [float(len(q)), 1.0]
        cache._embedding_provider = provider
        return cache


class TestSemanticCachePersistence:
    """Cache round-trips through JSON file correctly."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache1 = SemanticCache(cache_dir=cache_dir, threshold=0.9, max_entries=10)
        # Inject provider
        provider = MagicMock()
        provider.embed.return_value = [1.0, 0.0]
        cache1._embedding_provider = provider
        cache1.store("question", "stored answer", [{"article": "A"}])

        # Load a second cache from same dir
        cache2 = SemanticCache(cache_dir=cache_dir, threshold=0.9, max_entries=10)
        assert cache2.size == 1
        entry = cache2._entries[0]
        assert entry.query == "question"
        assert entry.answer == "stored answer"
        assert entry.citations == [{"article": "A"}]

    def test_corrupt_cache_file_handled_gracefully(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "semantic_cache.json"
        cache_file.write_text("not valid json", encoding="utf-8")
        # Should not raise
        cache = SemanticCache(cache_dir=cache_dir)
        assert cache.size == 0
