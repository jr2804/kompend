"""Tests for the content-addressed CacheStore."""

from __future__ import annotations

from pathlib import Path

from knox.core.cache.store import CacheStore, make_cache_key
from knox.core.models import Chunk, ElementType, ResolvedChunk

# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_deterministic(self) -> None:
        a = make_cache_key({"a": "1", "b": "2"})
        b = make_cache_key({"b": "2", "a": "1"})
        assert a == b  # order-independent

    def test_different_values_different_keys(self) -> None:
        assert make_cache_key({"x": "1"}) != make_cache_key({"x": "2"})

    def test_different_deps_different_keys(self) -> None:
        assert make_cache_key({"a": "1"}) != make_cache_key({"b": "1"})

    def test_empty_deps(self) -> None:
        key = make_cache_key({})
        assert isinstance(key, str)
        assert len(key) == 16


# ---------------------------------------------------------------------------
# CacheStore — basic round-trip
# ---------------------------------------------------------------------------


class TestCacheStoreRoundTrip:
    def test_miss_on_empty(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        assert store.get("test", {"src": "abc"}) is None

    def test_round_trip(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        deps = {"src": "abc", "cfg": "123"}
        value = {"answer": 42, "items": [1, 2, 3]}
        store.set("test", deps, value)
        result = store.get("test", deps)
        assert result == value

    def test_miss_on_dep_change(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        store.set("test", {"src": "abc"}, "old value")
        # Same layer+key but dep changed → should miss
        assert store.get("test", {"src": "xyz"}) is None

    def test_miss_on_removed_dep(self, tmp_path: Path) -> None:
        """Caller no longer tracks a dep that the entry was created with."""
        store = CacheStore(tmp_path)
        store.set("test", {"a": "1", "b": "2"}, "val")
        # Same key implicitly — only loads if deps match.
        # Entry stored with {a,b} but caller only provides {a}.
        result = store.get("test", {"a": "1"})
        assert result is None  # stored dep "b" not in caller's deps


# ---------------------------------------------------------------------------
# CacheStore — type preservation
# ---------------------------------------------------------------------------


class TestCacheStoreTypes:
    def test_list(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        val = [{"name": "foo"}, {"name": "bar"}]
        store.set("test", {"d": "1"}, val)
        assert store.get("test", {"d": "1"}) == val

    def test_none(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        store.set("test", {"d": "1"}, None)
        assert store.get("test", {"d": "1"}) is None

    def test_dataclass_round_trip(self, tmp_path: Path) -> None:
        """JSON codec preserves dataclasses, Path, StrEnum, bytes and tuple."""
        store = CacheStore(tmp_path)
        chunk = Chunk(
            id="c1",
            text="hello",
            element_types=[ElementType.TEXT],
            source_elements=[0],
            source_file=Path("x.pdf"),
            reading_order=(1, 2),
            metadata={"image_file": Path("img.png"), "raw": b"deadbeef"},
        )
        obj = ResolvedChunk(chunk=chunk)
        store.set("test", {"d": "1"}, obj)
        result = store.get("test", {"d": "1"})
        assert isinstance(result, ResolvedChunk)
        assert result == obj
        assert result.chunk.source_file == Path("x.pdf")
        assert result.chunk.reading_order == (1, 2)
        assert result.chunk.metadata["image_file"] == Path("img.png")
        assert result.chunk.metadata["raw"] == b"deadbeef"

    def test_custom_object(self, tmp_path: Path) -> None:
        """JSON codec handles plain nested dicts."""
        store = CacheStore(tmp_path)
        obj = {"name": "hello", "nested": {"a": [1, 2, 3]}}
        store.set("test", {"d": "1"}, obj)
        assert store.get("test", {"d": "1"}) == obj


# ---------------------------------------------------------------------------
# CacheStore — has / invalidate
# ---------------------------------------------------------------------------


class TestCacheStoreAdmin:
    def test_has(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        deps = {"src": "abc"}
        assert not store.has("test", deps)
        store.set("test", deps, "data")
        assert store.has("test", deps)

    def test_invalidate_layer(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        store.set("layer_a", {"k": "1"}, "a")
        store.set("layer_b", {"k": "2"}, "b")
        store.invalidate("layer_a")
        assert store.get("layer_a", {"k": "1"}) is None
        assert store.get("layer_b", {"k": "2"}) == "b"

    def test_invalidate_all(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        store.set("layer_a", {"k": "1"}, "a")
        store.set("layer_b", {"k": "2"}, "b")
        store.invalidate()
        assert store.get("layer_a", {"k": "1"}) is None
        assert store.get("layer_b", {"k": "2"}) is None

    def test_invalidate_empty(self, tmp_path: Path) -> None:
        """Invalidate on an empty cache should not raise."""
        store = CacheStore(tmp_path)
        store.invalidate()
        store.invalidate("nonexistent")


# ---------------------------------------------------------------------------
# CacheStore — error tolerance
# ---------------------------------------------------------------------------


class TestCacheStoreErrors:
    def test_corrupt_file_is_miss(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        deps = {"src": "abc"}
        store.set("test", deps, "data")
        # Corrupt the JSON cache file
        path = store._resolve("test", make_cache_key(deps))
        path.write_bytes(b"not valid json")
        assert store.get("test", deps) is None

    def test_non_existent_layer(self, tmp_path: Path) -> None:
        store = CacheStore(tmp_path)
        assert store.get("nonexistent", {"k": "1"}) is None
