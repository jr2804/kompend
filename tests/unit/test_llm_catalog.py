"""Tests for the model-catalogue snapshot cache (``kompend.llm.catalog``)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import llmdb
import pytest

from kompend.llm import catalog


class TestCacheRoundTrip:
    def test_snapshot_is_written_and_read_back(self, isolated_cache: Path) -> None:
        fetched = catalog.CatalogSnapshot(fetched_at=datetime.now(UTC), providers={"acme": _provider("acme")})
        catalog._write_cache(isolated_cache, fetched)

        loaded = catalog._read_cache(isolated_cache)

        assert loaded is not None
        assert loaded.fetched_at == fetched.fetched_at
        assert loaded.provider("acme") is not None
        assert loaded.model("acme", "m1") is not None

    def test_unreadable_cache_is_ignored(self, isolated_cache: Path) -> None:
        isolated_cache.write_text("not json", encoding="utf-8")

        assert catalog._read_cache(isolated_cache) is None

    def test_missing_cache_is_none(self, isolated_cache: Path) -> None:
        assert catalog._read_cache(isolated_cache) is None


class TestLookups:
    def test_provider_and_model_come_from_the_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        snapshot = catalog.CatalogSnapshot(fetched_at=datetime.now(UTC), providers={"acme": _provider("acme")})
        catalog._install(snapshot)

        assert catalog.provider("acme") is not None
        assert catalog.provider("other") is None
        assert catalog.model("acme", "m1") is not None
        assert catalog.model("acme", "nope") is None
        assert catalog.model("other", "m1") is None

    def test_no_catalogue_yields_no_lookups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        catalog._install(None)
        monkeypatch.setattr(catalog, "_read_cache", lambda path: None)
        monkeypatch.setattr(catalog, "_fetch", lambda: None)

        assert catalog.provider("acme") is None
        assert catalog.model("acme", "m1") is None
        assert catalog.snapshot() is None


class TestFreshness:
    def test_cache_from_the_cache_file_counts_as_fresh(self, isolated_cache: Path) -> None:
        fetched = catalog.CatalogSnapshot(fetched_at=datetime.now(UTC), providers={"acme": _provider("acme")})
        catalog._write_cache(isolated_cache, fetched)

        assert catalog.is_stale() is False
        assert catalog.ensure_fresh() is False

    def test_old_snapshot_is_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        old = datetime.now(UTC) - timedelta(days=catalog.CACHE_TTL_HOURS + 1)
        snapshot = catalog.CatalogSnapshot(fetched_at=old, providers={"acme": _provider("acme")})
        catalog._install(snapshot)

        assert catalog.is_stale() is True
        assert catalog.age() is not None

    def test_missing_snapshot_is_stale(self, monkeypatch: pytest.MonkeyPatch, isolated_cache: Path) -> None:
        catalog._install(None)
        monkeypatch.setattr(catalog, "_fetch", lambda: None)

        assert catalog.is_stale() is True
        assert catalog.age() is None


class TestRefresh:
    def test_refresh_stores_the_downloaded_snapshot(self, monkeypatch: pytest.MonkeyPatch, isolated_cache: Path) -> None:
        fresh = catalog.CatalogSnapshot(fetched_at=datetime.now(UTC), providers={"acme": _provider("acme")})
        monkeypatch.setattr(catalog, "_fetch", lambda: fresh)

        assert catalog.refresh() is True

        assert catalog.provider("acme") is not None
        assert json.loads(isolated_cache.read_text(encoding="utf-8"))[catalog._FETCHED_KEY]

    def test_failed_refresh_keeps_the_stale_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        old = datetime.now(UTC) - timedelta(days=catalog.CACHE_TTL_HOURS + 1)
        snapshot = catalog.CatalogSnapshot(fetched_at=old, providers={"acme": _provider("acme")})
        catalog._install(snapshot)
        monkeypatch.setattr(catalog, "_fetch", lambda: None)

        assert catalog.refresh() is False

        assert catalog.provider("acme") is not None

    def test_ensure_fresh_skips_a_fresh_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        snapshot = catalog.CatalogSnapshot(fetched_at=datetime.now(UTC), providers={"acme": _provider("acme")})
        catalog._install(snapshot)
        monkeypatch.setattr(catalog, "_fetch", lambda: pytest.fail("must not download a fresh snapshot"))

        assert catalog.ensure_fresh() is False


def _provider(provider_id: str, *, api: str = "https://example.test/v1") -> llmdb.Provider:
    """Build a minimal provider payload the cache can round-trip."""
    return llmdb.Provider.model_validate(
        {
            "id": provider_id,
            "name": provider_id,
            "api": api,
            "env": ["EXAMPLE_API_KEY"],
            "models": {
                "m1": {
                    "id": "m1",
                    "name": "m1",
                    "attachment": False,
                    "reasoning": False,
                    "tool_call": True,
                    "release_date": "2025-01-01",
                    "last_updated": "2025-01-01",
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "open_weights": False,
                    "limit": {"context": 1000, "output": 500},
                }
            },
        }
    )


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point the cache at a temp file and drop the in-process snapshot."""
    cache_file = tmp_path / "models.json"
    monkeypatch.setattr(catalog, "cache_path", lambda: cache_file)
    catalog.reset()
    yield cache_file
    catalog.reset()
