"""Tests for knox.llm.models_dev — models.dev catalog + litellm context-window lookup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from knox.llm import models_dev

# Sample catalog entry shape (mirrors models.dev/catalog.json)
_GEMMA_ENTRY = {"limit": {"context": 262144, "output": 32768}}


class TestResolveModelLimits:
    def test_known_litellm_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Litellm knows gpt-4o → returns its limits."""
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: {})
        mock_info = MagicMock(return_value={"max_input_tokens": 128_000, "max_output_tokens": 16_384})
        monkeypatch.setattr(models_dev.litellm, "get_model_info", mock_info)
        limits = models_dev.resolve_model_limits("openai/gpt-4o")
        assert limits == (128_000, 16_384)

    def test_unknown_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: {})
        mock_info = MagicMock(side_effect=Exception("not found"))
        monkeypatch.setattr(models_dev.litellm, "get_model_info", mock_info)
        assert models_dev.resolve_model_limits("nolab/no-such-model") is None

    def test_litellm_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: {})
        mock_info = MagicMock(side_effect=ConnectionError("offline"))
        monkeypatch.setattr(models_dev.litellm, "get_model_info", mock_info)
        assert models_dev.resolve_model_limits("openai/gpt-4o") is None

    def test_backward_compat_catalog_param(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Old callers passing catalog= kwarg get None back (ignored param)."""
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: {})
        mock_info = MagicMock(side_effect=Exception)
        monkeypatch.setattr(models_dev.litellm, "get_model_info", mock_info)
        assert models_dev.resolve_model_limits("nolab/no-such-model", catalog={"old": "data"}) is None


class TestCatalogFetch:
    def test_network_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Network error → empty dict cache, no raise."""
        monkeypatch.setattr(models_dev, "_CACHE", {"catalog": None})
        monkeypatch.setattr(models_dev, "_cache_ttl_seconds", lambda: 0)

        def _raise(*a: object, **kw: object) -> None:
            raise models_dev.httpx.HTTPError("offline")

        monkeypatch.setattr(models_dev.httpx, "get", _raise)
        assert models_dev._fetch_catalog() == {}

    def test_cache_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Second call does not re-fetch."""
        monkeypatch.setattr(models_dev, "_CACHE", {"catalog": {"x": {}}})
        call_count = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(models_dev.httpx, "get", call_count)
        models_dev._fetch_catalog()
        call_count.assert_not_called()


class TestDiskCache:
    def test_disk_cache_hit_skips_remote(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fresh disk cache is loaded without a remote fetch."""
        cache_file = tmp_path / "models-dev-catalog.json"
        models_dev._save_disk_cache(cache_file, {"openai/gpt-4o": _GEMMA_ENTRY})
        monkeypatch.setattr(models_dev, "_CACHE", {"catalog": None})
        monkeypatch.setattr(models_dev, "_cache_path", lambda: cache_file)
        monkeypatch.setattr(models_dev, "_cache_ttl_seconds", lambda: 99999)
        remote = MagicMock(return_value={"should": "not be called"})
        monkeypatch.setattr(models_dev, "_fetch_remote", remote)

        catalog = models_dev._fetch_catalog()
        assert catalog == {"openai/gpt-4o": _GEMMA_ENTRY}
        remote.assert_not_called()

    def test_disk_cache_stale_triggers_remote(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stale disk cache (old timestamp) falls through to remote."""
        cache_file = tmp_path / "models-dev-catalog.json"
        # Write a cache with a 10-year-old timestamp
        cache_file.write_text(json.dumps({"fetched_at": 0, "models": {"old": {}}}))
        monkeypatch.setattr(models_dev, "_CACHE", {"catalog": None})
        monkeypatch.setattr(models_dev, "_cache_path", lambda: cache_file)
        monkeypatch.setattr(models_dev, "_cache_ttl_seconds", lambda: 3600)
        monkeypatch.setattr(models_dev, "_fetch_remote", lambda: {"fresh": _GEMMA_ENTRY})

        catalog = models_dev._fetch_catalog()
        assert catalog == {"fresh": _GEMMA_ENTRY}

    def test_ttl_zero_skips_disk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """KNOX_MODELS_CACHE_TTL=0 forces remote fetch even with fresh disk cache."""
        cache_file = tmp_path / "models-dev-catalog.json"
        models_dev._save_disk_cache(cache_file, {"stale": _GEMMA_ENTRY})
        monkeypatch.setattr(models_dev, "_CACHE", {"catalog": None})
        monkeypatch.setattr(models_dev, "_cache_path", lambda: cache_file)
        monkeypatch.setattr(models_dev, "_cache_ttl_seconds", lambda: 0.0)
        monkeypatch.setattr(models_dev, "_fetch_remote", lambda: {"fresh": _GEMMA_ENTRY})

        catalog = models_dev._fetch_catalog()
        assert catalog == {"fresh": _GEMMA_ENTRY}

    def test_refresh_catalog_clears_both(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """refresh_catalog() wipes in-memory + deletes disk file."""
        cache_file = tmp_path / "models-dev-catalog.json"
        models_dev._save_disk_cache(cache_file, {"x": {}})
        monkeypatch.setattr(models_dev, "_CACHE", {"catalog": {"x": {}}})
        monkeypatch.setattr(models_dev, "_cache_path", lambda: cache_file)

        models_dev.refresh_catalog()
        assert models_dev._CACHE["catalog"] is None
        assert not cache_file.exists()


class TestTtlEnvVar:
    def test_default_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KNOX_MODELS_CACHE_TTL", raising=False)
        assert models_dev._cache_ttl_seconds() == 168.0 * 3600

    def test_custom_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KNOX_MODELS_CACHE_TTL", "12")
        assert models_dev._cache_ttl_seconds() == 12 * 3600

    def test_zero_forces_refresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KNOX_MODELS_CACHE_TTL", "0")
        assert models_dev._cache_ttl_seconds() == 0.0

    def test_invalid_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KNOX_MODELS_CACHE_TTL", "not-a-number")
        assert models_dev._cache_ttl_seconds() == 168.0 * 3600


class TestCatalogMatch:
    def test_exact_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        catalog = {"openai/gpt-4o": _GEMMA_ENTRY}
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: catalog)
        monkeypatch.setattr(models_dev.litellm, "get_model_info", MagicMock(side_effect=Exception))
        limits = models_dev.resolve_model_limits("openai/gpt-4o")
        assert limits == (262144, 32768)

    def test_provider_local_stem_exact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider-local model that exists by its bare stem."""
        catalog = {"llama3.2": _GEMMA_ENTRY}
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: catalog)
        monkeypatch.setattr(models_dev.litellm, "get_model_info", MagicMock(side_effect=Exception))
        limits = models_dev.resolve_model_limits("ollama/llama3.2")
        assert limits == (262144, 32768)

    def test_fuzzy_gemma_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ollama-cloud/gemma4:31b → google/gemma-4-31b-it via normalized stem + size."""
        catalog = {"google/gemma-4-31b-it": _GEMMA_ENTRY, "google/gemma-4-E4B-it": {"limit": {"context": 131072}}}
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: catalog)
        monkeypatch.setattr(models_dev.litellm, "get_model_info", MagicMock(side_effect=Exception))
        limits = models_dev.resolve_model_limits("ollama-cloud/gemma4:31b")
        assert limits == (262144, 32768)

    def test_fuzzy_size_disambiguates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Size token prevents matching wrong variant."""
        catalog = {
            "google/gemma-4-31b-it": _GEMMA_ENTRY,
            "google/gemma-4-26b-a4b-it": {"limit": {"context": 262144, "output": 32768}},
        }
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: catalog)
        monkeypatch.setattr(models_dev.litellm, "get_model_info", MagicMock(side_effect=Exception))
        # query for 26b should NOT match 31b entry; both share base stem
        limits = models_dev.resolve_model_limits("ollama-cloud/gemma4:26b")
        assert limits == (262144, 32768)

    def test_no_match_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        catalog = {"google/gemma-4-31b-it": _GEMMA_ENTRY}
        monkeypatch.setattr(models_dev, "_fetch_catalog", lambda: catalog)
        monkeypatch.setattr(models_dev.litellm, "get_model_info", MagicMock(side_effect=Exception))
        assert models_dev.resolve_model_limits("nolab/totally-different") is None


class TestNormalizeStem:
    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("gemma4", "gemma-4"),
            ("gemma-4-31b-it", "gemma-4-31b-it"),
            ("Llama3.2", "llama-3-2"),
            ("deepseek_v4", "deepseek-v-4"),
            ("qwen2.5", "qwen-2-5"),
            ("model:tag", "model"),
        ],
    )
    def test_normalize(self, inp: str, expected: str) -> None:
        assert models_dev._normalize_stem(inp) == expected


class TestExtractSize:
    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("ollama-cloud/gemma4:31b", "31b"),
            ("google/gemma-4-31b-it", "31b"),
            ("llama3.2", None),
            ("qwen2.5:70b", "70b"),
        ],
    )
    def test_extract(self, inp: str, expected: str | None) -> None:
        assert models_dev._extract_size(inp) == expected
