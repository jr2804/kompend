"""Tests for knox.llm.models_dev — models.dev catalog context-window lookup.

All tests pass an explicit ``catalog=`` fixture (or monkeypatch ``_load_catalog``)
so NO network access occurs in the unit suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from knox.llm import models_dev
from knox.llm.models_dev import (
    _as_int,
    _match_provider_model,
    resolve_model_limits,
)


@pytest.fixture
def catalog() -> dict:
    """A minimal catalog mirroring the real models.dev shape."""
    return {
        "models": {
            "deepseek/deepseek-v4-flash": {"limit": {"context": 1_000_000, "output": 384_000}},
            "anthropic/claude-opus-4-5": {"limit": {"context": 200_000, "input": 195_000, "output": 64_000}},
            "google/gemma-4-9b": {"limit": {"context": 131_072, "output": 8192}},
            # Global-only (no provider entry for 'cohere') so the input-or-context
            # preference in the global table is exercised directly.
            "cohere/cohere-test": {"limit": {"context": 128_000, "input": 126_000, "output": 4000}},
        },
        "providers": {
            "ollama-cloud": {
                "models": {
                    "deepseek-v4-flash": {"limit": {"context": 1_048_576, "output": 1_048_576}},
                    "gemma4:31b": {"limit": {"context": 262_144, "output": 262_144}},
                    "no-limits": {"name": "model without a limit table"},
                },
            },
            "anthropic": {
                "models": {
                    "claude-opus-4-5": {"limit": {"context": 200_000, "output": 64_000}},
                },
            },
        },
    }


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Clear the module-level _catalog cache between tests."""
    models_dev._catalog = None


class TestResolveModelLimits:
    def test_provider_local_exact_match(self, catalog: dict) -> None:
        limits = resolve_model_limits("ollama-cloud/deepseek-v4-flash", catalog=catalog)
        assert limits == (1_048_576, 1_048_576)

    def test_provider_local_tag_prefix_match(self, catalog: dict) -> None:
        """':cloud' tag stripped, base 'gemma4' matches 'gemma4:31b'."""
        limits = resolve_model_limits("ollama-cloud/gemma4:cloud", catalog=catalog)
        assert limits == (262_144, 262_144)

    def test_global_lab_exact_match(self, catalog: dict) -> None:
        limits = resolve_model_limits("deepseek/deepseek-v4-flash", catalog=catalog)
        assert limits == (1_000_000, 384_000)

    def test_global_lab_stem_match(self, catalog: dict) -> None:
        """Unknown-lab prefix still matches by stem scan."""
        limits = resolve_model_limits("somelab/deepseek-v4-flash", catalog=catalog)
        assert limits == (1_000_000, 384_000)

    def test_input_preferred_over_context(self, catalog: dict) -> None:
        """Explicit input cap wins over the wider context window (global-only model)."""
        limits = resolve_model_limits("cohere/cohere-test", catalog=catalog)
        assert limits == (126_000, 4000)

    def test_model_not_in_catalog(self, catalog: dict) -> None:
        assert resolve_model_limits("nolab/no-such-model", catalog=catalog) is None

    def test_entry_without_limit_table(self, catalog: dict) -> None:
        assert resolve_model_limits("ollama-cloud/no-limits", catalog=catalog) is None

    def test_empty_catalog_returns_none(self) -> None:
        assert resolve_model_limits("anything/here", catalog={}) is None

    def test_missing_limit_fields_returns_none(self) -> None:
        bad = {"providers": {}, "models": {"foo/bar": {"limit": {"context": 1000}}}}
        assert resolve_model_limits("foo/bar", catalog=bad) is None

    def test_offline_load_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When _load_catalog fails to fetch, lookups return None without raising."""
        monkeypatch.setattr(models_dev, "_load_catalog", lambda: {})
        assert resolve_model_limits("ollama-cloud/deepseek-v4-flash") is None


class TestLoadCatalog:
    def test_network_failure_caches_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fetch failure sets _catalog to {} so retries are short-circuited."""
        monkeypatch.setattr(
            models_dev.httpx,
            "get",
            MagicMock(side_effect=models_dev.httpx.ConnectError("offline")),
        )
        assert models_dev._load_catalog() == {}
        # Second call must not re-invoke httpx.get (cache hit).
        assert models_dev._load_catalog() == {}

    def test_successful_fetch_caches_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"models": {"foo/bar": {"limit": {"context": 10, "output": 5}}}, "providers": {}}
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        monkeypatch.setattr(models_dev.httpx, "get", MagicMock(return_value=resp))
        assert models_dev._load_catalog() == payload

    def test_invalid_json_caches_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = MagicMock()
        resp.json.side_effect = ValueError("not json")
        resp.raise_for_status.return_value = None
        monkeypatch.setattr(models_dev.httpx, "get", MagicMock(return_value=resp))
        assert models_dev._load_catalog() == {}

    def test_non_mapping_payload_caches_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resp = MagicMock()
        resp.json.return_value = ["not", "a", "mapping"]
        resp.raise_for_status.return_value = None
        monkeypatch.setattr(models_dev.httpx, "get", MagicMock(return_value=resp))
        assert models_dev._load_catalog() == {}


class TestMatchProviderModel:
    def test_exact_stem(self) -> None:
        models = {"gemma4:31b": {"limit": {}}}
        assert _match_provider_model(models, "gemma4:31b", "gemma4") == {"limit": {}}

    def test_exact_base(self) -> None:
        models = {"gemma4": {"limit": {}}}
        assert _match_provider_model(models, "gemma4:cloud", "gemma4") == {"limit": {}}

    def test_tag_prefix(self) -> None:
        models = {"gemma4:31b": {"limit": {}}}
        assert _match_provider_model(models, "gemma4:cloud", "gemma4") == {"limit": {}}

    def test_no_match(self) -> None:
        assert _match_provider_model({"other": {}}, "gemma4:cloud", "gemma4") is None

    def test_non_mapping_input(self) -> None:
        assert _match_provider_model(None, "gemma4", "gemma4") is None
        assert _match_provider_model([], "gemma4", "gemma4") is None


class TestAsInt:
    def test_int(self) -> None:
        assert _as_int(1_048_576) == 1_048_576

    def test_numeric_str(self) -> None:
        assert _as_int("1_048_576".replace("_", "")) == 1_048_576

    def test_bool_rejected(self) -> None:
        assert _as_int(True) is None
        assert _as_int(False) is None

    def test_none_rejected(self) -> None:
        assert _as_int(None) is None

    def test_junk_rejected(self) -> None:
        assert _as_int("not-a-number") is None
        assert _as_int([1, 2]) is None
