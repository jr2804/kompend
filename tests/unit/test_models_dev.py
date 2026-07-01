"""Tests for knox.llm.models_dev — litellm-backed context-window lookup."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from knox.llm import models_dev


class TestResolveModelLimits:
    def test_known_litellm_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Litellm knows gpt-4o → returns its limits."""
        mock_info = MagicMock(return_value={"max_input_tokens": 128_000, "max_output_tokens": 16_384})
        monkeypatch.setattr(models_dev.litellm, "get_model_info", mock_info)
        limits = models_dev.resolve_model_limits("openai/gpt-4o")
        assert limits == (128_000, 16_384)

    def test_fallback_match(self) -> None:
        """Embedding models in _FALLBACK return hardcoded limits."""
        limits = models_dev.resolve_model_limits("ollama/nomic-embed-text")
        assert limits == (8192, 8192)

    def test_fallback_base_match(self) -> None:
        limits = models_dev.resolve_model_limits("ollama/granite-embedding:278m")
        assert limits == (8192, 8192)

    def test_unknown_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown model returns None without raising."""
        mock_info = MagicMock(side_effect=Exception("not found"))
        monkeypatch.setattr(models_dev.litellm, "get_model_info", mock_info)
        assert models_dev.resolve_model_limits("nolab/no-such-model") is None

    def test_litellm_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Network error during lookup returns None."""
        mock_info = MagicMock(side_effect=ConnectionError("offline"))
        monkeypatch.setattr(models_dev.litellm, "get_model_info", mock_info)
        assert models_dev.resolve_model_limits("openai/gpt-4o") is None

    def test_backward_compat_catalog_param(self) -> None:
        """Old callers passing catalog= get None back (ignored param)."""
        limits = models_dev.resolve_model_limits("ollama/nomic-embed-text", catalog={"old": "data"})
        assert limits == (8192, 8192)
