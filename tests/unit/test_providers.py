"""Tests for KNOX custom litellm providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import litellm
import pytest

import knox.llm.providers  # noqa: F401 — imports to register providers
from knox.llm.providers import (
    _ClaudinIOProvider,
    _OllamaCloudProvider,
    _OpenCodeDummyProvider,
    _OpenCodeGoProvider,
    _OpenCodeZenProvider,
)


class TestClaudinIOProvider:
    _TEST_MODEL = "test-model"

    def test_provider_registered_on_import(self) -> None:
        """claudin-io provider should be auto-registered on import."""
        providers = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == "claudin-io"]
        assert len(providers) == 1
        assert providers[0]["custom_handler"] is not None

    def test_provider_model_format(self) -> None:
        """Provider should delegate with custom base URL."""
        # Test _delegate_kwargs generates the right model and api_base
        kwargs = _ClaudinIOProvider._delegate_kwargs(self._TEST_MODEL, {"temperature": 0.7})
        assert kwargs["model"] == f"openai/{self._TEST_MODEL}"
        assert kwargs["api_base"] == "https://api.claudin.io/v1"
        assert kwargs["temperature"] == 0.7

    def test_provider_uses_env_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider should use CLAUDINIO_API_KEY from environment."""
        monkeypatch.setenv("CLAUDINIO_API_KEY", "test-key-123")
        kwargs = _ClaudinIOProvider._delegate_kwargs(self._TEST_MODEL, {})
        assert kwargs["api_key"] == "test-key-123"

    def test_provider_no_api_key_when_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider should return empty string when env var not set."""
        monkeypatch.delenv("CLAUDINIO_API_KEY", raising=False)
        kwargs = _ClaudinIOProvider._delegate_kwargs(self._TEST_MODEL, {})
        assert kwargs["api_key"] == ""

    def test_completion_delegates_to_litellm(self) -> None:
        """completion() should delegate to litellm.completion with correct params."""
        handler = _ClaudinIOProvider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]

        with patch("litellm.completion", return_value=mock_response) as mock:
            handler.completion(
                model=self._TEST_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_base="",
                custom_prompt_dict={},
                model_response=MagicMock(),
                print_verbose=lambda x: None,
                encoding="utf-8",
                api_key="",
                logging_obj=None,
                optional_params={"temperature": 0.5},
            )

        mock.assert_called_once()
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["model"] == f"openai/{self._TEST_MODEL}"
        assert call_kwargs["api_base"] == "https://api.claudin.io/v1"
        assert call_kwargs["temperature"] == 0.5

    def test_streaming_delegates_to_litellm(self) -> None:
        """streaming() should delegate to litellm.completion with stream=True."""
        handler = _ClaudinIOProvider()
        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([])

        with patch("litellm.completion", return_value=mock_stream) as mock:
            list(
                handler.streaming(
                    model=self._TEST_MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                    api_base="",
                    custom_prompt_dict={},
                    model_response=MagicMock(),
                    print_verbose=lambda x: None,
                    encoding="utf-8",
                    api_key="",
                    logging_obj=None,
                    optional_params={},
                )
            )

        mock.assert_called_once()
        assert mock.call_args.kwargs["stream"] is True
        assert mock.call_args.kwargs["model"] == f"openai/{self._TEST_MODEL}"

    @pytest.mark.asyncio
    async def test_acompletion_delegates_to_litellm(self) -> None:
        """acompletion() should delegate to litellm.acompletion."""
        handler = _ClaudinIOProvider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]

        async def fake_acompletion(**kwargs: Any) -> MagicMock:
            return mock_response

        with patch("litellm.acompletion", side_effect=fake_acompletion) as mock:
            await handler.acompletion(
                model=self._TEST_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_base="",
                custom_prompt_dict={},
                model_response=MagicMock(),
                print_verbose=lambda x: None,
                encoding="utf-8",
                api_key="",
                logging_obj=None,
                optional_params={},
            )

        mock.assert_called_once()
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["model"] == f"openai/{self._TEST_MODEL}"
        assert call_kwargs["api_base"] == "https://api.claudin.io/v1"

    @pytest.mark.asyncio
    async def test_astreaming_delegates_to_litellm(self) -> None:
        """astreaming() should delegate to litellm.acompletion with stream=True."""
        handler = _ClaudinIOProvider()

        async def fake_acompletion(**kwargs: Any) -> AsyncGenerator[MagicMock]:
            async def gen() -> AsyncGenerator[MagicMock]:
                yield MagicMock()

            return gen()

        with patch("litellm.acompletion", side_effect=fake_acompletion) as mock:
            chunks = []
            async for chunk in handler.astreaming(
                model=self._TEST_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_base="",
                custom_prompt_dict={},
                model_response=MagicMock(),
                print_verbose=lambda x: None,
                encoding="utf-8",
                api_key="",
                logging_obj=None,
                optional_params={},
            ):
                chunks.append(chunk)

        mock.assert_called_once()
        assert mock.call_args.kwargs["stream"] is True
        assert mock.call_args.kwargs["model"] == f"openai/{self._TEST_MODEL}"


class TestOllamaCloudProvider:
    def test_provider_registered_on_import(self) -> None:
        """ollama-cloud provider should be auto-registered on import."""
        providers = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == "ollama-cloud"]
        assert len(providers) == 1

    def test_provider_model_format(self) -> None:
        """Provider should delegate to ollama_chat/ prefix."""
        kwargs = _OllamaCloudProvider._delegate_kwargs("deepseek-v4-flash", {"temperature": 0.7})
        assert kwargs["model"] == "ollama_chat/deepseek-v4-flash"
        assert kwargs["api_base"] == "https://ollama.com"


class TestOpenCodeZenProvider:
    def test_provider_registered(self) -> None:
        providers = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == "opencode-zen"]
        assert len(providers) == 1

    def test_delegate_kwargs_defaults_to_openai_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-key")
        _OpenCodeZenProvider._model_cache = {"deepseek-v4-flash-free": "openai"}
        try:
            kwargs = _OpenCodeZenProvider._delegate_kwargs("deepseek-v4-flash-free", {"temperature": 0.5})
            assert kwargs["model"] == "openai/deepseek-v4-flash-free"
            assert kwargs["api_base"] == "https://opencode.ai/zen/v1/chat/completions"
            assert kwargs["temperature"] == 0.5
        finally:
            _OpenCodeZenProvider._model_cache = None

    def test_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
        monkeypatch.setenv("OPENCODE_API_KEY", "generic-key")
        assert _OpenCodeZenProvider._api_key() == "generic-key"

    def test_specific_key_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-specific")
        monkeypatch.setenv("OPENCODE_API_KEY", "generic-key")
        assert _OpenCodeZenProvider._api_key() == "zen-specific"

    def test_no_api_key_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        assert _OpenCodeZenProvider._api_key() == ""

    def test_stream_kwarg_filtered(self) -> None:

        kwargs = _OpenCodeZenProvider._delegate_kwargs("model", {"stream": True, "temperature": 0.3})
        assert "stream" not in kwargs

    def test_completion_delegates(self) -> None:

        handler = _OpenCodeZenProvider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        with patch("litellm.completion", return_value=mock_response) as mock:
            handler.completion(
                model="deepseek-v4-flash-free",
                messages=[{"role": "user", "content": "hi"}],
                api_base="",
                custom_prompt_dict={},
                model_response=MagicMock(),
                print_verbose=lambda x: None,
                encoding="utf-8",
                api_key="",
                logging_obj=None,
                optional_params={},
            )
        assert mock.call_args.kwargs["model"] == "openai/deepseek-v4-flash-free"
        assert mock.call_args.kwargs["api_base"] == "https://opencode.ai/zen/v1/chat/completions"

    def test_model_discovery_populates_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-key")
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"data": [{"id": "m1"}, {"id": "m2"}]}

        with patch("httpx.get", return_value=fake_resp):
            _OpenCodeZenProvider._model_cache = None
            cache = _OpenCodeZenProvider._discover_models()

        assert cache == {"m1": "openai", "m2": "openai"}

    def test_model_discovery_raises_on_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            _OpenCodeZenProvider._discover_models()

    def test_model_discovery_raises_on_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-key")
        with patch("httpx.get", side_effect=Exception("connection refused")), pytest.raises(ValueError, match="Failed to discover models"):
            _OpenCodeZenProvider._discover_models()


class TestOpenCodeGoProvider:
    def test_provider_registered(self) -> None:
        providers = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == "opencode-go"]
        assert len(providers) == 1

    def test_delegate_kwargs_uses_go_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")
        _OpenCodeGoProvider._model_cache = {"deepseek-v4-flash": "openai"}
        try:
            kwargs = _OpenCodeGoProvider._delegate_kwargs("deepseek-v4-flash", {})
            assert kwargs["api_base"] == "https://opencode.ai/zen/go/v1/chat/completions"
        finally:
            _OpenCodeGoProvider._model_cache = None

    def test_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
        monkeypatch.setenv("OPENCODE_API_KEY", "generic-key")
        assert _OpenCodeGoProvider._api_key() == "generic-key"

    def test_specific_key_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-specific")
        monkeypatch.setenv("OPENCODE_API_KEY", "generic-key")
        assert _OpenCodeGoProvider._api_key() == "go-specific"


class TestOpenCodeDummyProvider:
    def test_provider_registered(self) -> None:
        providers = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == "opencode"]
        assert len(providers) == 1

    def test_completion_raises(self) -> None:

        handler = _OpenCodeDummyProvider()
        with pytest.raises(ValueError, match="opencode-go.*opencode-zen"):
            handler.completion(
                model="m",
                messages=[],
                api_base="",
                custom_prompt_dict={},
                model_response=MagicMock(),
                print_verbose=lambda x: None,
                encoding="utf-8",
                api_key="",
                logging_obj=None,
                optional_params={},
            )

    def test_streaming_raises(self) -> None:

        handler = _OpenCodeDummyProvider()
        with pytest.raises(ValueError, match="opencode-go.*opencode-zen"):
            handler.streaming(
                model="m",
                messages=[],
                api_base="",
                custom_prompt_dict={},
                model_response=MagicMock(),
                print_verbose=lambda x: None,
                encoding="utf-8",
                api_key="",
                logging_obj=None,
                optional_params={},
            )
