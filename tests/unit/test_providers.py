"""Tests for KNOX custom litellm providers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import litellm
import pytest

import knox.llm.providers  # noqa: F401 — imports to register providers


class TestClaudinIOProvider:
    def test_provider_registered_on_import(self) -> None:
        """claudin-io provider should be auto-registered on import."""
        providers = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == "claudin-io"]
        assert len(providers) == 1
        assert providers[0]["custom_handler"] is not None

    def test_provider_model_format(self) -> None:
        """Provider should delegate to anthropic/ prefix with custom base URL."""
        from knox.llm.providers import _ClaudinIOProvider

        # Test _delegate_kwargs generates the right model and api_base
        kwargs = _ClaudinIOProvider._delegate_kwargs("claude-3.5-sonnet", {"temperature": 0.7})
        assert kwargs["model"] == "anthropic/claude-3.5-sonnet"
        assert kwargs["api_base"] == "https://api.claudin.io/v1/messages"
        assert kwargs["temperature"] == 0.7

    def test_provider_uses_env_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider should use CLAUDINIO_API_KEY from environment."""
        from knox.llm.providers import _ClaudinIOProvider

        monkeypatch.setenv("CLAUDINIO_API_KEY", "test-key-123")
        kwargs = _ClaudinIOProvider._delegate_kwargs("claude-3.5-sonnet", {})
        assert kwargs["api_key"] == "test-key-123"

    def test_provider_no_api_key_when_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider should return empty string when env var not set."""
        from knox.llm.providers import _ClaudinIOProvider

        monkeypatch.delenv("CLAUDINIO_API_KEY", raising=False)
        kwargs = _ClaudinIOProvider._delegate_kwargs("claude-3.5-sonnet", {})
        assert kwargs["api_key"] == ""

    def test_completion_delegates_to_litellm(self) -> None:
        """completion() should delegate to litellm.completion with correct params."""
        from knox.llm.providers import _ClaudinIOProvider

        handler = _ClaudinIOProvider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]

        with patch("litellm.completion", return_value=mock_response) as mock:
            handler.completion(
                model="claude-3.5-sonnet",
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
        assert call_kwargs["model"] == "anthropic/claude-3.5-sonnet"
        assert call_kwargs["api_base"] == "https://api.claudin.io/v1/messages"
        assert call_kwargs["temperature"] == 0.5

    def test_streaming_delegates_to_litellm(self) -> None:
        """streaming() should delegate to litellm.completion with stream=True."""
        from knox.llm.providers import _ClaudinIOProvider

        handler = _ClaudinIOProvider()
        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter([])

        with patch("litellm.completion", return_value=mock_stream) as mock:
            list(
                handler.streaming(
                    model="claude-3.5-sonnet",
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
        assert mock.call_args.kwargs["model"] == "anthropic/claude-3.5-sonnet"

    @pytest.mark.asyncio
    async def test_acompletion_delegates_to_litellm(self) -> None:
        """acompletion() should delegate to litellm.acompletion."""
        from knox.llm.providers import _ClaudinIOProvider

        handler = _ClaudinIOProvider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]

        async def fake_acompletion(**kwargs):
            return mock_response

        with patch("litellm.acompletion", side_effect=fake_acompletion) as mock:
            await handler.acompletion(
                model="claude-3.5-sonnet",
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
        assert call_kwargs["model"] == "anthropic/claude-3.5-sonnet"
        assert call_kwargs["api_base"] == "https://api.claudin.io/v1/messages"

    @pytest.mark.asyncio
    async def test_astreaming_delegates_to_litellm(self) -> None:
        """astreaming() should delegate to litellm.acompletion with stream=True."""
        from knox.llm.providers import _ClaudinIOProvider

        handler = _ClaudinIOProvider()

        async def fake_acompletion(**kwargs):
            async def gen():
                yield MagicMock()

            return gen()

        with patch("litellm.acompletion", side_effect=fake_acompletion) as mock:
            chunks = []
            async for chunk in handler.astreaming(
                model="claude-3.5-sonnet",
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
        assert mock.call_args.kwargs["model"] == "anthropic/claude-3.5-sonnet"


class TestOllamaCloudProvider:
    def test_provider_registered_on_import(self) -> None:
        """ollama-cloud provider should be auto-registered on import."""
        providers = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == "ollama-cloud"]
        assert len(providers) == 1

    def test_provider_model_format(self) -> None:
        """Provider should delegate to ollama_chat/ prefix."""
        from knox.llm.providers import _OllamaCloudProvider

        kwargs = _OllamaCloudProvider._delegate_kwargs("deepseek-v4-flash", {"temperature": 0.7})
        assert kwargs["model"] == "ollama_chat/deepseek-v4-flash"
        assert kwargs["api_base"] == "https://ollama.com"
