"""Tests for KNOX LLMClient — the shared LiteLLM-backed call site."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from knox.llm.client import LLMClient
from knox.llm.errors import ConfigurationError, PermanentError, TransientError


class _FakeExc(Exception):
    """Stand-in for a litellm exception class."""


def _make_exc(name: str) -> Exception:
    cls = type(name, (_FakeExc,), {})
    return cls("boom")


def _mock_response(content: str) -> MagicMock:
    """Build a fake litellm response that works as both stream and non-stream."""
    # Non-streaming attributes.
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.delta = message  # streaming uses .delta
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    # Streaming: make it iterable, yielding one event.
    event = MagicMock()
    event.choices = [choice]
    event.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    resp.__iter__ = lambda self: iter([event])
    return resp


class TestClientHappyPath:
    def test_complete_returns_content(self) -> None:
        client = LLMClient(default_model="ollama/test", max_retries_per_model=0)
        with patch("litellm.completion", return_value=_mock_response("hello")) as mock:
            result = client.complete([{"role": "user", "content": "hi"}])
        assert result.content == "hello"
        assert result.json_data is None
        assert result.diagnostics.model_used == "ollama/test"
        assert result.diagnostics.attempts == 1
        mock.assert_called_once()
        kwargs = mock.call_args.kwargs
        assert kwargs["model"] == "ollama/test"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["temperature"] == 0.0

    def test_complete_parses_json_when_requested(self) -> None:
        client = LLMClient(default_model="ollama/test", max_retries_per_model=0)
        payload = {"entities": [{"name": "Apple"}], "relations": []}
        with patch("litellm.completion", return_value=_mock_response(json.dumps(payload))):
            result = client.complete(
                [{"role": "user", "content": "extract"}],
                response_format={"type": "json_object"},
            )
        assert result.json_data == payload

    def test_max_tokens_forwarded(self) -> None:
        client = LLMClient(default_model="ollama/test", max_retries_per_model=0)
        with patch("litellm.completion", return_value=_mock_response("hi")) as mock:
            client.complete(
                [{"role": "user", "content": "hi"}],
                max_tokens=64,
            )
        assert mock.call_args.kwargs["max_tokens"] == 64


class TestClientErrorHandling:
    def test_empty_content_raises_permanent(self) -> None:
        client = LLMClient(default_model="ollama/test", max_retries_per_model=0)
        with patch("litellm.completion", return_value=_mock_response("")), pytest.raises(PermanentError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_invalid_json_raises_permanent(self) -> None:
        client = LLMClient(default_model="ollama/test", max_retries_per_model=0)
        with patch("litellm.completion", return_value=_mock_response("not json")), pytest.raises(PermanentError):
            client.complete(
                [{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )

    def test_non_object_json_raises_permanent(self) -> None:
        """When JSON object requested but a list comes back."""
        client = LLMClient(default_model="ollama/test", max_retries_per_model=0)
        with patch("litellm.completion", return_value=_mock_response("[1,2,3]")), pytest.raises(PermanentError):
            client.complete(
                [{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )

    def test_authentication_error_raises_configuration(self) -> None:
        client = LLMClient(default_model="ollama/test", max_retries_per_model=2)
        with patch("litellm.completion", side_effect=_make_exc("AuthenticationError")), pytest.raises(ConfigurationError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_rate_limit_retries_then_succeeds(self) -> None:
        """Transient errors trigger retry within the same model."""
        client = LLMClient(default_model="ollama/test", max_retries_per_model=2)
        attempts = []

        def side_effect(**_kwargs: object) -> MagicMock:
            attempts.append(1)
            if len(attempts) < 2:
                raise _make_exc("RateLimitError")
            return _mock_response("ok")

        with patch("litellm.completion", side_effect=side_effect), patch("knox.llm.failover.time.sleep"):
            result = client.complete([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        assert result.diagnostics.attempts == 2


class TestFailoverCascade:
    def test_primary_fails_fallback_succeeds(self) -> None:
        """Rate-limit on primary → next model in chain is used."""
        client = LLMClient(
            default_model="openai/gpt-4o",
            fallback_chain=["anthropic/claude-3-haiku"],
            max_retries_per_model=1,
        )
        seen: list[str] = []

        def side_effect(*, model: str, **_kwargs: object) -> MagicMock:
            seen.append(model)
            if model == "openai/gpt-4o":
                raise _make_exc("RateLimitError")
            return _mock_response("from fallback")

        with patch("litellm.completion", side_effect=side_effect), patch("knox.llm.failover.time.sleep"):
            result = client.complete([{"role": "user", "content": "hi"}])
        assert result.content == "from fallback"
        assert result.diagnostics.model_used == "anthropic/claude-3-haiku"
        # primary tried twice (1 + 1 retry), fallback succeeds first try
        assert seen == ["openai/gpt-4o", "openai/gpt-4o", "anthropic/claude-3-haiku"]

    def test_all_models_fail_raises(self) -> None:
        client = LLMClient(
            default_model="m1",
            fallback_chain=["m2"],
            max_retries_per_model=0,
        )
        with patch("litellm.completion", side_effect=_make_exc("RateLimitError")), pytest.raises(TransientError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_try_complete_returns_none_on_failure(self) -> None:
        """try_complete swallows LLMError and returns None."""
        client = LLMClient(default_model="m1", max_retries_per_model=0)
        with patch("litellm.completion", side_effect=_make_exc("RateLimitError")):
            result = client.try_complete([{"role": "user", "content": "hi"}])
        assert result is None


class TestRoleRouting:
    def test_role_specific_chain_used(self) -> None:
        client = LLMClient(
            default_model="default/model",
            role_models={"resolve": ["resolve/model"]},
        )
        chain = client.chain_for("resolve")
        assert chain.models == ["resolve/model"]

    def test_unknown_role_uses_default_chain(self) -> None:
        client = LLMClient(
            default_model="default/model",
            fallback_chain=["fallback/model"],
        )
        chain = client.chain_for("anything")
        assert chain.models == ["default/model", "fallback/model"]
