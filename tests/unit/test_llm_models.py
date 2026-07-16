"""Tests for knox.llm.models.build_model() and _OpencodeModel discovery."""

from __future__ import annotations

from typing import Never
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from knox.llm import models as knox_models
from knox.llm.models import _OpencodeModel, build_model

# ── build_model: standard providers via pydantic-ai auto-resolution ────


@pytest.fixture(autouse=True)
def _provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy API keys so model construction doesn't fail on auth."""
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OLLAMA_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "CLAUDINIO_API_KEY",
    ):
        monkeypatch.setenv(key, "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


class TestBuildModelStandardProviders:
    def test_ollama_prefix_returns_ollama_model(self) -> None:
        m = build_model("ollama/qwen3")
        assert isinstance(m, OllamaModel)
        assert m.model_name == "qwen3"

    def test_ollama_multi_slash_model_name_preserved(self) -> None:
        """Model names may contain slashes (e.g. hf.co/qwen3.6)."""
        m = build_model("ollama/hf.co/qwen3.6")
        assert isinstance(m, OllamaModel)
        assert m.model_name == "hf.co/qwen3.6"

    def test_openai_prefix_returns_responses_model(self) -> None:
        m = build_model("openai/gpt-5.2")
        assert isinstance(m, OpenAIResponsesModel)
        assert m.model_name == "gpt-5.2"

    def test_openai_chat_prefix(self) -> None:
        m = build_model("openai-chat/gpt-5.2")
        assert isinstance(m, OpenAIChatModel)

    def test_anthropic_prefix(self) -> None:
        m = build_model("anthropic/claude-sonnet-4-5")
        assert isinstance(m, AnthropicModel)
        assert m.model_name == "claude-sonnet-4-5"

    def test_google_prefix(self) -> None:
        m = build_model("google/gemini-3-flash")
        assert isinstance(m, GoogleModel)


# ── build_model: custom providers ──────────────────────────────────────


class TestBuildModelCustomProviders:
    def test_claudinio_returns_openai_chat_at_claudinio_endpoint(self) -> None:
        m = build_model("claudinio/claudinio")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "claudinio"
        assert str(m.base_url).rstrip("/") == "https://api.claudin.io/v1"

    def test_ollama_cloud_returns_openai_chat_at_ollama_endpoint(self) -> None:
        m = build_model("ollama-cloud/deepseek-v4-flash")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "deepseek-v4-flash"
        assert str(m.base_url).rstrip("/") == "https://ollama.com/v1"

    def test_opencode_zen_returns_opencode_model(self) -> None:
        m = build_model("opencode-zen/gpt-5.4-nano")
        assert isinstance(m, _OpencodeModel)
        assert m.model_name == "gpt-5.4-nano"

    def test_opencode_go_returns_opencode_model(self) -> None:
        m = build_model("opencode-go/kimi-k2.6")
        assert isinstance(m, _OpencodeModel)
        assert m.model_name == "kimi-k2.6"

    def test_claudinio_alias_no_dash(self) -> None:
        m = build_model("claudinio/claudinio")
        assert isinstance(m, OpenAIChatModel)


# ── build_model: error handling ────────────────────────────────────────


class TestBuildModelErrors:
    def test_unknown_prefix_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown model string"):
            build_model("nonexistent-provider/some-model")

    def test_empty_string_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            build_model("")


# ── _OpencodeModel endpoint config ─────────────────────────────────────


class TestOpencodeEndpointConfig:
    def test_zen_base_url_and_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")
        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        m = _OpencodeModel("opencode-zen", "gpt-5.4-nano")
        base, key = m._endpoint_config()
        assert base == "https://opencode.ai/zen"
        assert key == "zen-key"

    def test_go_base_url_and_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        m = _OpencodeModel("opencode-go", "kimi-k2.6")
        base, key = m._endpoint_config()
        assert base == "https://opencode.ai/zen/go"
        assert key == "go-key"

    def test_fallback_to_common_opencode_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
        monkeypatch.setenv("OPENCODE_API_KEY", "common-key")
        m = _OpencodeModel("opencode-zen", "gpt-5.4-nano")
        _, key = m._endpoint_config()
        assert key == "common-key"


# ── _OpencodeModel lazy discovery (mocked probes) ──────────────────────


class TestOpencodeDiscovery:
    def setup_method(self) -> None:
        # Clear the discovery cache between tests.
        knox_models._opencode_cache.clear()

    @pytest.mark.asyncio
    async def test_discovers_openai_chat_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """First probe (OpenAI Chat) succeeds → delegates to OpenAIChatModel."""
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")

        async def fake_probe(url: str, headers: dict[str, str], body: dict[str, object]) -> bool:
            # Only the OpenAI Chat endpoint accepts the model.
            return url.endswith("/v1/chat/completions")

        with patch.object(knox_models, "_probe_endpoint", AsyncMock(side_effect=fake_probe)):
            m = _OpencodeModel("opencode-zen", "deepseek-v4-flash")
            delegate = await m._resolve()

        assert isinstance(delegate, OpenAIChatModel)
        assert delegate.model_name == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_discovers_anthropic_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAI Chat fails, Anthropic succeeds → delegates to AnthropicModel."""
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")

        async def fake_probe(url: str, headers: dict[str, str], body: dict[str, object]) -> bool:
            return url.endswith("/v1/messages")

        with patch.object(knox_models, "_probe_endpoint", AsyncMock(side_effect=fake_probe)):
            m = _OpencodeModel("opencode-zen", "claude-sonnet-4-6")
            delegate = await m._resolve()

        assert isinstance(delegate, AnthropicModel)
        assert delegate.model_name == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_discovers_openai_responses_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chat + Anthropic fail, Responses succeeds → OpenAIResponsesModel."""
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")

        async def fake_probe(url: str, headers: dict[str, str], body: dict[str, object]) -> bool:
            return url.endswith("/v1/responses")

        with patch.object(knox_models, "_probe_endpoint", AsyncMock(side_effect=fake_probe)):
            m = _OpencodeModel("opencode-zen", "gpt-5.4-nano")
            delegate = await m._resolve()

        assert isinstance(delegate, OpenAIResponsesModel)

    @pytest.mark.asyncio
    async def test_auth_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """401 during probe raises PermissionError, not a silent fallthrough."""
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "bad-key")

        async def fake_probe(url: str, headers: dict[str, str], body: dict[str, object]) -> Never:
            raise PermissionError("HTTP 401: Invalid API key")

        with patch.object(knox_models, "_probe_endpoint", AsyncMock(side_effect=fake_probe)):
            m = _OpencodeModel("opencode-zen", "gpt-5.4-nano")
            with pytest.raises(PermissionError, match="401"):
                await m._resolve()

    @pytest.mark.asyncio
    async def test_all_formats_rejected_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")

        async def fake_probe(url: str, headers: dict[str, str], body: dict[str, object]) -> bool:
            return False  # every endpoint rejects

        with patch.object(knox_models, "_probe_endpoint", AsyncMock(side_effect=fake_probe)):
            m = _OpencodeModel("opencode-zen", "nonexistent-model")
            with pytest.raises(ValueError, match="Could not discover API format"):
                await m._resolve()

    @pytest.mark.asyncio
    async def test_discovery_cached_across_instances(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Second instance reuses cached discovery — no second probe."""
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")

        probe_mock = AsyncMock(return_value=True)  # OpenAI Chat always succeeds
        with patch.object(knox_models, "_probe_openai_chat", probe_mock):
            m1 = _OpencodeModel("opencode-zen", "deepseek-v4-flash")
            await m1._resolve()
            m2 = _OpencodeModel("opencode-zen", "deepseek-v4-flash")
            await m2._resolve()

        # Probe called once (for m1), m2 hit the cache.
        assert probe_mock.call_count == 1
