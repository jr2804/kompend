"""Tests for knox.llm.models.build_model() and resolve_model_limits()."""

from __future__ import annotations

import os

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from knox.llm.models import _split, build_model, resolve_model_limits

# ── _split ─────────────────────────────────────────────────────────────


class TestSplit:
    def test_slash_separator(self) -> None:
        assert _split("ollama/qwen3") == ("ollama", "qwen3")

    def test_colon_separator(self) -> None:
        assert _split("openai:gpt-5.2") == ("openai", "gpt-5.2")

    def test_multi_slash(self) -> None:
        """provider/sub/sub_model → provider, 'sub/sub_model'."""
        assert _split("ollama/hf.co/qwen3.6") == ("ollama", "hf.co/qwen3.6")

    def test_colon_with_slash(self) -> None:
        """provider:sub/model → provider, 'sub/model'."""
        assert _split("deepseek:sub/model") == ("deepseek", "sub/model")

    def test_slash_with_colon_in_model(self) -> None:
        """provider/model:version → provider, 'model:version'."""
        assert _split("ollama/gemma4:e2b") == ("ollama", "gemma4:e2b")

    def test_multi_slash_with_colon(self) -> None:
        assert _split("ollama/hf.co/qwen3.6:latest") == ("ollama", "hf.co/qwen3.6:latest")

    def test_colon_with_multi_slash_and_colon(self) -> None:
        assert _split("deepseek:sub/model:v2") == ("deepseek", "sub/model:v2")

    def test_first_separator_wins(self) -> None:
        assert _split("a/b:c") == ("a", "b:c")
        assert _split("a:b/c") == ("a", "b/c")

    def test_no_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            _split("justamodelname")

    def test_leading_separator_raises(self) -> None:
        """'/model' or ':model' has an empty provider → invalid."""
        with pytest.raises(ValueError, match="Invalid model string"):
            _split("/qwen3")
        with pytest.raises(ValueError, match="Invalid model string"):
            _split(":qwen3")


# ── build_model: native pydantic-ai providers ──────────────────────────


@pytest.fixture(autouse=True)
def _provider_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys for hosted providers (read by pydantic-ai's own providers).

    OLLAMA_BASE_URL is intentionally NOT set here — the native path must
    default it. See ``test_ollama_works_without_base_url_env``.
    """
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    # Catalog API keys for the fallback path
    for key in ("CLAUDINIO_API_KEY", "OPENCODE_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.setenv(key, "test-key")


class TestBuildModelNative:
    def test_ollama_works_without_base_url_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ollama must resolve without OLLAMA_BASE_URL set — the standard local
        default (OpenAI-compatible, includes ``/v1``) is passed to the
        constructor directly, never injected as an environment variable.
        """
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        m = build_model("ollama/qwen3")
        assert m.model_name == "qwen3"
        assert str(m.base_url).rstrip("/") == "http://localhost:11434/v1"

    def test_ollama_base_url_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
        m = build_model("ollama/qwen3")
        assert str(m.base_url).rstrip("/") == "http://gpu-box:11434/v1"

    def test_ollama_multi_slash_model_name_preserved(self) -> None:
        m = build_model("ollama/hf.co/qwen3.6")
        assert m.model_name == "hf.co/qwen3.6"

    def test_ollama_colon_in_model_name_preserved(self) -> None:
        m = build_model("ollama/gemma4:e2b")
        assert m.model_name == "gemma4:e2b"

    def test_openai_returns_responses_model(self) -> None:
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

    def test_colon_separator_native_provider(self) -> None:
        m = build_model("ollama:qwen3")
        assert m.model_name == "qwen3"

    def test_native_path_does_not_mutate_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Building a model must never set environment variables."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        build_model("ollama/qwen3")
        assert "OLLAMA_BASE_URL" not in os.environ


# ── build_model: models-dev catalog fallback ───────────────────────────
#
# Providers pydantic-ai has no native class for but the catalog describes
# as OpenAI-compatible build a generic OpenAIChatModel at the catalog URL.


class TestBuildModelCatalog:
    def test_ollama_cloud_via_catalog(self) -> None:
        m = build_model("ollama-cloud/deepseek-v4-flash")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "deepseek-v4-flash"
        assert str(m.base_url).rstrip("/") == "https://ollama.com/v1"

    def test_opencode_via_catalog(self) -> None:
        m = build_model("opencode/gpt-5.4-nano")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "gpt-5.4-nano"
        assert str(m.base_url).rstrip("/") == "https://opencode.ai/zen/v1"

    def test_claudinio_via_catalog(self) -> None:
        m = build_model("claudinio/claude-4")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "claude-4"
        assert str(m.base_url).rstrip("/") == "https://api.claud.in/v1" or str(
            m.base_url
        ).rstrip("/") == "https://api.claudin.io/v1"

    def test_catalog_api_key_from_advertised_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The catalog advertises CLAUDINIO_API_KEY; the key flows to the client."""
        monkeypatch.setenv("CLAUDINIO_API_KEY", "sk-catalog-key")
        m = build_model("claudinio/claude-4")
        assert isinstance(m, OpenAIChatModel)

    def test_catalog_keyless_endpoint_uses_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no API-key env var is set, a placeholder is used so the OpenAI
        SDK accepts the client (local/keyless compatible endpoints).
        """
        monkeypatch.delenv("CLAUDINIO_API_KEY", raising=False)
        m = build_model("claudinio/claude-4")
        assert isinstance(m, OpenAIChatModel)


# ── build_model: native wins over catalog ──────────────────────────────


class TestBuildModelNativeWins:
    def test_deepseek_uses_native_not_catalog(self) -> None:
        """DeepSeek is both native (DeepSeekProvider) and in the catalog.
        The native path must win — pydantic-ai's DeepSeekProvider bakes in
        its own endpoint handling.
        """
        m = build_model("deepseek/deepseek-chat")
        assert isinstance(m, OpenAIChatModel)
        # Native DeepSeekProvider sets its own base_url internally
        assert m.model_name == "deepseek-chat"


# ── build_model: errors ────────────────────────────────────────────────


class TestBuildModelErrors:
    def test_unknown_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            build_model("nonexistent-provider/some-model")

    def test_no_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            build_model("justamodelname")


# ── resolve_model_limits ───────────────────────────────────────────────


class TestResolveModelLimits:
    def test_openai_gpt4o(self) -> None:
        max_input, max_output = resolve_model_limits("openai/gpt-4o")
        assert max_input == 128000
        assert max_output == 16384

    def test_anthropic_claude(self) -> None:
        max_input, _ = resolve_model_limits("anthropic/claude-sonnet-4-20250514")
        assert max_input == 200000

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            resolve_model_limits("nonexistent/fake-model-9000")

    def test_no_separator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            resolve_model_limits("no-slash-here")

    def test_colon_separator(self) -> None:
        max_input, max_output = resolve_model_limits("openai:gpt-4o")
        assert max_input == 128000
        assert max_output == 16384
