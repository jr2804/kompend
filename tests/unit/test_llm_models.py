"""Tests for knox.llm.models.build_model() and resolve_model_limits()."""

from __future__ import annotations

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from knox.llm.models import build_model, resolve_model_limits

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
    def test_ollama_prefix_returns_openai_compatible_model(self) -> None:
        """Ollama is resolved as a local OpenAI-compatible provider, not
        pydantic-ai's native OllamaModel (which requires OLLAMA_BASE_URL).
        """
        m = build_model("ollama/qwen3")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "qwen3"
        assert m.model_name == "qwen3"

    def test_ollama_multi_slash_model_name_preserved(self) -> None:
        m = build_model("ollama/hf.co/qwen3.6")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "hf.co/qwen3.6"

    def test_ollama_colon_in_model_name_preserved(self) -> None:
        """Ollama model tags contain a colon (e.g. ``gemma4:e2b``); the
        internal colon must survive the ``provider/model`` split.
        """
        m = build_model("ollama/gemma4:e2b")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "gemma4:e2b"

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


# ── build_model: custom providers via models-dev catalog ──────────────


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

    def test_opencode_returns_openai_chat_at_opencode_endpoint(self) -> None:
        m = build_model("opencode/gpt-5.4-nano")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "gpt-5.4-nano"
        assert str(m.base_url).rstrip("/") == "https://opencode.ai/zen/v1"

    def test_opencode_go_returns_openai_chat_at_opencode_go_endpoint(self) -> None:
        m = build_model("opencode-go/kimi-k2.6")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "kimi-k2.6"
        assert str(m.base_url).rstrip("/") == "https://opencode.ai/zen/go/v1"

    def test_deepseek_returns_openai_chat_with_v1_appended(self) -> None:
        """Deepseek's api URL lacks /v1; the code appends it."""
        m = build_model("deepseek/deepseek-chat")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "deepseek-chat"
        assert str(m.base_url).rstrip("/") == "https://api.deepseek.com/v1"


# ── build_model: error handling ────────────────────────────────────────


class TestBuildModelErrors:
    def test_unknown_prefix_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent-provider/some-model")

    def test_no_slash_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            build_model("justamodelname")


# ── resolve_model_limits: models.dev catalog lookup ───────────────────


class TestResolveModelLimits:
    def test_openai_gpt4o_returns_limits(self) -> None:
        limits = resolve_model_limits("openai/gpt-4o")
        assert limits is not None
        max_input, max_output = limits
        assert max_input == 128000
        assert max_output == 16384

    def test_anthropic_claude_returns_limits(self) -> None:
        limits = resolve_model_limits("anthropic/claude-sonnet-4-20250514")
        assert limits is not None
        max_input, _ = limits
        assert max_input == 200000

    def test_unknown_model_returns_none(self) -> None:
        limits = resolve_model_limits("nonexistent/fake-model-9000")
        assert limits is None

    def test_missing_slash_returns_none(self) -> None:
        limits = resolve_model_limits("no-slash-here")
        assert limits is None
