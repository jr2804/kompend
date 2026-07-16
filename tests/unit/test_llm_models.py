"""Tests for knox.llm.models.build_model() and resolve_model_limits()."""

from __future__ import annotations

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from knox.llm.models import _parse_model_string, build_model, resolve_model_limits

# ── _parse_model_string ────────────────────────────────────────────────


class TestParseModelString:
    def test_slash_separator(self) -> None:
        assert _parse_model_string("ollama/qwen3") == ("ollama", "qwen3")

    def test_colon_separator(self) -> None:
        assert _parse_model_string("openai:gpt-5.2") == ("openai", "gpt-5.2")

    def test_multi_slash(self) -> None:
        """provider/sub/sub_model → provider='provider', model='sub/sub_model'"""
        assert _parse_model_string("ollama/hf.co/qwen3.6") == ("ollama", "hf.co/qwen3.6")

    def test_colon_with_slash(self) -> None:
        """provider:sub/model → provider='provider', model='sub/model'"""
        assert _parse_model_string("deepseek:sub/model") == ("deepseek", "sub/model")

    def test_slash_with_colon_in_model(self) -> None:
        """provider/model:version → provider='provider', model='model:version'"""
        assert _parse_model_string("ollama/gemma4:e2b") == ("ollama", "gemma4:e2b")

    def test_multi_slash_with_colon(self) -> None:
        """provider/sub/model:version → provider, sub/model:version"""
        assert _parse_model_string("ollama/hf.co/qwen3.6:latest") == ("ollama", "hf.co/qwen3.6:latest")

    def test_colon_with_multi_slash_and_colon(self) -> None:
        """provider:sub/model:version → provider, sub/model:version"""
        assert _parse_model_string("deepseek:sub/model:v2") == ("deepseek", "sub/model:v2")

    def test_first_separator_wins(self) -> None:
        """When both / and : exist, the first one splits."""
        assert _parse_model_string("a/b:c") == ("a", "b:c")
        assert _parse_model_string("a:b/c") == ("a", "b/c")

    def test_no_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            _parse_model_string("justamodelname")

    def test_empty_model_name_raises(self) -> None:
        """'provider/' or 'provider:' with empty model name."""
        # sep_pos would be > 0 but model_name would be empty
        assert _parse_model_string("ollama/") == ("ollama", "")
        assert _parse_model_string("ollama:") == ("ollama", "")


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

    def test_colon_separator_standard_provider(self) -> None:
        """provider:model works for standard providers."""
        m = build_model("ollama:qwen3")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "qwen3"

    def test_colon_separator_with_slash_in_model(self) -> None:
        """provider:sub/model works."""
        m = build_model("ollama:hf.co/qwen3.6")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "hf.co/qwen3.6"

    def test_colon_separator_with_colon_in_model(self) -> None:
        """provider:model:version works — first colon splits."""
        m = build_model("ollama:gemma4:e2b")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "gemma4:e2b"

    def test_multi_slash_with_version(self) -> None:
        """provider/sub/model:version works."""
        m = build_model("ollama/hf.co/qwen3.6:latest")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "hf.co/qwen3.6:latest"


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

    def test_no_separator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            build_model("justamodelname")


# ── resolve_model_limits: models.dev catalog lookup ───────────────────


class TestResolveModelLimits:
    def test_openai_gpt4o_returns_limits(self) -> None:
        max_input, max_output = resolve_model_limits("openai/gpt-4o")
        assert max_input == 128000
        assert max_output == 16384

    def test_anthropic_claude_returns_limits(self) -> None:
        max_input, _ = resolve_model_limits("anthropic/claude-sonnet-4-20250514")
        assert max_input == 200000

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_model_limits("nonexistent/fake-model-9000")

    def test_no_separator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            resolve_model_limits("no-slash-here")

    def test_colon_separator_works(self) -> None:
        max_input, max_output = resolve_model_limits("openai:gpt-4o")
        assert max_input == 128000
        assert max_output == 16384
