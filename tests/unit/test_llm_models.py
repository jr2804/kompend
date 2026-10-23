"""Tests for kompend.llm.models.build_model() and resolve_model_limits()."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from types import SimpleNamespace

import llmdb
import pydantic
import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

import kompend.llm.models as mod
from kompend.core.config._configs import ProviderEndpoint
from kompend.llm import catalog
from kompend.llm.models import (
    OLLAMA_CONTEXT_LENGTH_ENV,
    _resolve_env_ref,
    _split,
    build_model,
    requires_output_ceiling,
    resolve_model_limits,
    set_user_providers,
)

# ── Catalogue fixture ──────────────────────────────────────────────────
#
# The model catalogue is live data (models.dev), so these tests pin it to a
# synthetic snapshot instead of whatever the network served today: URLs and
# limits move, and a test that asserted on them would break on every refresh.

_CATALOGUE_ENTRIES: dict[str, dict[str, object]] = {
    "anthropic": {
        "api": "https://api.anthropic.com/v1",
        "env": ["ANTHROPIC_API_KEY"],
        "models": {
            "claude-sonnet-4-20250514": (200000, 64000, False),
            "claude-sonnet-4-5": (200000, 64000, False),
        },
    },
    "claudinio": {
        "api": "https://api.claudin.io/v1",
        "env": ["CLAUDINIO_API_KEY"],
        "models": {"claude-4": (200000, 32000, False)},
    },
    "google": {
        "api": "https://generativelanguage.googleapis.com",
        "env": ["GOOGLE_API_KEY"],
        "models": {
            "gemini-3-flash": (1000000, 65536, True),
            "gemini-2.5-flash": (1048576, 65536, True),
        },
    },
    "minimax": {
        "api": "https://api.minimax.io/v1",
        "env": ["MINIMAX_API_KEY"],
        "models": {
            "MiniMax-M2.7-highspeed": (196608, 128000, False),
            "MiniMax-M3.1-Flash": (1000000, 524288, True),
        },
    },
    "ollama-cloud": {
        "api": "https://ollama.com/v1",
        "env": [],
        "models": {"deepseek-v4-flash": (131072, 8192, False)},
    },
    "openai": {
        "api": "https://api.openai.com/v1",
        "env": ["OPENAI_API_KEY"],
        "models": {
            "gpt-4o": (128000, 16384, True),
            "gpt-5.2": (400000, 128000, True),
        },
    },
    "opencode": {
        "api": "https://opencode.ai/inference/openai/v1",
        "env": ["OPENCODE_API_KEY"],
        "models": {"gpt-5.4-nano": (200000, 32768, False)},
    },
}


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


class TestBuildModelNative:
    def test_ollama_returns_native_model_with_default_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ollama returns a native OllamaModel with the local default base_url.

        No OLLAMA_BASE_URL needed — the local default
        ``http://127.0.0.1:11434`` is used when the env var is unset.
        """
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        m = build_model("ollama/qwen3")
        assert isinstance(m, OllamaModel)
        assert m.model_name == "qwen3"

    def test_ollama_base_url_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OLLAMA_BASE_URL env var overrides the local default."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:1234")
        m = build_model("ollama/qwen3")
        assert m.model_name == "qwen3"

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
        assert str(m.base_url).rstrip("/") == "https://opencode.ai/inference/openai/v1"

    def test_claudinio_via_catalog(self) -> None:
        m = build_model("claudinio/claude-4")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "claude-4"
        assert str(m.base_url).rstrip("/") == "https://api.claud.in/v1" or str(m.base_url).rstrip("/") == "https://api.claudin.io/v1"

    def test_catalog_api_key_from_advertised_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The catalog advertises CLAUDINIO_API_KEY; the key flows to the client."""
        monkeypatch.setenv("CLAUDINIO_API_KEY", "sk-catalog-key")
        m = build_model("claudinio/claude-4")
        assert isinstance(m, OpenAIChatModel)

    def test_catalog_keyless_endpoint_uses_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no API-key env var is set, a placeholder is used so the OpenAI
        SDK accepts the client (local/keyless compatible endpoints).
        """
        monkeypatch.delenv("CLAUDINIO_API_KEY", raising=False)
        m = build_model("claudinio/claude-4")
        assert isinstance(m, OpenAIChatModel)


# ── build_model: errors ────────────────────────────────────────────────


class TestBuildModelErrors:
    def test_unknown_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            build_model("nonexistent-provider/some-model")

    def test_no_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            build_model("justamodelname")


# ── _resolve_env_ref ───────────────────────────────────────────────────


class TestResolveEnvRef:
    def test_resolves_known_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "sk-real-value")
        assert _resolve_env_ref("${MY_KEY}") == "sk-real-value"

    def test_unset_var_returns_original(self) -> None:
        result = _resolve_env_ref("${UNSET_VAR}")
        assert result == "${UNSET_VAR}"

    def test_no_ref_returns_original(self) -> None:
        assert _resolve_env_ref("plain-string") == "plain-string"

    def test_empty_string(self) -> None:
        assert _resolve_env_ref("") == ""


# ── build_model: user-defined providers ────────────────────────────────


class TestBuildModelUserProvider:
    """User-defined providers from [providers] in kompend.toml."""

    def setup_method(self) -> None:
        set_user_providers(
            {
                "my-llama": {
                    "base_url": "http://localhost:8081/v1",
                    "api_key": "${MY_LLAMA_KEY}",
                },
                "keyless": {
                    "base_url": "http://localhost:9999/v1",
                },
            }
        )

    def teardown_method(self) -> None:
        set_user_providers({})

    def test_user_provider_returns_openai_chat_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_LLAMA_KEY", "sk-llama-key")
        m = build_model("my-llama/mistral-nemo")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "mistral-nemo"
        assert str(m.base_url).rstrip("/") == "http://localhost:8081/v1"

    def test_user_provider_api_key_resolved_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_LLAMA_KEY", "sk-resolved-key")
        m = build_model("my-llama/mistral-nemo")
        assert isinstance(m, OpenAIChatModel)

    def test_user_provider_no_api_key_uses_placeholder(self) -> None:
        m = build_model("keyless/some-model")
        assert isinstance(m, OpenAIChatModel)
        assert m.model_name == "some-model"

    def test_user_provider_missing_base_url_falls_through(self) -> None:
        """Override-only blocks (no base_url) skip model building, so an
        unknown provider id surfaces the standard catalog/unknown error
        instead of a misleading 'no base_url' one.
        """
        set_user_providers({"broken": {"api_key": "sk-key"}})
        with pytest.raises(ValueError, match="Unknown provider"):
            build_model("broken/model")

    def test_user_provider_unknown_provider_falls_through(self) -> None:
        """A provider not in _user_providers should still raise the standard error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            build_model("not-registered/some-model")

    def test_api_type_dispatch(self) -> None:
        """api_type selects the wire protocol model class."""
        set_user_providers(
            {
                "chat": {"base_url": "http://x/v1"},
                "responses": {"base_url": "http://x/v1", "api_type": "openai-responses"},
                "anthropic": {"base_url": "http://x/v1", "api_type": "anthropic", "api_key": "k"},
            }
        )
        assert isinstance(build_model("chat/m"), OpenAIChatModel)
        assert isinstance(build_model("responses/m"), OpenAIResponsesModel)
        assert isinstance(build_model("anthropic/m"), AnthropicModel)

    def test_invalid_api_type_rejected_by_schema(self) -> None:
        """api_type is a closed Literal — unknown values fail config validation."""
        with pytest.raises(pydantic.ValidationError):
            ProviderEndpoint.model_validate({"base_url": "http://x/v1", "api_type": "grpc"})

    def test_user_provider_beats_catalog_for_known_id(self) -> None:
        """A user-defined provider shadows a models.dev catalog entry with the same id.

        Regression: 'minimax' exists in the catalog, so a user redefining it
        (e.g. an anthropic-protocol proxy) got the catalog's OpenAI chat model
        against the wrong endpoint — 404s on every request.
        """
        set_user_providers({"minimax": {"base_url": "https://api.minimax.io/anthropic", "api_key": "k", "api_type": "anthropic"}})
        try:
            m = build_model("minimax/MiniMax-M2.7-highspeed")
            assert isinstance(m, AnthropicModel)
            assert str(m.base_url).rstrip("/") == "https://api.minimax.io/anthropic"
        finally:
            set_user_providers({})


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


class TestQueryOllamaContextLength:
    """``_query_ollama_num_ctx`` prefers the context the server actually loaded."""

    def test_loaded_context_from_ps_wins_over_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A memory-capped slot (ps) beats the aspirational GGUF metadata."""
        monkeypatch.delenv(OLLAMA_CONTEXT_LENGTH_ENV, raising=False)
        monkeypatch.setattr(mod.ollama, "ps", lambda: SimpleNamespaceLike(models=self._ps_models(65536)))
        monkeypatch.setattr(
            mod.ollama,
            "show",
            lambda name: SimpleNamespaceLike(parameters="", modelinfo={"qwen3.context_length": 262144}),
        )
        assert mod._query_ollama_num_ctx("m") == 65536

    def test_env_override_beats_ps(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setenv(OLLAMA_CONTEXT_LENGTH_ENV, "32768")
        monkeypatch.setattr(mod.ollama, "ps", lambda: SimpleNamespaceLike(models=self._ps_models(65536)))
        monkeypatch.setattr(mod.ollama, "show", lambda name: SimpleNamespaceLike(parameters="", modelinfo={}))
        assert mod._query_ollama_num_ctx("m") == 32768

    def test_ps_failure_falls_back_to_show(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.delenv(OLLAMA_CONTEXT_LENGTH_ENV, raising=False)

        def _boom() -> object:
            raise RuntimeError("server down")

        monkeypatch.setattr(mod.ollama, "ps", _boom)
        monkeypatch.setattr(
            mod.ollama,
            "show",
            lambda name: SimpleNamespaceLike(parameters="num_ctx 8192", modelinfo={}),
        )
        assert mod._query_ollama_num_ctx("m") == 8192

    def _ps_models(self, ctx: int) -> list:
        """Build a minimal ``ollama.ps()``-shaped model list."""
        return [SimpleNamespace(name="m", model="m", context_length=ctx)]


class SimpleNamespaceLike:
    """Attribute container used by the num_ctx tests."""

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


# ── vision support + VLM fallback ─────────────────────────────────────────


class TestVisionSupport:
    def teardown_method(self) -> None:
        mod._VISION_CACHE.clear()

    def test_catalog_vision_model_has_image_modality(self) -> None:
        # gpt-5.2 modalities.input includes 'image' in the models-dev catalog.
        assert mod.model_supports_vision("openai/gpt-5.2") or mod.model_supports_vision("openai/gpt-5")
        assert mod.model_supports_vision("google/gemini-2.5-flash")

    def test_unknown_model_in_catalog_not_listed_is_no_vision(self) -> None:
        # A model string that parses but is not in the catalog → False.
        assert mod.model_supports_vision("openai/definitely-not-a-real-model-xyz") is False

    def test_invalid_model_string_is_false(self) -> None:
        assert mod.model_supports_vision("no-separator") is False

    def test_ollama_probe_failure_is_conservative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dead Ollama server must not enable the fallback (no wrong VLM send)."""

        def _fail(name: str) -> None:
            raise ConnectionError("refused")

        monkeypatch.setattr(mod.ollama, "show", _fail)
        assert mod.model_supports_vision("ollama/anything") is False
        assert "ollama/anything" in mod._VISION_CACHE


class TestResolveVlmModel:
    def teardown_method(self) -> None:
        mod._VISION_CACHE.clear()

    def test_explicit_vlm_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "model_supports_vision", lambda s: True)
        assert mod.resolve_vlm_model("openai/gpt-5.2", "openai/gpt-4o") == "openai/gpt-4o"

    def test_default_model_as_fallback_when_vision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "model_supports_vision", lambda s: True)
        assert mod.resolve_vlm_model("openai/gpt-5.2", None) == "openai/gpt-5.2"

    def test_no_fallback_when_no_vision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "model_supports_vision", lambda s: False)
        assert mod.resolve_vlm_model("openai/gpt-5.2", None) is None

    def test_no_vlm_no_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "model_supports_vision", lambda s: True)
        assert mod.resolve_vlm_model(None, None) is None


class TestRequiresOutputCeiling:
    """Only the Anthropic protocol rejects a request without max_tokens."""

    def test_anthropic_needs_one(self) -> None:
        model = AnthropicModel("claude-x", provider=AnthropicProvider(api_key="x"))

        assert requires_output_ceiling(model) is True

    def test_openai_and_ollama_do_not(self) -> None:
        assert requires_output_ceiling(OpenAIChatModel("gpt-x", provider=OpenAIProvider(api_key="x"))) is False
        assert requires_output_ceiling(OllamaModel("qwen", provider=OllamaProvider(base_url="http://127.0.0.1:11434"))) is False

    def test_follows_a_delegate_and_a_fallback_chain(self) -> None:
        anthropic = AnthropicModel("claude-x", provider=AnthropicProvider(api_key="x"))
        openai = OpenAIChatModel("gpt-x", provider=OpenAIProvider(api_key="x"))

        class _Wrapper:
            delegate = anthropic

        assert requires_output_ceiling(anthropic) is True
        assert requires_output_ceiling(FallbackModel(openai, anthropic)) is True
        assert requires_output_ceiling(FallbackModel(openai)) is False


@pytest.fixture(autouse=True)
def _catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the model catalogue to ``_CATALOGUE_ENTRIES`` for every test here."""
    providers: dict[str, llmdb.Provider] = {}
    for provider_id, entry in _CATALOGUE_ENTRIES.items():
        models = entry["models"]
        if not isinstance(models, dict):
            continue
        providers[provider_id] = llmdb.Provider.model_validate(
            {
                "id": provider_id,
                "name": provider_id,
                "api": entry.get("api"),
                "env": entry.get("env", []),
                "models": {model_id: _catalog_model(model_id, limits) for model_id, limits in models.items()},
            }
        )
    snapshot = catalog.CatalogSnapshot(fetched_at=datetime.now(UTC), providers=providers)
    catalog._install(snapshot)


def _catalog_model(model_id: str, limits: tuple[int, int, bool]) -> dict[str, object]:
    """Build one catalogue model payload with the fields ``llmdb`` requires."""
    context, output, image = limits
    return {
        "id": model_id,
        "name": model_id,
        "attachment": False,
        "reasoning": False,
        "tool_call": True,
        "release_date": "2025-01-01",
        "last_updated": "2025-01-01",
        "modalities": {"input": ["text", "image"] if image else ["text"], "output": ["text"]},
        "open_weights": False,
        "limit": {"context": context, "output": output},
    }


# ── build_model: native pydantic-ai providers ──────────────────────────


@pytest.fixture(autouse=True)
def _provider_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys for hosted providers (read by pydantic-ai's own providers).

    OLLAMA_BASE_URL is intentionally NOT set here -- the ollama provider owns
    the transport (endpoint, auth) and defaults to localhost:11434.
    """
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    # Catalog API keys for the fallback path
    for key in ("CLAUDINIO_API_KEY", "OPENCODE_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.setenv(key, "test-key")
