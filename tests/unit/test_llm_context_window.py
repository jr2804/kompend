"""Tests for ``discover_context_window`` cap coercion.

Regression coverage for the bug where a ``context_window_cap`` loaded from
TOML with env-var interpolation (e.g. ``context_window_cap = "${CTX_CAP}"``)
arrives as a string, causing ``TypeError: '<' not supported between
instances of 'str' and 'int'`` in the ``min()`` call.
"""

from __future__ import annotations

from unittest.mock import patch

from knox.llm.context_window import (
    ContextSource,
    _coerce_context_window_cap,
    _split_model,
    discover_context_window,
)
from knox.llm.models import set_user_providers


class TestCoerceContextWindowCap:
    def test_int_passthrough(self) -> None:
        assert _coerce_context_window_cap(65536) == 65536
        assert _coerce_context_window_cap(0) == 0

    def test_numeric_string_coerces(self) -> None:
        assert _coerce_context_window_cap("128000") == 128000
        assert _coerce_context_window_cap("65536") == 65536

    def test_none_returns_none(self) -> None:
        assert _coerce_context_window_cap(None) is None

    def test_unexpanded_env_placeholder_returns_none(self) -> None:
        # Unset env var leaves the literal ${VAR} placeholder.
        assert _coerce_context_window_cap("${KNOX_LOCAL_CTX_CAP}") is None

    def test_empty_string_returns_none(self) -> None:
        assert _coerce_context_window_cap("") is None
        assert _coerce_context_window_cap("   ") is None

    def test_garbage_returns_none(self) -> None:
        assert _coerce_context_window_cap("abc") is None


class TestDiscoverContextWindowCap:
    """The cap must be applied regardless of whether it arrives as int or str."""

    def setup_method(self) -> None:
        self._patch_resolve()

    def teardown_method(self) -> None:
        patch.stopall()

    def test_int_cap_clamps(self) -> None:
        cw = discover_context_window("test/model", context_window_cap=30000)
        assert cw.max_input == 30000
        assert cw.max_output == 30000

    def test_string_cap_clamps(self) -> None:
        # Regression: this used to raise TypeError.
        cw = discover_context_window("test/model", context_window_cap="30000")
        assert cw.max_input == 30000
        assert cw.max_output == 30000

    def test_unexpanded_env_placeholder_ignored(self) -> None:
        # Regression: unset env var left a literal ${...} — must not crash.
        cw = discover_context_window("test/model", context_window_cap="${MISSING_VAR}")
        assert cw.max_input == 100000
        assert cw.max_output == 50000

    def test_none_cap_no_clamp(self) -> None:
        cw = discover_context_window("test/model", context_window_cap=None)
        assert cw.max_input == 100000
        assert cw.max_output == 50000
        assert cw.source == ContextSource.MODELS_DEV

    def _patch_resolve(self) -> None:
        # Bypass network/catalog lookups; supply deterministic limits.
        patcher = patch(
            "knox.llm.context_window.resolve_model_limits",
            return_value=(100000, 50000),
        )
        patcher.start()


# ── _split_model ───────────────────────────────────────────────────────


class TestSplitModel:
    def test_slash_separator(self) -> None:
        assert _split_model("my-llama/mistral-nemo") == ("my-llama", "mistral-nemo")

    def test_colon_separator(self) -> None:
        assert _split_model("my-llama:mistral-nemo") == ("my-llama", "mistral-nemo")

    def test_multi_slash(self) -> None:
        assert _split_model("my-llama/hf.co/qwen3") == ("my-llama", "hf.co/qwen3")

    def test_slash_with_colon_in_model(self) -> None:
        assert _split_model("my-llama/gemma4:e2b") == ("my-llama", "gemma4:e2b")

    def test_no_separator_returns_empty_provider(self) -> None:
        assert _split_model("justamodel") == ("", "justamodel")


# ── discover_context_window: user-defined providers ───────────────────


class TestDiscoverContextWindowUserProvider:
    """User-defined providers supply context_window_cap from config."""

    def setup_method(self) -> None:
        set_user_providers(
            {
                "my-llama": {
                    "base_url": "http://localhost:8081/v1",
                    "context_window_cap": 32768,
                },
                "no-cap": {
                    "base_url": "http://localhost:9999/v1",
                },
            }
        )

    def teardown_method(self) -> None:
        set_user_providers({})

    def test_user_provider_cap_used(self) -> None:
        """User-defined provider with context_window_cap uses that value."""
        cw = discover_context_window("my-llama/mistral-nemo")
        assert cw.max_input == 32768
        assert cw.max_output == 32768
        assert cw.source == ContextSource.FALLBACK

    def test_user_provider_no_cap_falls_through(self) -> None:
        """User-defined provider without context_window_cap falls through to fallback."""
        cw = discover_context_window("no-cap/some-model", fallback_max_input=64000, fallback_max_output=32000)
        assert cw.max_input == 64000
        assert cw.max_output == 32000
        assert cw.source == ContextSource.FALLBACK

    def test_user_provider_cap_with_global_cap(self) -> None:
        """Global context_window_cap can further clamp the user-provider cap."""
        cw = discover_context_window("my-llama/mistral-nemo", context_window_cap=16000)
        assert cw.max_input == 16000
        assert cw.max_output == 16000

    def test_user_provider_unknown_falls_through(self) -> None:
        """Unknown provider falls through to fallback."""
        cw = discover_context_window("unknown/model", fallback_max_input=64000, fallback_max_output=32000)
        assert cw.max_input == 64000
        assert cw.max_output == 32000
        assert cw.source == ContextSource.FALLBACK
