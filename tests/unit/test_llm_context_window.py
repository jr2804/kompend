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
    discover_context_window,
)


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
