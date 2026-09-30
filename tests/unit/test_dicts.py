"""Tests for the dotted-key helpers in ``knox.core.utils.dicts``.

``resolve_dotted`` exists because ``get_dot`` cannot see a dict key that itself
contains dots — the case that broke ``knox config check`` for provider model
names (``models."hf.co/qwen3.6"``). Only that helper is covered here; the
others are exercised through the config engine.
"""

from __future__ import annotations

from knox.core.utils.dicts import resolve_dotted


def test_plain_path_matches_get_dot() -> None:
    data = {"compile": {"llm": {"timeout": 120}}}
    assert resolve_dotted(data, "compile.llm.timeout") == (True, 120)


def test_key_containing_dots_resolves() -> None:
    """The dotted name is one key, not three levels of nesting."""
    data = {"providers": {"ollama": {"models": {"hf.co/qwen3.6": {"context_window": 262144}}}}}
    assert resolve_dotted(data, "providers.ollama.models.hf.co/qwen3.6.context_window") == (True, 262144)


def test_nested_dotted_keys_at_several_levels() -> None:
    data = {"a.b": {"c.d": {"e": 1}}}
    assert resolve_dotted(data, "a.b.c.d.e") == (True, 1)


def test_ambiguous_path_prefers_the_longer_key() -> None:
    """``a.b`` as a key wins over nesting through ``a`` → ``b``."""
    data = {"a": {"b": "nested"}, "a.b": "flat"}
    assert resolve_dotted(data, "a.b") == (True, "flat")


def test_absent_key_reports_not_found() -> None:
    """No match returns ``(False, None)`` rather than raising."""
    data = {"compile": {"workers": 1}}
    assert resolve_dotted(data, "compile.nope") == (False, None)
    assert resolve_dotted(data, "nope.workers") == (False, None)


def test_descends_through_a_scalar_is_not_found() -> None:
    data = {"compile": {"workers": 1}}
    assert resolve_dotted(data, "compile.workers.deeper") == (False, None)
