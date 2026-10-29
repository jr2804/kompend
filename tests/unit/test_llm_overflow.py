"""Tests for the provider-overflow handling.

A provider that rejects an oversized request names the limit it enforces.
These tests pin the three things that make that number usable: the overflow
is recognised across providers, the limit is read out of the message, and a
stage that hits one lowers its window and re-plans instead of dying.
"""

from __future__ import annotations

import pytest

from kompend.core.utils.exceptions import LLMContextOverflowError
from kompend.llm.context_window import ContextSource, ContextWindow
from kompend.llm.overflow import is_context_overflow, stated_context_size
from kompend.llm.request_budget import EffectiveWindow

# -- recognising the overflow -------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "request (33145 tokens) exceeds the available context size (32768 tokens), try increasing it",
        "maximum context length is 8192 tokens",
        "This model's maximum context length is 4096 tokens",
        "context_length_exceeded",
        "Please reduce the length of the messages",
    ],
)
def test_overflow_is_recognised_across_providers(message: str) -> None:
    assert is_context_overflow(RuntimeError(message))


def test_a_different_failure_is_not_an_overflow() -> None:
    assert not is_context_overflow(RuntimeError("500 internal server error"))


def test_the_overflow_is_found_through_the_exception_chain() -> None:
    """Gateway errors arrive wrapped — the original carries the message."""
    with pytest.raises(ValueError, match="provider call failed") as caught:
        _wrapped_overflow()
    assert is_context_overflow(caught.value)
    assert stated_context_size(caught.value) == 32768


def _wrapped_overflow() -> ValueError:
    """A gateway error that wraps the original overflow message."""
    try:
        msg = "request (33145 tokens) exceeds the available context size (32768 tokens)"
        raise RuntimeError(msg)
    except RuntimeError as inner:
        raise ValueError("provider call failed") from inner


# -- reading the provider's limit ----------------------------------------------


def test_the_stated_limit_is_read_from_the_message() -> None:
    exc = RuntimeError("request (33145 tokens) exceeds the available context size (32768 tokens), try increasing it")
    assert stated_context_size(exc) == 32768


def test_a_thousand_separator_does_not_break_the_limit() -> None:
    exc = RuntimeError("maximum context length is 131,072 tokens")
    assert stated_context_size(exc) == 131072


def test_no_limit_in_the_message_means_no_limit() -> None:
    """Without a number kompend cannot plan — the caller must not guess."""
    exc = RuntimeError("request exceeds the available context size")
    assert stated_context_size(exc) is None


def test_the_requested_size_is_not_mistaken_for_the_limit() -> None:
    """Only the server's limit may lower the window, never the request size."""
    exc = RuntimeError("request (33145 tokens) exceeds the available context size (32768 tokens)")
    limit = stated_context_size(exc)
    assert limit == 32768
    assert limit != 33145


# -- the effective window -----------------------------------------------------


def test_the_window_starts_at_the_discovered_one() -> None:
    window = _window()
    effective = EffectiveWindow(window)
    assert effective.current == window
    assert not effective.clamped


def test_a_provider_limit_lowers_the_window() -> None:
    effective = EffectiveWindow(_window(max_input=64000))
    assert effective.clamp(32768, reason="provider enforces 32768 tokens")
    assert effective.current.max_input == 32768
    assert effective.clamped
    assert effective.reason == "provider enforces 32768 tokens"


def test_the_window_is_never_raised_again() -> None:
    """A provider that rejected a request has rejected the evidence for more."""
    effective = EffectiveWindow(_window(max_input=64000))
    effective.clamp(32768, reason="provider enforces 32768 tokens")
    assert not effective.clamp(64000, reason="provider enforces 64000 tokens")
    assert effective.current.max_input == 32768


def test_a_limit_above_the_current_window_is_ignored() -> None:
    effective = EffectiveWindow(_window(max_input=32768))
    assert not effective.clamp(64000, reason="noise")
    assert effective.current.max_input == 32768


def test_the_discovered_window_is_remembered() -> None:
    """The log line names both numbers — what was planned against, what the server allows."""
    effective = EffectiveWindow(_window(max_input=64000))
    effective.clamp(32768, reason="provider enforces 32768 tokens")
    assert effective.discovered is not None
    assert effective.discovered.max_input == 64000
    assert "64000" in effective.report()
    assert "32768" in effective.report()


def test_the_clamped_window_is_marked_as_a_fallback() -> None:
    """A clamped window is no longer the discovered one — the source says so."""
    effective = EffectiveWindow(_window(max_input=64000))
    effective.clamp(32768, reason="provider enforces 32768 tokens")
    assert effective.current.source == ContextSource.FALLBACK


# -- the exception the agent base raises ---------------------------------------


def test_the_overflow_exception_carries_the_limit() -> None:
    exc = LLMContextOverflowError("too big", stated_limit=32768)
    assert exc.stated_limit == 32768


def test_the_overflow_exception_may_carry_no_limit() -> None:
    exc = LLMContextOverflowError("too big")
    assert exc.stated_limit is None


def _window(max_input: int = 65536, max_output: int = 8192) -> ContextWindow:
    return ContextWindow(
        max_input=max_input,
        max_output=max_output,
        source=ContextSource.OLLAMA,
    )
