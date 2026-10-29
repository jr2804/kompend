"""Tests for the shared request-fit planner.

Every stage that sends a request to a model sizes it against the same
window. These tests pin the planner's arithmetic: the answer reserve and the
safety margin are always paid, the reserved payloads are named and summed,
and a request that cannot fit fails loudly instead of being cropped.
"""

from __future__ import annotations

import pytest

from kompend.compilation._two_phase_core import context_safety_tokens as reexported_safety_tokens
from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.context_window import ContextSource, ContextWindow
from kompend.llm.request_budget import (
    CONTEXT_SAFETY_FRACTION,
    CONTEXT_SAFETY_TOKENS,
    DEFAULT_CHARS_PER_TOKEN,
    RequestBudget,
    chars_to_tokens,
    context_safety_tokens,
    tokens_to_chars,
)

# -- the shared safety margin ------------------------------------------------


def test_safety_margin_is_a_fraction_of_the_window() -> None:
    assert context_safety_tokens(65536) == round(CONTEXT_SAFETY_FRACTION * 65536)


def test_safety_margin_has_a_floor_for_narrow_windows() -> None:
    assert context_safety_tokens(2048) == CONTEXT_SAFETY_TOKENS


def test_safety_margin_is_shared_by_every_stage() -> None:
    """The margin is defined once — the compilation layer re-exports it.

    A second definition is how the stages drifted apart on how much of the
    window was really available.
    """
    assert reexported_safety_tokens is context_safety_tokens


# -- chars/token conversion --------------------------------------------------


def test_conversion_round_trips_at_the_configured_ratio() -> None:
    assert tokens_to_chars(100, chars_per_token=4.0) == 400
    assert chars_to_tokens(400, chars_per_token=4.0) == 100


def test_conversion_defaults_to_the_shared_ratio() -> None:
    assert tokens_to_chars(100) == int(100 * DEFAULT_CHARS_PER_TOKEN)
    assert chars_to_tokens(int(100 * DEFAULT_CHARS_PER_TOKEN)) == 100


def test_conversion_never_returns_zero_tokens() -> None:
    """A non-empty string is at least one token; zero would hide content."""
    assert chars_to_tokens(1, chars_per_token=4.0) == 1


# -- the budget ---------------------------------------------------------------


def test_budget_pays_the_answer_reserve_and_the_safety_margin() -> None:
    window = _window()
    budget = RequestBudget(window=window, reserved={"system": 1200})

    assert budget.answer_reserve == window.answer_reserve
    assert budget.safety == context_safety_tokens(window.max_input)
    assert budget.reserved_total == 1200
    assert budget.available_tokens == window.max_input - window.answer_reserve - budget.safety - 1200
    assert budget.content_tokens == budget.available_tokens


def test_reserved_payloads_are_summed_and_named() -> None:
    budget = RequestBudget(window=_window(), reserved={"system": 100, "known_titles": 4000})
    assert budget.reserved_total == 4100
    assert "known_titles=4000" in budget.report()


def test_utilization_scales_the_content_budget() -> None:
    budget = RequestBudget(window=_window(), reserved={"system": 0}, utilization=0.5)
    assert budget.content_tokens == int(budget.available_tokens * 0.5)


def test_content_budget_is_never_negative() -> None:
    """An impossible budget yields 1 token, not a negative size.

    The failure is reported by :meth:`require`, not by a negative budget that
    would silently skip content.
    """
    budget = RequestBudget(window=_window(max_input=2048), reserved={"system": 100_000})
    assert budget.available_tokens < 0
    assert budget.content_tokens == 1


def test_fits_reports_whether_content_fits() -> None:
    budget = RequestBudget(window=_window(), reserved={"system": 1000})
    assert budget.fits(budget.content_tokens)
    assert not budget.fits(budget.content_tokens + 1)


def test_require_raises_and_names_the_components() -> None:
    budget = RequestBudget(window=_window(max_input=8192), reserved={"system": 1200})
    with pytest.raises(LLMOutputError) as excinfo:
        budget.require(budget.content_tokens + 1, stage="Extraction batch")

    message = str(excinfo.value)
    assert "Extraction batch" in message
    assert "8192-token context window" in message
    assert "system=1200" in message
    # The message must say that content is not cropped — that is the rule.
    assert "never cropped" in message


def test_require_is_silent_when_the_content_fits() -> None:
    budget = RequestBudget(window=_window(), reserved={"system": 1000})
    budget.require(budget.content_tokens, stage="Extraction batch")


def test_content_chars_uses_the_configured_ratio() -> None:
    budget = RequestBudget(window=_window(), reserved={"system": 0})
    assert budget.content_chars(4.0) == budget.content_tokens * 4


def test_explicit_answer_and_safety_override_the_window_defaults() -> None:
    """A stage with a measured reason to differ may pass its own numbers."""
    window = _window()
    budget = RequestBudget(window=window, reserved={}, answer_tokens=1024, safety_tokens=0)
    assert budget.answer_reserve == 1024
    assert budget.safety == 0
    assert budget.available_tokens == window.max_input - 1024


def _window(max_input: int = 65536, max_output: int = 8192) -> ContextWindow:
    return ContextWindow(
        max_input=max_input,
        max_output=max_output,
        source=ContextSource.FALLBACK,
    )
