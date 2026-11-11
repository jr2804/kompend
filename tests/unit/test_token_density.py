"""Tests for the shared chars-per-token heuristic.

A request is sized in characters and paid for in tokens. The planner, the
request-fit guard and the compaction estimator must count in the same unit:
when the planner sized a segment with one ratio (3.85) and the guard counted
it with another (2.5), the segment arrived roughly 54 % over the window and
the provider answered "prompt too long" after minutes of upstream work.
"""

from __future__ import annotations

import pytest

from kompend.core.config.models.llm import LLMConfig
from kompend.core.text.density import CONSERVATIVE_CHARS_PER_TOKEN, conservative_tokens
from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.context_window import ContextSource, ContextWindow
from kompend.llm.request_budget import DEFAULT_CHARS_PER_TOKEN, RequestBudget, chars_to_tokens


def test_the_default_ratio_is_the_pessimistic_density_bound() -> None:
    """One number owns the heuristic — the nominal median is gone."""
    assert CONSERVATIVE_CHARS_PER_TOKEN == 2.5
    assert DEFAULT_CHARS_PER_TOKEN == CONSERVATIVE_CHARS_PER_TOKEN
    assert LLMConfig().default_chars_per_token == CONSERVATIVE_CHARS_PER_TOKEN


def test_the_planner_and_the_compaction_estimator_count_in_the_same_unit() -> None:
    text = "dense content " * 10_000
    planned = chars_to_tokens(len(text), chars_per_token=LLMConfig().default_chars_per_token)
    counted = conservative_tokens(text)
    assert counted - planned <= 1


def test_a_dense_segment_is_refused_before_the_provider_sees_it() -> None:
    """The regression: the guard counted in a different unit than the planner.

    The old planner filled the content budget at 3.85 chars per token. The
    guard counted the same segment at 2.5, so it was over the window — and
    the guard's refusal came too late to save the run.
    """
    budget = RequestBudget(window=_window())
    chars = int(budget.content_tokens * 3.85)
    optimistic = chars_to_tokens(chars, chars_per_token=3.85)
    pessimistic = chars_to_tokens(chars, chars_per_token=CONSERVATIVE_CHARS_PER_TOKEN)
    assert optimistic <= budget.content_tokens
    assert pessimistic > budget.content_tokens
    with pytest.raises(LLMOutputError):
        budget.require(pessimistic, stage="Agentic generation")


def _window(max_input: int = 131_072) -> ContextWindow:
    """A window like the one the p863 vault compiles with."""
    return ContextWindow(max_input=max_input, max_output=8_192, source=ContextSource.MODELS_DEV)


def test_a_configured_ratio_moves_the_planner_and_the_guard_together() -> None:
    config = LLMConfig(default_chars_per_token=3.0)
    assert config.default_chars_per_token == 3.0
    assert chars_to_tokens(3_000, chars_per_token=config.default_chars_per_token) == 1_000


def test_the_compaction_estimator_counts_at_the_shared_ratio() -> None:
    assert conservative_tokens("x" * 250) == 100
    assert conservative_tokens("") == 0
