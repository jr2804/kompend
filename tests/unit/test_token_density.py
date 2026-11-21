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
from kompend.llm.request_budget import (
    DEFAULT_CHARS_PER_TOKEN,
    RequestBudget,
    calibrated_density,
    calibrated_density_for,
    chars_to_tokens,
    sizing_density,
)
from kompend.llm.usage import UsageTracker


class TestCalibratedDensity:
    """The measured density tightens sizing; it never loosens it.

    The floor is the worst case — it keeps a planned request inside the
    window for every density the corpus shows. The run's own samples are
    the better-informed signal: a deployed tokenizer denser than the floor
    is caught before it overflows a window the floor said was safe.
    Replacing the floor with an optimistic measurement would invert the
    safety direction, so the blend is the minimum of the two.
    """

    def test_no_measurement_keeps_the_floor(self) -> None:
        assert calibrated_density(2.5, None) == CONSERVATIVE_CHARS_PER_TOKEN
        assert calibrated_density(None, None) == CONSERVATIVE_CHARS_PER_TOKEN

    def test_a_denser_tokenizer_tightens_sizing(self) -> None:
        assert calibrated_density(2.5, 2.0) == 2.0

    def test_a_sparser_tokenizer_never_loosens_sizing(self) -> None:
        assert calibrated_density(2.5, 3.8) == CONSERVATIVE_CHARS_PER_TOKEN

    def test_configured_pessimism_still_wins(self) -> None:
        assert calibrated_density(2.0, 2.2) == 2.0

    def test_two_conservative_signals_take_the_tighter(self) -> None:
        assert calibrated_density(2.0, 1.8) == 1.8

    def test_without_a_tracker_the_floor_sizes(self) -> None:
        assert calibrated_density_for(LLMConfig(), None, "m") == CONSERVATIVE_CHARS_PER_TOKEN

    def test_measured_calls_tighten_the_stage(self) -> None:
        tracker = UsageTracker()
        for _ in range(5):
            tracker.record(prompt_tokens=1000, completion_tokens=10, model="m", elapsed_s=1.0, prompt_chars=2000)
        assert calibrated_density_for(LLMConfig(), tracker, "m") == 2.0

    def test_too_few_samples_keep_the_floor(self) -> None:
        tracker = UsageTracker()
        for _ in range(4):
            tracker.record(prompt_tokens=1000, completion_tokens=10, model="m", elapsed_s=1.0, prompt_chars=2000)
        assert calibrated_density_for(LLMConfig(), tracker, "m") == CONSERVATIVE_CHARS_PER_TOKEN

    def test_an_optimistic_measurement_never_sizes(self) -> None:
        tracker = UsageTracker()
        for _ in range(5):
            tracker.record(prompt_tokens=1000, completion_tokens=10, model="m", elapsed_s=1.0, prompt_chars=3800)
        assert calibrated_density_for(LLMConfig(), tracker, "m") == CONSERVATIVE_CHARS_PER_TOKEN


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


def test_a_dense_segment_can_no_longer_slip_past_the_guard() -> None:
    """The regression, closed by construction rather than by a second unit.

    The old planner filled the content budget at 3.85 chars per token while
    the guard counted the same segment at 2.5, so a request the planner called
    exact-fit was over the window - and the refusal arrived too late to save
    the run. Sizing now always counts at the floor, so planner and guard cannot
    disagree, and the segment the optimistic plan would have sent is refused
    before the provider sees it.
    """
    budget = RequestBudget(window=_window())
    chars = int(budget.content_tokens * 3.85)
    planned = chars_to_tokens(chars, chars_per_token=3.85)
    counted = chars_to_tokens(chars, chars_per_token=CONSERVATIVE_CHARS_PER_TOKEN)
    assert planned == counted
    assert planned > budget.content_tokens
    with pytest.raises(LLMOutputError):
        budget.require(counted, stage="Agentic generation")


def _window(max_input: int = 131_072) -> ContextWindow:
    """A window like the one the p863 vault compiles with."""
    return ContextWindow(max_input=max_input, max_output=8_192, source=ContextSource.MODELS_DEV)


def test_a_configured_ratio_below_the_floor_moves_the_planner_and_the_guard_together() -> None:
    """Going denser than the floor is honoured: the estimate stays conservative."""
    config = LLMConfig(default_chars_per_token=2.0)
    assert config.default_chars_per_token == 2.0
    assert chars_to_tokens(2_000, chars_per_token=config.default_chars_per_token) == 1_000


def test_a_configured_ratio_above_the_floor_is_clamped() -> None:
    """An optimistic ratio cannot size a request - the floor wins."""
    assert chars_to_tokens(3_000, chars_per_token=3.0) == 1_200
    assert sizing_density(3.85) == CONSERVATIVE_CHARS_PER_TOKEN


def test_the_compaction_estimator_counts_at_the_shared_ratio() -> None:
    assert conservative_tokens("x" * 250) == 100
    assert conservative_tokens("") == 0
