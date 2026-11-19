"""Tests for the measured-throughput cost model and the per-request deadline.

The point of both is the same: a configured ceiling is a bound, not a plan. A
15-hour run (pyvas, 2026-11-18) sent 49k-token batches to a model measured at
~1.4 tokens/s against a 120 s timeout, and every one of them timed out at the
same size. These tests pin the arithmetic that turns a measured call into a
timeout, and the seam that applies it per request.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import UsageLimits
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel

from kompend.agents.base.deadline import MIN_TURN_TIMEOUT_S, DeadlineState, RequestDeadline
from kompend.agents.base.retries import session_deadline_s
from kompend.llm.throughput import (
    MIN_SAMPLES,
    MIN_TIMEOUT_S,
    SAFETY_FACTOR,
    CallSample,
    Throughput,
    fit,
    timeout_for,
)
from kompend.llm.usage import UsageTracker

#: Rates the synthetic calls are drawn from: 1,000 prompt tokens/s of prefill,
#: 50 completion tokens/s of decode.
_PREFILL = 1_000.0
_DECODE = 50.0


class TestFit:
    def test_the_fit_recovers_the_rates_the_calls_were_drawn_from(self) -> None:
        throughput = _fitted()

        assert throughput.prefill_rate == pytest.approx(_PREFILL, rel=0.02)
        assert throughput.decode_rate == pytest.approx(_DECODE, rel=0.02)

    def test_too_few_calls_fit_nothing(self) -> None:
        assert fit(_samples()[: MIN_SAMPLES - 1]) is None

    def test_identical_calls_fit_nothing(self) -> None:
        """Two unknowns need two directions; the same call twice has one."""
        same = _sample(1_000, 100)

        assert fit([same, same, same, same]) is None

    def test_a_stalled_call_does_not_establish_a_rate(self) -> None:
        """A call that took an hour is a stall, not a throughput measurement.

        The fit refuses rather than averaging it in, so the caller keeps its
        ceiling instead of planning against a rate no real call supports.
        """
        stalled = CallSample(prompt_tokens=1_000, completion_tokens=10, elapsed_s=3_600.0)

        assert fit([stalled, *(_samples()[: MIN_SAMPLES - 1])]) is None
        assert fit([stalled] * MIN_SAMPLES) is None


class TestTimeoutFor:
    def test_the_ceiling_wins_when_the_estimate_is_larger(self) -> None:
        assert timeout_for(_fitted(), 10_000_000, 1_000, ceiling_s=120.0) == 120.0

    def test_a_cheap_request_gets_a_shorter_timeout(self) -> None:
        """20k prompt tokens + 100 answer tokens is ~22 s at these rates."""
        estimate = 20_000 / _PREFILL + 100 / _DECODE

        timeout = timeout_for(_fitted(), 20_000, 100, ceiling_s=120.0)

        assert timeout == pytest.approx(estimate * SAFETY_FACTOR)

    def test_no_fit_leaves_the_ceiling_alone(self) -> None:
        assert timeout_for(None, 20_000, 100, ceiling_s=120.0) == 120.0

    def test_a_turn_is_never_given_less_than_the_floor(self) -> None:
        assert timeout_for(_fitted(), 10, 1, ceiling_s=120.0) == MIN_TIMEOUT_S

    def test_a_prompt_that_cannot_fit_gets_no_budget(self) -> None:
        assert _fitted().prompt_tokens_within(1.0, completion_tokens=10_000) == 0

    def test_a_prompt_that_fits_gets_a_positive_budget(self) -> None:
        assert _fitted().prompt_tokens_within(30.0, completion_tokens=100) > 0


class TestUsageSamples:
    def test_recorded_calls_fit_a_throughput(self) -> None:
        tracker = UsageTracker()
        for sample in _samples():
            tracker.record(
                prompt_tokens=sample.prompt_tokens,
                completion_tokens=sample.completion_tokens,
                model="ollama/test",
                elapsed_s=sample.elapsed_s,
            )

        throughput = tracker.throughput("ollama/test")

        assert throughput is not None
        assert throughput.samples == len(_samples())

    def test_a_call_without_a_timing_is_not_a_sample(self) -> None:
        tracker = UsageTracker()

        tracker.record(prompt_tokens=100, completion_tokens=10, model="ollama/test")

        assert tracker.samples_for("ollama/test") == []

    def test_samples_are_per_model(self) -> None:
        tracker = UsageTracker()
        for sample in _samples():
            tracker.record(
                prompt_tokens=sample.prompt_tokens,
                completion_tokens=sample.completion_tokens,
                model="ollama/a",
                elapsed_s=sample.elapsed_s,
            )

        assert tracker.throughput("ollama/b") is None


class TestDeadlineState:
    def test_an_unarmed_deadline_reports_nothing(self) -> None:
        assert DeadlineState().remaining() is None

    def test_arming_starts_the_clock(self) -> None:
        state = DeadlineState()

        state.arm(60.0)
        remaining = state.remaining()

        assert remaining is not None
        assert 0.0 < remaining <= 60.0

    def test_clearing_the_budget_removes_the_bound(self) -> None:
        state = DeadlineState()
        state.arm(60.0)

        state.arm(None)

        assert state.remaining() is None


class TestRequestDeadline:
    def test_the_fitted_timeout_replaces_the_ceiling(self) -> None:
        capability = RequestDeadline(lambda _messages: 17.5)

        result = asyncio.run(capability.before_model_request(None, self._context()))

        assert result.model_settings == {"timeout": 17.5}

    def test_the_original_context_is_not_mutated(self) -> None:
        context = self._context()
        capability = RequestDeadline(lambda _messages: 17.5)

        asyncio.run(capability.before_model_request(None, context))

        assert context.model_settings == {"timeout": 120.0}

    def test_no_estimate_leaves_the_request_untouched(self) -> None:
        context = self._context()
        capability = RequestDeadline(lambda _messages: None)

        result = asyncio.run(capability.before_model_request(None, context))

        assert result is context

    def _context(self) -> ModelRequestContext:
        return ModelRequestContext(
            model=TestModel(),
            messages=[],
            model_settings={"timeout": 120.0},
            model_request_parameters=ModelRequestParameters(),
        )


class _Host:
    """Minimal RetryHost stand-in: everything the deadline needs, nothing else."""

    llm = None
    _rlm_usage_limits = None
    timeout = 120.0
    model_string = "ollama/test"
    usage_tracker: UsageTracker | None = None


class TestSessionDeadline:
    def test_without_a_fit_every_turn_costs_the_ceiling(self) -> None:
        host = _Host()

        deadline = session_deadline_s(host, units=10, per_unit=2, headroom_frac=1.1, session_overhead=2)

        # ceil(22 * 1.1) = 25 turns, each priced at the 120 s ceiling
        assert deadline == pytest.approx(25 * 120.0)

    def test_a_measured_model_gets_a_shorter_session(self) -> None:
        tracker = UsageTracker()
        for sample in _samples():
            tracker.record(
                prompt_tokens=sample.prompt_tokens,
                completion_tokens=sample.completion_tokens,
                model="ollama/test",
                elapsed_s=sample.elapsed_s,
            )
        host = _Host()
        host.usage_tracker = tracker

        deadline = session_deadline_s(host, units=10, per_unit=2, headroom_frac=1.1, session_overhead=2)

        assert deadline is not None
        assert deadline < 25 * 120.0

    def test_an_rlm_session_keeps_its_own_bounds(self) -> None:
        host = _Host()
        host._rlm_usage_limits = UsageLimits(request_limit=1)

        assert session_deadline_s(host, units=10, per_unit=2) is None

    def test_nothing_to_plan_means_no_deadline(self) -> None:
        assert session_deadline_s(_Host(), units=0, per_unit=0, session_overhead=0) is None

    def test_the_turn_floor_is_the_minimum_a_session_can_plan_with(self) -> None:
        assert MIN_TURN_TIMEOUT_S > 0.0


def _fitted() -> Throughput:
    throughput = fit(_samples())
    assert throughput is not None
    return throughput


def _samples() -> list[CallSample]:
    return [_sample(prompt, completion) for prompt, completion in [(400, 50), (1_200, 120), (3_000, 90), (800, 200)]]


def _sample(prompt_tokens: int, completion_tokens: int) -> CallSample:
    """One call that took exactly what the model's rates predict."""
    return CallSample(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_s=prompt_tokens / _PREFILL + completion_tokens / _DECODE,
    )
