"""The per-model circuit breaker: a quota-dead model is not asked again.

A provider allowance that dies mid-run stays dead for the rest of the run,
but the pipeline cannot tell that from a per-minute throttle by status alone
— both surface as 429. These tests pin the breaker's contract: quota bodies
trip unconditionally, plain throttles keep retrying, a run of identical
failures trips past one full retry cycle, and a tripped model refuses
without sending.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from kompend.agents.base import KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps
from kompend.core.config._configs import LLMConfig
from kompend.core.utils.exceptions import LLMTransientError
from kompend.llm.breaker import BreakerOpenError, ModelBreaker, breaker_for_tracker
from kompend.llm.errors import is_quota_refusal, is_transient_error, quota_cause
from kompend.llm.usage import UsageTracker

_QUOTA_BODY = "you (janreimes) have reached your weekly usage limit. Upgrade for higher limits or add usage credits to your plan."


_DEPS = PipelineAgentDeps(text="source body", source_id="src-1")


class TestModelBreaker:
    def test_trip_opens_and_refuses(self, caplog: pytest.LogCaptureFixture) -> None:
        breaker = ModelBreaker()
        with caplog.at_level(logging.WARNING):
            breaker.trip("provider/model", "quota refusal (429)", stage="seg_0")
        assert breaker.is_open("provider/model") is True
        with pytest.raises(BreakerOpenError):
            breaker.check("provider/model")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, "one dead model logs one WARNING, not one per call"

    def test_second_trip_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        breaker = ModelBreaker()
        with caplog.at_level(logging.WARNING):
            breaker.trip("provider/model", "quota", stage="a")
            breaker.trip("provider/model", "quota", stage="b")
        assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1

    def test_run_of_identical_failures_trips_past_one_full_cycle(self) -> None:
        """Budget 2: the loop's own 2 retries ride first; the 3rd failure trips."""
        breaker = ModelBreaker()
        assert breaker.record_failure("m", TimeoutError("a"), 2) is False
        assert breaker.record_failure("m", TimeoutError("a"), 2) is False
        assert breaker.record_failure("m", TimeoutError("a"), 2) is True
        assert breaker.is_open("m") is True

    def test_a_different_failure_restarts_the_run(self) -> None:
        breaker = ModelBreaker()
        assert breaker.record_failure("m", TimeoutError("a"), 2) is False
        assert breaker.record_failure("m", TimeoutError("a"), 2) is False
        assert breaker.record_failure("m", ValueError("b"), 2) is False
        assert breaker.is_open("m") is False

    def test_success_resets_the_run(self) -> None:
        breaker = ModelBreaker()
        breaker.record_failure("m", TimeoutError("a"), 2)
        breaker.record_failure("m", TimeoutError("a"), 2)
        breaker.record_success("m")
        breaker.record_failure("m", TimeoutError("a"), 2)
        assert breaker.is_open("m") is False

    def test_zero_budget_still_needs_a_pair(self) -> None:
        """A lone blip never condemns a model, even with retries set to 0."""
        breaker = ModelBreaker()
        assert breaker.record_failure("m", TimeoutError("a"), 0) is False
        assert breaker.record_failure("m", TimeoutError("a"), 0) is True

    def test_models_trip_independently(self) -> None:
        breaker = ModelBreaker()
        breaker.trip("dead/model", "quota", stage="a")
        assert breaker.is_open("other/model") is False


class TestBreakerRegistry:
    def test_same_tracker_shares_one_breaker(self) -> None:
        tracker = UsageTracker()
        assert breaker_for_tracker(tracker) is breaker_for_tracker(tracker)

    def test_trackers_do_not_share_breakers(self) -> None:
        first, second = UsageTracker(), UsageTracker()
        assert breaker_for_tracker(first) is not breaker_for_tracker(second)

    def test_no_tracker_means_no_breaker(self) -> None:
        assert breaker_for_tracker(None) is None

    def test_dead_tracker_ids_are_not_reused(self) -> None:
        tracker = UsageTracker()
        old = breaker_for_tracker(tracker)
        old_key = id(tracker)
        del tracker, old
        fresh = UsageTracker()
        if id(fresh) == old_key:  # pragma: no cover - allocator dependent
            assert breaker_for_tracker(fresh) is not None


class _ScriptedAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """Runs the real retry loop against a scripted model: errors, then results."""

    template_name: ClassVar[str] = ""
    system_template_name: ClassVar[str] = ""

    def __init__(self, script: list[Any], **kwargs: Any) -> None:
        super().__init__(model_string="test/model", **kwargs)
        self._script = script
        self.calls = 0

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:
        return deps.text

    def render_system_prompt(self, deps: PipelineAgentDeps) -> str:
        return ""

    def _build_agent(self) -> Any:
        calls = self

        class _Run:
            def run_sync(self, *_args: Any, **_kwargs: Any) -> Any:
                calls.calls += 1
                outcome = calls._script.pop(0) if calls._script else "ok"
                if isinstance(outcome, BaseException):
                    raise outcome
                return SimpleNamespace(output=outcome)

        return _Run()


def test_quota_body_on_429_is_a_quota_refusal() -> None:
    assert is_quota_refusal(_quota_error()) is True


def test_plain_throttle_on_429_is_not_a_quota_refusal() -> None:
    """'Too many requests' with no quota wording stays retryable."""
    assert is_quota_refusal(_throttle_error()) is False
    assert is_transient_error(_throttle_error()) is True


def test_quota_words_without_429_are_not_a_quota_refusal() -> None:
    """A 5xx that mentions quota is a server fault, not an allowance death."""
    err = ModelHTTPError(503, "test-model", body="quota backend exploded")
    assert is_quota_refusal(err) is False


def test_quota_cause_names_the_marker() -> None:
    assert "usage limit" in quota_cause(_quota_error())


def test_breaker_refusal_is_never_transient() -> None:
    """A refusal raised inside a transient handler still does not retry."""
    refusal = BreakerOpenError("refused")
    refusal.__cause__ = TimeoutError("read timed out")
    assert is_transient_error(refusal) is False


def test_quota_trips_and_second_run_sends_nothing() -> None:
    """After the breaker trips, no further request goes to that model."""
    tracker = UsageTracker()
    agent = _ScriptedAgent([_quota_error()], llm=_llm(), usage_tracker=tracker)
    with pytest.raises(BreakerOpenError):
        agent.run(_DEPS, user_prompt="x")
    assert agent.calls == 1
    with pytest.raises(BreakerOpenError):
        agent.run(_DEPS, user_prompt="x")
    assert agent.calls == 1, "the tripped model must not be asked again"


def _quota_error() -> ModelHTTPError:
    """The provider's quota death: 429 whose body names the allowance."""
    return ModelHTTPError(429, "test-model", body=_QUOTA_BODY)


def test_repeated_identical_failures_trip_across_runs() -> None:
    """Three timeouts burn one full cycle; the next run refuses up front."""
    tracker = UsageTracker()
    agent = _ScriptedAgent([TimeoutError("t")] * 9, llm=_llm(), usage_tracker=tracker)
    with pytest.raises(LLMTransientError):
        agent.run(_DEPS, user_prompt="x")
    assert agent.calls == 3, "one attempt plus two configured retries, exactly as before"
    with pytest.raises(BreakerOpenError):
        agent.run(_DEPS, user_prompt="x")
    assert agent.calls == 3, "no further request after the trip"


def test_transient_blip_that_recovers_never_trips() -> None:
    """Two failures inside the budget followed by success: no trip, no refusal."""
    tracker = UsageTracker()
    agent = _ScriptedAgent([TimeoutError("a"), TimeoutError("b"), "fine"], llm=_llm(), usage_tracker=tracker)
    agent.run(_DEPS, user_prompt="x")
    assert agent.calls == 3
    agent.run(_DEPS, user_prompt="x")
    assert agent.calls == 4, "a recovered model keeps serving"


def test_throttle_keeps_retrying_without_tripping() -> None:
    """Plain 429s ride the loop's own retries and never trip the breaker."""
    tracker = UsageTracker()
    agent = _ScriptedAgent([_throttle_error(), _throttle_error(), "fine"], llm=_llm(), usage_tracker=tracker)
    agent.run(_DEPS, user_prompt="x")
    assert agent.calls == 3


def _throttle_error() -> ModelHTTPError:
    """A per-minute throttle: 429 with no quota vocabulary."""
    return ModelHTTPError(429, "test-model", body="Rate limit exceeded, retry after 60s.")


def _llm(*, retries: int = 2) -> LLMConfig:
    return LLMConfig(transient_retries=retries, transient_backoff=0.0)
