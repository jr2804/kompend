"""Tests for KNOX FailoverChain."""

from __future__ import annotations

from typing import Any

import pytest

from knox.llm.errors import (
    ConfigurationError,
    PermanentError,
    TransientError,
)
from knox.llm.failover import CallDiagnostics, FailoverChain


class _FakeExc(Exception):
    """Stand-in for a litellm exception."""


def _exc(name: str) -> Exception:
    cls = type(name, (_FakeExc,), {})
    return cls("boom")


class _RecordingSleep:
    """Captures sleep durations without actually sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _chain(models: list[str], **kwargs: Any) -> tuple[FailoverChain, _RecordingSleep]:
    sleeper = _RecordingSleep()
    chain = FailoverChain(
        models=models,
        sleep=sleeper,
        initial_backoff_s=1.0,
        max_backoff_s=8.0,
        **kwargs,
    )
    return chain, sleeper


class TestFailoverChainConstruction:

    def test_empty_models_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            FailoverChain(models=[])


class TestSingleModelHappyPath:

    def test_first_attempt_succeeds(self) -> None:
        chain, sleeper = _chain(["m1"])
        result = chain.run(lambda model: f"ok-{model}")
        assert result.value == "ok-m1"
        assert result.diagnostics.model_used == "m1"
        assert result.diagnostics.attempts == 1
        assert sleeper.calls == []


class TestTransientRetry:

    def test_retries_then_succeeds(self) -> None:
        chain, sleeper = _chain(["m1"], max_retries_per_model=2)
        calls = []

        def call(model: str) -> str:
            calls.append(model)
            if len(calls) < 3:
                raise _exc("RateLimitError")
            return "ok"

        result = chain.run(call)
        assert result.value == "ok"
        assert result.diagnostics.attempts == 3
        # Backoff: 1s, 2s
        assert sleeper.calls == [1.0, 2.0]

    def test_backoff_capped_at_max(self) -> None:
        chain, sleeper = _chain(["m1"], max_retries_per_model=5)

        attempts: list[int] = []

        def call(model: str) -> str:
            attempts.append(1)
            if len(attempts) < 6:
                raise _exc("RateLimitError")
            return "ok"

        chain.run(call)
        # 1, 2, 4, 8, 8 (capped)
        assert sleeper.calls == [1.0, 2.0, 4.0, 8.0, 8.0]


class TestChainAdvancement:

    def test_advances_to_next_model_after_transient_exhaustion(self) -> None:
        chain, sleeper = _chain(["m1", "m2"], max_retries_per_model=1)
        seen: list[str] = []

        def call(model: str) -> str:
            seen.append(model)
            if model == "m1":
                raise _exc("RateLimitError")
            return f"ok-{model}"

        result = chain.run(call)
        assert result.value == "ok-m2"
        assert result.diagnostics.model_used == "m2"
        # m1 tried twice (1 + 1 retry) then m2 succeeds on first try
        assert seen == ["m1", "m1", "m2"]
        assert result.diagnostics.attempts == 3

    def test_permanent_error_advances_without_retrying(self) -> None:
        chain, sleeper = _chain(["m1", "m2"], max_retries_per_model=3)
        seen: list[str] = []

        def call(model: str) -> str:
            seen.append(model)
            if model == "m1":
                raise _exc("BadRequestError")
            return "ok"

        result = chain.run(call)
        # m1 raises Permanent → no retries, jump to m2
        assert seen == ["m1", "m2"]
        assert result.value == "ok"
        assert sleeper.calls == []  # no backoff for permanent

    def test_configuration_error_is_fatal(self) -> None:
        chain, sleeper = _chain(["m1", "m2"], max_retries_per_model=3)
        seen: list[str] = []

        def call(model: str) -> str:
            seen.append(model)
            raise _exc("AuthenticationError")

        with pytest.raises(ConfigurationError):
            chain.run(call)
        # Auth errors never retry, never advance
        assert seen == ["m1"]


class TestChainExhaustion:

    def test_all_transient_raises_transient(self) -> None:
        chain, _ = _chain(["m1", "m2"], max_retries_per_model=1)

        def call(model: str) -> str:
            raise _exc("RateLimitError")

        with pytest.raises(TransientError) as exc_info:
            chain.run(call)
        # 2 models × 2 attempts each = 4
        assert exc_info.value.attempts == 4

    def test_all_permanent_raises_permanent(self) -> None:
        chain, _ = _chain(["m1", "m2"], max_retries_per_model=3)

        def call(model: str) -> str:
            raise _exc("BadRequestError")

        with pytest.raises(PermanentError) as exc_info:
            chain.run(call)
        # No retries for permanent → one attempt per model
        assert exc_info.value.attempts == 2

    def test_diagnostics_present_on_success(self) -> None:
        chain, _ = _chain(["m1"])
        result = chain.run(lambda model: "ok")
        diagnostics: CallDiagnostics = result.diagnostics
        assert diagnostics.model_used == "m1"
        assert diagnostics.attempts == 1
        assert diagnostics.total_latency_s >= 0.0
