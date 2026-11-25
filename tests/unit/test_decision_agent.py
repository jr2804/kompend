"""Tests for the optional decision-model sidecar (agents.decision)."""

from __future__ import annotations

import pytest

from kompend.agents import decision as decision_module
from kompend.agents.decision import DecisionAgent, DecisionDeps, decide_error_action, decide_unit_adaptation
from kompend.agents.failure import AgentFailureKind
from kompend.compilation.adaptation import AdaptationSignal
from kompend.core.config._configs import DecisionConfig
from kompend.core.utils.exceptions import LLMTransientError
from kompend.llm.breaker import BreakerOpenError
from kompend.llm.decision import (
    DecisionState,
    ErrorAction,
    ErrorActionOutcome,
    ErrorDecision,
    UnitAdaptationOutcome,
    decision_model_string,
)


class TestDecisionModelString:
    def test_disabled_config_has_no_model(self) -> None:
        assert decision_model_string(DecisionConfig()) is None
        assert decision_model_string(None) is None

    def test_model_string_is_used_verbatim(self) -> None:
        """Any provider serves decision models - no hardcoded provider prefix."""
        assert decision_model_string(DecisionConfig(enabled=True, model="ollama/clef")) == "ollama/clef"
        assert decision_model_string(DecisionConfig(enabled=True, model="typesafe/jev-latest")) == "typesafe/jev-latest"

    def test_error_action_is_a_str_enum(self) -> None:
        assert ErrorAction.RETRY == "retry"
        assert isinstance(ErrorAction.SKIP, str)
        assert [action.value for action in ErrorAction] == ["retry", "skip", "abort"]


class TestDecideErrorAction:
    def test_disabled_returns_not_configured_without_building_an_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("no agent may be built when the sidecar is disabled")

        monkeypatch.setattr(decision_module, "DecisionAgent", _fail)
        outcome = decide_error_action(DecisionConfig(), error="boom", stage="s", attempt=1, max_attempts=3)
        assert outcome == ErrorActionOutcome(state=DecisionState.NOT_CONFIGURED)

    def test_action_comes_from_the_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(DecisionAgent, "run", lambda self, deps: ErrorDecision(action=ErrorAction.RETRY))
        config = DecisionConfig(enabled=True, model="jev-latest")
        outcome = decide_error_action(config, error="timeout", stage="s", attempt=1, max_attempts=3)
        assert outcome == ErrorActionOutcome(state=DecisionState.ANSWERED, action=ErrorAction.RETRY)

    def test_agent_failure_is_unavailable_not_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dead sidecar is UNAVAILABLE, so the caller can degrade the run and say so."""

        def _boom(self: DecisionAgent, deps: DecisionDeps) -> ErrorDecision:
            raise RuntimeError("sidecar down")

        monkeypatch.setattr(DecisionAgent, "run", _boom)
        config = DecisionConfig(enabled=True, model="jev-latest")
        outcome = decide_error_action(config, error="timeout", stage="s", attempt=1, max_attempts=3)
        assert outcome == ErrorActionOutcome(state=DecisionState.UNAVAILABLE)

    def test_quota_dead_primary_fails_over_to_the_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fallback chain entry answers; the outcome is still ANSWERED."""
        seen: list[str] = []

        def _run(self: DecisionAgent, deps: DecisionDeps) -> ErrorDecision:
            seen.append(self.model_string or "?")
            if "primary" in (self.model_string or ""):
                raise BreakerOpenError("primary is quota-dead")
            return ErrorDecision(action=ErrorAction.SKIP)

        monkeypatch.setattr(DecisionAgent, "run", _run)
        config = DecisionConfig(enabled=True, model="typesafe/primary")
        outcome = decide_error_action(
            config,
            error="quota",
            stage="s",
            attempt=1,
            max_attempts=3,
            fallback_models=["typesafe/primary", "ollama/local"],
        )
        assert outcome == ErrorActionOutcome(state=DecisionState.ANSWERED, action=ErrorAction.SKIP)
        assert seen == ["typesafe/primary", "ollama/local"]

    def test_failed_failover_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(self: DecisionAgent, deps: DecisionDeps) -> ErrorDecision:
            raise LLMTransientError("everything is down")

        monkeypatch.setattr(DecisionAgent, "run", _boom)
        config = DecisionConfig(enabled=True, model="typesafe/primary")
        outcome = decide_error_action(
            config,
            error="down",
            stage="s",
            attempt=1,
            max_attempts=3,
            fallback_models=["ollama/local"],
        )
        assert outcome == ErrorActionOutcome(state=DecisionState.UNAVAILABLE)

    def test_config_error_does_not_fail_over(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bad configuration must stay visible, not be masked by a fallback."""
        seen: list[str] = []

        def _run(self: DecisionAgent, deps: DecisionDeps) -> ErrorDecision:
            seen.append(self.model_string or "?")
            raise ValueError("unknown provider 'typo'")

        monkeypatch.setattr(DecisionAgent, "run", _run)
        config = DecisionConfig(enabled=True, model="typo/model")
        outcome = decide_error_action(config, error="bad", stage="s", attempt=1, max_attempts=3, fallback_models=["ollama/local"])
        assert outcome == ErrorActionOutcome(state=DecisionState.UNAVAILABLE)
        assert seen == ["typo/model"]


class TestDecideUnitAdaptation:
    def test_disabled_returns_not_configured(self) -> None:
        outcome = decide_unit_adaptation(DecisionConfig(), signal=self._signal(), stage="s", max_adaptations=2)
        assert outcome == UnitAdaptationOutcome(state=DecisionState.NOT_CONFIGURED)

    def test_quota_dead_primary_fails_over_to_the_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []

        def _run(self: DecisionAgent, deps: DecisionDeps) -> ErrorDecision:
            seen.append(self.model_string or "?")
            if "primary" in (self.model_string or ""):
                raise BreakerOpenError("primary is quota-dead")
            return ErrorDecision(action=ErrorAction.RETRY, shrink=True)

        monkeypatch.setattr(decision_module.AdaptationAgent, "run", _run)
        config = DecisionConfig(enabled=True, model="typesafe/primary")
        outcome = decide_unit_adaptation(config, signal=self._signal(), stage="s", max_adaptations=2, fallback_models=["ollama/local"])
        assert outcome.state is DecisionState.ANSWERED
        assert outcome.decision is not None
        assert outcome.decision.shrink is True
        assert seen == ["typesafe/primary", "ollama/local"]

    def _signal(self) -> AdaptationSignal:
        return AdaptationSignal(
            seg_id="seg_0",
            kind=AgentFailureKind.OTHER,
            adaptations_used=0,
            concepts_assigned=4,
            concepts_remaining=4,
            fetches=2,
            context_tokens=None,
            context_budget=None,
            last_error="boom",
        )
