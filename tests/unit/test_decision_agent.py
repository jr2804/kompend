"""Tests for the optional decision-model sidecar (agents.decision)."""

from __future__ import annotations

import pytest

from kompend.agents import decision as decision_module
from kompend.agents.decision import DecisionAgent, DecisionDeps, decide_error_action
from kompend.core.config._configs import DecisionConfig
from kompend.llm.decision import ErrorAction, ErrorDecision, decision_model_string


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
    def test_disabled_returns_none_without_building_an_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("no agent may be built when the sidecar is disabled")

        monkeypatch.setattr(decision_module, "DecisionAgent", _fail)
        assert decide_error_action(DecisionConfig(), error="boom", stage="s", attempt=1, max_attempts=3) is None

    def test_action_comes_from_the_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(DecisionAgent, "run", lambda self, deps: ErrorDecision(action=ErrorAction.RETRY))
        config = DecisionConfig(enabled=True, model="jev-latest")
        assert decide_error_action(config, error="timeout", stage="s", attempt=1, max_attempts=3) == "retry"

    def test_agent_failure_falls_back_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(self: DecisionAgent, deps: DecisionDeps) -> ErrorDecision:
            raise RuntimeError("sidecar down")

        monkeypatch.setattr(DecisionAgent, "run", _boom)
        config = DecisionConfig(enabled=True, model="jev-latest")
        assert decide_error_action(config, error="timeout", stage="s", attempt=1, max_attempts=3) is None
