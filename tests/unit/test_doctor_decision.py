"""Tests for the doctor's decision-model checks.

Two questions, two failure families. ``decision-schema`` is offline: it asks the
decision grammar whether it can express every shipped result type, because a
refused type degrades a judge tier to its fallback with no error reaching the
caller. ``connectivity-decision`` is one live SystemOne request, because a dead
endpoint degrades the same tiers silently in the other direction.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from kompend.agents.decision_probe import DecisionProbeResult
from kompend.core.config import DecisionConfig
from kompend.core.utils.exceptions import LLMOutputError
from kompend.diagnostics import decision as diagnostics_decision
from kompend.diagnostics.base import CheckResult
from kompend.diagnostics.decision import (
    DECISION_CONNECTIVITY_CHECK,
    DECISION_SCHEMA_CHECK,
    check_decision_connectivity,
    check_decision_schema,
)
from kompend.diagnostics.runner import run_doctor
from kompend.llm.decision import DECISION_RESULT_TYPES, CandidateVerdict


class _BatchedVerdicts(BaseModel):
    """The shape the grammar refuses: one verdict model per candidate."""

    verdicts: list[CandidateVerdict] = Field(description="One verdict per candidate.")


class TestDecisionSchemaCheck:
    def test_it_accepts_every_shipped_result_type(self) -> None:
        result = check_decision_schema()

        assert result.name == DECISION_SCHEMA_CHECK
        assert result.passed
        assert not result.warning
        assert f"{len(DECISION_RESULT_TYPES)} decision result types accepted" in result.message
        for label, _ in DECISION_RESULT_TYPES:
            assert label in result.message

    def test_the_registry_names_are_the_documented_ones(self) -> None:
        names = [label for label, _ in DECISION_RESULT_TYPES]

        assert names == ["error-triage", "query-intent", "figure-relevance", "source-search-verdict"]

    def test_a_refused_result_type_fails_with_the_grammars_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            diagnostics_decision,
            "DECISION_RESULT_TYPES",
            (*DECISION_RESULT_TYPES, ("batched-verdicts", _BatchedVerdicts)),
        )

        result = check_decision_schema()

        assert result.name == DECISION_SCHEMA_CHECK
        assert result.passed is False
        assert "batched-verdicts" in result.message
        assert "list must be of two or more string options" in result.message

    def test_moved_internals_warn_instead_of_passing_silently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(diagnostics_decision, "_GRAMMAR_MODULE", "pydantic_ai.models._no_such_module")

        result = check_decision_schema()

        assert result.passed is True
        assert result.warning is True
        assert "pydantic-ai internals changed" in result.message


class TestDecisionConnectivityCheck:
    def test_disabled_is_a_warning_not_a_failure(self) -> None:
        result = check_decision_connectivity(DecisionConfig())

        assert result.name == DECISION_CONNECTIVITY_CHECK
        assert result.passed is True
        assert result.warning is True
        assert "disabled" in result.message

    def test_no_config_at_all_is_the_same_warning(self) -> None:
        result = check_decision_connectivity(None)

        assert result.passed is True
        assert result.warning is True
        assert "disabled" in result.message

    def test_enabled_without_a_model_is_a_warning(self) -> None:
        result = check_decision_connectivity(DecisionConfig(enabled=True, model=None))

        assert result.passed is True
        assert result.warning is True
        assert "no model configured" in result.message

    def test_a_live_answer_passes_with_the_endpoint_family(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, float]] = []

        def _probe(model_string: str, *, timeout: float) -> DecisionProbeResult:
            seen.append((model_string, timeout))
            return DecisionProbeResult(replied=True)

        monkeypatch.setattr(diagnostics_decision, "probe_decision_model", _probe)

        result = check_decision_connectivity(DecisionConfig(enabled=True, model="ollama-cloud/clef", timeout=3.5))

        assert result.passed is True
        assert not result.warning
        assert "SystemOne" in result.message
        assert "ollama-cloud/clef" in result.message
        assert seen == [("ollama-cloud/clef", 3.5)]

    def test_a_failed_ping_reports_the_real_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cause = "status_code: 404, model_name: clef, body: {'error': 'path \"/v1/v1/systemone\" not found'}"

        def _probe(model_string: str, *, timeout: float) -> Any:
            raise LLMOutputError(cause)

        monkeypatch.setattr(diagnostics_decision, "probe_decision_model", _probe)

        result = check_decision_connectivity(DecisionConfig(enabled=True, model="badcloud/clef"))

        assert result.name == DECISION_CONNECTIVITY_CHECK
        assert result.passed is False
        assert "badcloud/clef" in result.message
        assert "/v1/v1/systemone" in result.message

    def test_a_missing_api_key_is_the_reported_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cause = "No API key was provided. Pass api_key or set the TYPESAFE_API_KEY environment variable."

        def _probe(model_string: str, *, timeout: float) -> Any:
            raise ValueError(cause)

        monkeypatch.setattr(diagnostics_decision, "probe_decision_model", _probe)

        result = check_decision_connectivity(DecisionConfig(enabled=True, model="nokey/clef"))

        assert result.passed is False
        assert "No API key was provided" in result.message

    def test_an_answer_of_no_warns_rather_than_passing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(diagnostics_decision, "probe_decision_model", lambda model_string, *, timeout: DecisionProbeResult(replied=False))

        result = check_decision_connectivity(DecisionConfig(enabled=True, model="ollama-cloud/clef"))

        assert result.passed is True
        assert result.warning is True
        assert "answered no" in result.message


class TestDecisionChecksInTheRun:
    def test_both_checks_are_reported_without_a_config(self) -> None:
        results = run_doctor()
        names = [result.name for result in results]

        assert DECISION_SCHEMA_CHECK in names
        assert DECISION_CONNECTIVITY_CHECK in names
        assert all(isinstance(result, CheckResult) for result in results)

    def test_a_config_without_a_decision_block_skips_the_ping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _no_network(model_string: str, *, timeout: float) -> Any:
            raise AssertionError("the doctor must not ping a disabled decision model")

        monkeypatch.setattr(diagnostics_decision, "probe_decision_model", _no_network)

        result = next(result for result in run_doctor() if result.name == DECISION_CONNECTIVITY_CHECK)

        assert result.warning is True
        assert "disabled" in result.message

    def test_the_decision_checks_run_before_the_completion_models(self) -> None:
        names = [result.name for result in run_doctor()]

        assert names.index(DECISION_SCHEMA_CHECK) < names.index(DECISION_CONNECTIVITY_CHECK)
        assert names.index(DECISION_CONNECTIVITY_CHECK) < names.index("connectivity-llm")
        assert names.index("output-directory") < names.index(DECISION_SCHEMA_CHECK)
