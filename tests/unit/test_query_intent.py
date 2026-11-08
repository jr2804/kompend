"""Query-intent routing: the regex rules first, the decision sidecar second.

knox-ukfi candidate 10. The classifier must stay deterministic: with no
decision model configured (or a failing one) the regex rules alone answer,
and only a query those rules leave as knowledge reaches the model.
"""

from __future__ import annotations

import pytest

from kompend.agents import decision as decision_module
from kompend.agents.decision import QueryIntentAgent, QueryIntentDeps, decide_query_intent
from kompend.agents.specs import get_spec
from kompend.core.config._configs import DecisionConfig
from kompend.llm.decision import QueryIntentDecision, QueryIntentVerdict
from kompend.search.classifier import QueryIntent, classify

_KNOWLEDGE_SHAPED = "how does flash attention reduce memory traffic"


class TestRegexRulesStillDecideAlone:
    def test_greeting_needs_no_model(self) -> None:
        result = classify("hello there")

        assert result.intent is QueryIntent.GREETING
        assert result.response is not None

    def test_thanks_needs_no_model(self) -> None:
        assert classify("thanks, that helped").intent is QueryIntent.GRATITUDE

    def test_knowledge_query_is_not_answered_without_a_model(self) -> None:
        result = classify(_KNOWLEDGE_SHAPED)

        assert result.intent is QueryIntent.KNOWLEDGE
        assert result.response is None

    def test_disabled_config_never_reaches_the_sidecar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("the sidecar must not run when decision.enabled is false")

        monkeypatch.setattr(decision_module, "decide_query_intent", _fail)

        assert classify(_KNOWLEDGE_SHAPED, decision=DecisionConfig()).intent is QueryIntent.KNOWLEDGE


class TestDecisionSidecarDecidesTheAmbiguousRest:
    def test_small_talk_the_rules_miss_becomes_offtopic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            decision_module,
            "decide_query_intent",
            lambda config, *, query: QueryIntentVerdict.OFFTOPIC,  # noqa: ARG005 — the sidecar contract
        )

        result = classify(
            "could you tell me a story about dragons",
            decision=DecisionConfig(enabled=True, model="ollama/clef"),
        )

        assert result.intent is QueryIntent.OFFTOPIC
        assert result.response is not None

    def test_a_knowledge_verdict_still_goes_to_retrieval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            decision_module,
            "decide_query_intent",
            lambda config, *, query: QueryIntentVerdict.KNOWLEDGE,  # noqa: ARG005 — the sidecar contract
        )

        result = classify(_KNOWLEDGE_SHAPED, decision=DecisionConfig(enabled=True, model="ollama/clef"))

        assert result.intent is QueryIntent.KNOWLEDGE
        assert result.response is None

    def test_obvious_small_talk_costs_no_model_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("a regex match must not spend a decision call")

        monkeypatch.setattr(decision_module, "decide_query_intent", _fail)

        assert classify("hello", decision=DecisionConfig(enabled=True, model="ollama/clef")).intent is QueryIntent.GREETING


class TestDecideQueryIntent:
    def test_disabled_returns_none_without_building_an_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("no agent may be built when the sidecar is disabled")

        monkeypatch.setattr(decision_module, "QueryIntentAgent", _fail)

        assert decide_query_intent(DecisionConfig(), query="hello") is None
        assert decide_query_intent(None, query="hello") is None

    def test_verdict_comes_from_the_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            QueryIntentAgent,
            "run",
            lambda self, deps: QueryIntentDecision(intent=QueryIntentVerdict.HELP),  # noqa: ARG005 — the agent contract
        )

        verdict = decide_query_intent(DecisionConfig(enabled=True, model="ollama/clef"), query="what can you do?")

        assert verdict is QueryIntentVerdict.HELP

    def test_agent_failure_falls_back_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(self: QueryIntentAgent, deps: QueryIntentDeps) -> QueryIntentDecision:  # noqa: ARG001 — the agent contract
            raise RuntimeError("sidecar down")

        monkeypatch.setattr(QueryIntentAgent, "run", _boom)

        assert decide_query_intent(DecisionConfig(enabled=True, model="ollama/clef"), query="hello") is None


def test_the_spec_is_registered_with_its_own_result_type() -> None:
    spec = get_spec("query_intent")

    assert spec.name == "query_intent"
    assert spec.result_type is QueryIntentDecision
    assert spec.deps_type is QueryIntentDeps
    assert spec.template_prefix == "decision/"
