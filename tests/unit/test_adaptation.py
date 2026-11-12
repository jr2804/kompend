"""Loop-1 deterministic shrink: policy, splitting, and the re-queue loop.

Loop 1 answers size failures with arithmetic — halve the unit that
overflowed and re-queue the halves — instead of another tuning knob. Each
split strictly reduces the unit, so the loop terminates by construction,
bounded by ``budget.segment_shrink_attempts``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import kompend.compilation.flows.generation.segments as segments_module
from kompend.agents.failure import AgentFailureKind, EmptyAnswerError
from kompend.compilation.adaptation import (
    AdaptationSignal,
    plan_adaptation,
    split_concepts,
    split_sources,
    split_unit,
)
from kompend.compilation.concept_broker import ConceptBroker
from kompend.compilation.flows.generation import _consult_loop2, _SegmentUnit, generate_all_pages_agentic
from kompend.compilation.schema import GenerationOutput, GenerationPage
from kompend.compilation.two_phase import TwoPhaseCompiler
from kompend.core.config import EmbeddingConfig, LLMConfig
from kompend.core.pipeline.orchestrator import PipelineConfig
from kompend.core.utils.exceptions import LLMContextOverflowError, LLMOutputError
from kompend.llm.decision import ErrorAction, ErrorDecision
from tests.conftest import make_tmp_vault

CONCEPTS = [
    {"name": "Alpha", "kind": "concept"},
    {"name": "Beta", "kind": "concept"},
    {"name": "Gamma", "kind": "concept"},
    {"name": "Delta", "kind": "concept"},
]


class TestWindowPressure:
    @pytest.mark.parametrize(
        ("tokens", "budget", "expected"),
        [
            (None, None, False),
            (None, 128000, False),
            (138319, None, False),
            (0, 128000, False),
            (121599, 128000, False),  # just under the line
            (121601, 128000, True),
            (138319, 128000, True),
        ],
    )
    def test_pressure_needs_a_snapshot_at_the_line(self, tokens: int | None, budget: int | None, expected: bool) -> None:
        assert _signal(AgentFailureKind.OTHER, context_tokens=tokens, context_budget=budget).window_pressure is expected


class TestPlanAdaptation:
    def test_provider_overflow_splits_without_any_snapshot(self) -> None:
        plan = plan_adaptation(_signal(AgentFailureKind.CONTEXT_OVERFLOW), max_adaptations=2)
        assert plan.split

    @pytest.mark.parametrize(
        "kind",
        [AgentFailureKind.SCHEMA_MISMATCH, AgentFailureKind.OTHER, AgentFailureKind.EMPTY_ANSWER],
    )
    def test_ambiguous_failure_splits_only_under_pressure(self, kind: AgentFailureKind) -> None:
        assert not plan_adaptation(_signal(kind), max_adaptations=2).split
        pressured = _signal(kind, context_tokens=138319, context_budget=128000)
        assert plan_adaptation(pressured, max_adaptations=2).split

    def test_transport_never_splits_even_at_the_window(self) -> None:
        pressured = _signal(AgentFailureKind.TRANSIENT, context_tokens=138319, context_budget=128000)
        assert not plan_adaptation(pressured, max_adaptations=2).split

    def test_spent_budget_keeps_todays_loud_failure(self) -> None:
        used_up = _signal(AgentFailureKind.CONTEXT_OVERFLOW, adaptations_used=2)
        plan = plan_adaptation(used_up, max_adaptations=2)
        assert not plan.split

    def test_zero_budget_disables_the_shrink(self) -> None:
        plan = plan_adaptation(_signal(AgentFailureKind.CONTEXT_OVERFLOW), max_adaptations=0)
        assert not plan.split


class TestSplitSources:
    def test_pairs_stay_intact_and_chars_balance(self) -> None:
        sources = [("a", "x" * 100), ("b", "y" * 100), ("c", "z" * 100)]
        first, second = split_sources(sources)
        assert [label for label, _ in first] == ["a", "b"]
        assert [label for label, _ in second] == ["c"]
        assert sum(len(c) for _, c in first) == 200
        assert sum(len(c) for _, c in second) == 100
        assert "".join(c for _, c in first + second) == "x" * 100 + "y" * 100 + "z" * 100

    def test_single_source_splits_chars_with_label_on_both(self) -> None:
        (first, second) = split_sources([("a", "w" * 100)])
        (label_a, content_a), (label_b, content_b) = first[0], second[0]
        assert (label_a, label_b) == ("a", "a")
        assert content_a + content_b == "w" * 100
        assert content_a
        assert content_b

    def test_nothing_to_shrink_gives_up(self) -> None:
        assert split_sources([]) == ([], [])
        assert split_sources([("a", "")]) == ([("a", "")], [])


class TestSplitConcepts:
    def test_halves_are_contiguous(self) -> None:
        concepts = [{"name": f"c{i}"} for i in range(5)]
        first, second = split_concepts(concepts)
        assert [c["name"] for c in first] == ["c0", "c1"]
        assert [c["name"] for c in second] == ["c2", "c3", "c4"]

    def test_empty_stays_empty(self) -> None:
        assert split_concepts([]) == ([], [])


class TestSplitUnit:
    def test_split_covers_every_concept(self) -> None:
        segment = [("a", "x" * 60), ("b", "y" * 60)]
        concepts = [{"name": f"c{i}"} for i in range(4)]
        halves = split_unit(segment, concepts)
        assert halves is not None
        (sources_a, concepts_a), (sources_b, concepts_b) = halves
        assert len(concepts_a) + len(concepts_b) == 4
        assert "".join(c for _, c in sources_a + sources_b) == "x" * 60 + "y" * 60

    def test_lone_concept_grounds_against_both_halves(self) -> None:
        halves = split_unit([("a", "x" * 60)], [{"name": "only"}])
        assert halves is not None
        (_, concepts_a), (_, concepts_b) = halves
        assert concepts_a == concepts_b == [{"name": "only"}]

    def test_nothing_assigned_cannot_shrink(self) -> None:
        assert split_unit([("a", "xxx")], []) is None

    def test_degenerate_text_cannot_shrink(self) -> None:
        assert split_unit([("a", "")], [{"name": "c0"}]) is None


class TestBrokerAdoptAndDrain:
    def test_sub_pool_serves_after_adopt(self) -> None:
        broker = ConceptBroker([{"name": "c0"}, {"name": "c1"}], num_callers=1)
        assert broker.fetch("seg_0", limit=10) == [{"name": "c0"}, {"name": "c1"}]
        broker.adopt_caller("seg_0/sa", [{"name": "c0"}])
        assert broker.fetch("seg_0/sa", limit=10) == [{"name": "c0"}]
        assert broker.fetch("seg_0/sa", limit=10) == []

    def test_adopting_twice_is_a_programming_error(self) -> None:
        broker = ConceptBroker([{"name": "c0"}], num_callers=1)
        broker.adopt_caller("seg_0/sa", [{"name": "c0"}])
        with pytest.raises(ValueError, match="already registered"):
            broker.adopt_caller("seg_0/sa", [{"name": "c0"}])

    def test_drain_hides_moved_work_but_keeps_consumed(self) -> None:
        broker = ConceptBroker([{"name": "c0"}, {"name": "c1"}], num_callers=1)
        broker.fetch("seg_0", limit=1)
        broker.drain_caller("seg_0")
        assert broker.remaining("seg_0") == 0
        assert broker.fetch("seg_0", limit=10) == []
        assert broker.consumed("seg_0") == [{"name": "c0"}, {"name": "c1"}]


class TestShrinkLoop:
    """The seg_4 shape: the parent overflows after fetching, the halves page."""

    def test_overflow_halves_and_merges(self) -> None:
        compiler = self._compiler()
        calls: list[str] = []
        failed = False

        def _factory(**kwargs: object) -> object:
            agent = MagicMock()
            agent.result_type = GenerationOutput

            def _run(deps: object, **run_kwargs: object) -> GenerationOutput:
                nonlocal failed
                broker = getattr(deps, "concept_broker", None)
                source_id = str(getattr(deps, "source_id", ""))
                batch = broker.fetch(source_id, limit=10) if broker is not None else []
                calls.append(source_id)
                if source_id == "seg_0" and not failed:
                    # The seg_4 shape: concepts fetched, then the request died.
                    failed = True
                    raise LLMContextOverflowError("request exceeds the available context size (131072 tokens)")
                return GenerationOutput(
                    pages=[GenerationPage(title=str(c["name"]), content=f"Body for {c['name']}.", provenance=["sources/corpus.pdf"]) for c in batch]
                )

            agent.run = _run
            return agent

        with patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", side_effect=_factory):
            nodes, _, caller_progress = generate_all_pages_agentic(
                compiler,
                [dict(c) for c in CONCEPTS],
                [[("a", "x" * 500)]],
                [],
                pinned_facts=[],
                known_titles=[],
            )
        # Both halves ran and paged their concepts; the parent key merged them.
        assert calls[0] == "seg_0"
        assert sorted(calls[1:]) == ["seg_0/sa", "seg_0/sb"]
        assert sorted(caller_progress.keys()) == ["seg_0"]
        assert sorted(caller_progress["seg_0"]) == ["alpha", "beta", "delta", "gamma"]
        assert len(nodes) == 4

    def test_spent_budget_keeps_the_loud_failure(self) -> None:
        compiler = self._compiler()
        compiler._config.budget.segment_shrink_attempts = 0

        def _factory(**kwargs: object) -> object:
            agent = MagicMock()
            agent.result_type = GenerationOutput

            def _run(deps: object, **run_kwargs: object) -> GenerationOutput:
                raise LLMContextOverflowError("request exceeds the available context size (131072 tokens)")

            agent.run = _run
            return agent

        with (
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", side_effect=_factory),
            pytest.raises(LLMContextOverflowError),
        ):
            generate_all_pages_agentic(
                compiler,
                [dict(c) for c in CONCEPTS],
                [[("a", "x" * 500)]],
                [],
                pinned_facts=[],
                known_titles=[],
            )

    def _compiler(self) -> TwoPhaseCompiler:
        vault = make_tmp_vault()
        config = PipelineConfig(
            base_dir=vault,
            source_dir=vault / "sources",
            output_dir=vault / "wiki",
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm", timeout=120.0),
            embedding=EmbeddingConfig(model="hash"),
        )
        return TwoPhaseCompiler(config=config)


class TestDescribe:
    def test_record_holds_counts_not_content(self) -> None:
        text = _signal(
            AgentFailureKind.OTHER,
            seg_id="seg_3",
            adaptations_used=1,
            concepts_assigned=12,
            concepts_remaining=7,
            fetches=10,
            context_tokens=138319,
            context_budget=128000,
            last_error="validation failed",
        ).describe()
        assert "unit=seg_3" in text
        assert "kind=other" in text
        assert "adaptations=1" in text
        assert "concepts_assigned=12 concepts_remaining=7" in text
        assert "fetches=10" in text
        assert "138319/128000" in text
        assert "validation failed" in text

    def test_record_never_carries_request_content(self) -> None:
        content = "x" * 500
        text = _signal(AgentFailureKind.CONTEXT_OVERFLOW).describe()
        assert content not in text
        assert "no usage snapshot" in text


class TestErrorDecisionShrink:
    def test_shrink_defaults_off_for_old_answers(self) -> None:
        assert ErrorDecision.model_validate({"action": "retry"}).shrink is False

    def test_shrink_survives_the_round_trip(self) -> None:
        decision = ErrorDecision.model_validate({"action": "retry", "shrink": True})
        assert (decision.action, decision.shrink) == (ErrorAction.RETRY, True)


class TestConsultGate:
    @pytest.mark.parametrize("kind", [AgentFailureKind.TRANSIENT, AgentFailureKind.CONTEXT_OVERFLOW])
    def test_arithmetic_failures_never_reach_the_model(self, kind: AgentFailureKind) -> None:
        with patch.object(segments_module, "decide_unit_adaptation") as decide:
            assert _consult_loop2(self._compiler(), self._unit(), _loop2_signal(kind), 2) is None
            decide.assert_not_called()

    def test_spent_budget_never_reaches_the_model(self) -> None:
        with patch.object(segments_module, "decide_unit_adaptation") as decide:
            signal = _loop2_signal(AgentFailureKind.OTHER, adaptations_used=2)
            assert _consult_loop2(self._compiler(), self._unit(depth=2), signal, 2) is None
            decide.assert_not_called()

    def test_no_model_keeps_todays_path(self) -> None:
        with patch.object(segments_module, "decide_unit_adaptation", return_value=None) as decide:
            signal = _loop2_signal(AgentFailureKind.OTHER)
            assert _consult_loop2(self._compiler(), self._unit(), signal, 2) is None
            decide.assert_called_once()

    def test_verdict_passes_through(self) -> None:
        verdict = ErrorDecision(action=ErrorAction.RETRY, shrink=True)
        with patch.object(segments_module, "decide_unit_adaptation", return_value=verdict):
            signal = _loop2_signal(AgentFailureKind.OTHER)
            assert _consult_loop2(self._compiler(), self._unit(), signal, 2) is verdict

    def _compiler(self) -> MagicMock:
        compiler = MagicMock()
        compiler._config.decision = None
        return compiler

    def _unit(self, depth: int = 0) -> _SegmentUnit:
        return _SegmentUnit(label="segment 1/1", seg_id="seg_0", segment=[], run_prefix=None, seg_idx=0, depth=depth)


class TestLoop2Shrink:
    """An ambiguous failure Loop 1 declines can still earn a split."""

    def test_model_shrink_splits_what_loop1_declined(self) -> None:
        def _fail(source_id: str, batch: list[dict[str, Any]]) -> None:
            if source_id == "seg_0":
                raise EmptyAnswerError("Please return text or include your response in a tool call")

        def _decide(signal: AdaptationSignal) -> ErrorDecision:
            return ErrorDecision(action=ErrorAction.RETRY, shrink=True)

        calls, seen_signals, result = self._run_loop(_fail, _decide)
        assert calls[0] == "seg_0"
        assert sorted(calls[1:]) == ["seg_0/sa", "seg_0/sb"]
        assert any("kind=empty_answer" in s and "unit=seg_0" in s for s in seen_signals)
        assert all(("x" * 100) not in s for s in seen_signals)
        nodes, _, caller_progress = result
        assert sorted(caller_progress.keys()) == ["seg_0"]
        assert len(nodes) == 4

    def test_a_model_skip_spends_the_budget_and_is_recorded_as_a_hole(self, caplog: pytest.LogCaptureFixture) -> None:
        """A skip belongs to the attempt: an empty answer keeps asking.

        The ADR "A skip verdict ends the attempt, not the segment" reversed
        the old rule that one skip ended the segment. An empty answer is a
        continuation kind, so every pass carries a fresh prompt and the
        verdict must not cut the segment short of its budget; the tail then
        records the hole instead of calling it a hard failure.
        """

        def _fail(source_id: str, batch: list[dict[str, Any]]) -> None:
            raise EmptyAnswerError("Please return text or include your response in a tool call")

        def _decide(signal: AdaptationSignal) -> ErrorDecision:
            return ErrorDecision(action=ErrorAction.SKIP)

        with caplog.at_level(logging.WARNING):
            calls, seen_signals, result = self._run_loop(_fail, _decide)
        assert calls == ["seg_0"] * 5
        assert "the decision model skipped it" in caplog.text
        assert "verdict=skip" in caplog.text
        # The tail decides the run outcome: a skipped segment is a recorded
        # hole, not a failure - this run only has the one segment, so the
        # result carries no page while the call itself returns normally.
        nodes, _, caller_progress = result
        assert nodes == []
        assert caller_progress == {"seg_0": []}

    def test_no_model_keeps_todays_retry_path(self) -> None:
        def _fail(source_id: str, batch: list[dict[str, Any]]) -> None:
            raise EmptyAnswerError("Please return text or include your response in a tool call")

        calls, seen_signals, exc = self._run_loop(_fail, lambda signal: None)
        # EMPTY_ANSWER is a continuation kind: 4 concepts + 1 = 5 passes, no split.
        assert calls == ["seg_0"] * 5
        assert not [c for c in calls if "/s" in c]
        assert isinstance(exc, LLMOutputError)

    def _compiler(self) -> TwoPhaseCompiler:
        vault = make_tmp_vault()
        config = PipelineConfig(
            base_dir=vault,
            source_dir=vault / "sources",
            output_dir=vault / "wiki",
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm", timeout=120.0),
            embedding=EmbeddingConfig(model="hash"),
        )
        return TwoPhaseCompiler(config=config)

    def _run_loop(
        self,
        fail: Callable[[str, list[dict[str, Any]]], None],
        decide: Callable[[AdaptationSignal], ErrorDecision | None],
    ) -> Any:
        compiler = self._compiler()
        calls: list[str] = []
        seen_signals: list[str] = []

        def _factory(**kwargs: object) -> object:
            agent = MagicMock()
            agent.result_type = GenerationOutput

            def _run(deps: object, **run_kwargs: object) -> GenerationOutput:
                broker = getattr(deps, "concept_broker", None)
                source_id = str(getattr(deps, "source_id", ""))
                batch = broker.fetch(source_id, limit=10) if broker is not None else []
                calls.append(source_id)
                fail(source_id, batch)
                return GenerationOutput(
                    pages=[GenerationPage(title=str(c["name"]), content=f"Body for {c['name']}.", provenance=["sources/corpus.pdf"]) for c in batch]
                )

            agent.run = _run
            return agent

        def _decide(config: object, *, signal: AdaptationSignal, stage: str, max_adaptations: int) -> Any:
            seen_signals.append(signal.describe())
            return decide(signal)

        with (
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", side_effect=_factory),
            patch.object(segments_module, "decide_unit_adaptation", side_effect=_decide),
        ):
            try:
                result = generate_all_pages_agentic(
                    compiler,
                    [dict(c) for c in CONCEPTS],
                    [[("a", "x" * 500)]],
                    [],
                    pinned_facts=[],
                    known_titles=[],
                )
            except LLMOutputError as exc:
                return calls, seen_signals, exc
            return calls, seen_signals, result
        raise AssertionError("unreachable")


def _loop2_signal(kind: AgentFailureKind, **overrides: Any) -> AdaptationSignal:
    return _signal(kind, **overrides)


def _signal(kind: AgentFailureKind, **overrides: Any) -> AdaptationSignal:
    base: dict[str, Any] = {
        "seg_id": "seg_0",
        "kind": kind,
        "adaptations_used": 0,
        "concepts_assigned": 12,
        "concepts_remaining": 12,
        "fetches": 10,
        "context_tokens": None,
        "context_budget": None,
        "last_error": "boom",
    }
    base.update(overrides)
    return AdaptationSignal(**base)  # type: ignore[arg-type]
