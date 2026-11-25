"""Failure-path behaviour of the agentic generation pass loop.

One pass = one agent run. An empty answer ends that pass, not the
segment: the next pass asks again with the material it has, so a model
that returned nothing once does not cost the whole segment. Every other
failure keeps the old behaviour - the segment stops and the compile
fails loudly, so nothing downstream reports a partial wiki as a complete
one.

A decision-model skip is the one deliberate exception: it spends a pass
and the segment asks again, and a segment the model skipped outright
costs its page, not the compile - the run says so in the log and in the
verdict it prints.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, SqliteStepStore

from kompend.agents.failure import EmptyAnswerError
from kompend.compilation import GraphCompiler
from kompend.compilation.flows.extraction.prompt import _estimate_history_tokens
from kompend.compilation.flows.generation import flow as gen_flow
from kompend.compilation.flows.generation import prepare_concepts
from kompend.compilation.flows.generation.budget import _token_estimate
from kompend.compilation.flows.generation.flow import generate_all_pages_agentic
from kompend.compilation.flows.generation.segments import _failure_action
from kompend.compilation.schema import (
    Concept,
    Entity,
    GenerationOutput,
    GenerationPage,
)
from kompend.compilation.step_recovery import generation_conversation, pass_run_id, segment_run_prefix
from kompend.compilation.two_phase import Extraction, ExtractionSet, TwoPhaseCompiler
from kompend.core.config import BudgetConfig, EmbeddingConfig, LLMConfig
from kompend.core.models import Chunk, ElementType, ResolvedChunk, WikiNode
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.context_window import ContextSource, ContextWindow
from kompend.llm.decision import DecisionState, ErrorAction, ErrorActionOutcome, ErrorDecision, UnitAdaptationOutcome
from kompend.llm.request_budget import RequestBudget
from tests.conftest import make_tmp_vault


class TestAgenticPassLoop:
    """The pass loop decides whether a failed pass ends the segment."""

    def test_empty_answer_does_not_end_the_segment(self) -> None:
        """An empty answer is a failed pass, not a failed segment."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = [
            EmptyAnswerError("the model returned nothing"),
            _page_output(),
        ]

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
        ):
            wiki = compiler.compile(_chunks())

        assert agent.run.call_count == 2
        assert any(node.title == "Listening Effort" for node in wiki.nodes)

    def test_empty_answer_is_recorded_with_its_kind(self, caplog: pytest.LogCaptureFixture) -> None:
        """The failure log names the unit and the classified kind."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = [
            EmptyAnswerError("the model returned nothing"),
            _page_output(),
        ]

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            caplog.at_level(logging.WARNING),
        ):
            compiler.compile(_chunks())

        assert "agentic-generation" in caplog.text
        assert "empty_answer" in caplog.text

    def test_other_failure_stops_the_segment(self) -> None:
        """An unexplained failure keeps the old behaviour: one pass, then stop."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("boom")

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            pytest.raises(LLMOutputError, match=r"produced no page\(s\)"),
        ):
            compiler.compile(_chunks())

        assert agent.run.call_count == 1

    def test_empty_answers_on_every_pass_still_fail_loudly(self) -> None:
        """The segment runs out of passes - then the compile fails, as before."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = EmptyAnswerError("the model returned nothing")

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            pytest.raises(LLMOutputError, match=r"produced no page\(s\)"),
        ):
            compiler.compile(_chunks())

        assert agent.run.call_count == 2


class TestModelSkipVerdict:
    """A decision-model skip ends the attempt, not the segment.

    The decision model can say "stop spending passes here" for a failure
    the deterministic path found ambiguous. That verdict belongs to the
    attempt: the next pass carries new material, so the segment asks
    again within its budget. Only the tail decides the run outcome - a
    segment the model skipped is recorded, not fatal.
    """

    def test_a_model_skip_consumes_a_pass_and_the_segment_asks_again(self) -> None:
        """The skip spends one pass; the segment still has budget left."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = [
            EmptyAnswerError("the model returned nothing"),
            _page_output(),
        ]

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            patch(
                "kompend.compilation.flows.generation.segments.decide_error_action",
                return_value=ErrorActionOutcome(state=DecisionState.ANSWERED, action=ErrorAction.RETRY),
            ),
            patch(
                "kompend.compilation.flows.generation.segments.decide_unit_adaptation",
                return_value=UnitAdaptationOutcome(
                    state=DecisionState.ANSWERED,
                    decision=ErrorDecision(action=ErrorAction.SKIP, shrink=False),
                ),
            ),
        ):
            wiki = compiler.compile(_chunks())

        assert agent.run.call_count == 2
        assert any(node.title == "Listening Effort" for node in wiki.nodes)

    def test_a_skipped_segment_does_not_fail_the_run(self, caplog: pytest.LogCaptureFixture) -> None:
        """A segment the model skipped costs its page, not the compile."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = [
            EmptyAnswerError("the model returned nothing"),
            EmptyAnswerError("the model returned nothing"),
            _page_output(title="Rainfall Accumulation"),
        ]

        with (
            patch.object(compiler, "extract", return_value=_two_concept_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            patch(
                "kompend.compilation.flows.generation.flow.prepare_source_segments",
                return_value=[[("first.pdf", "first")], [("second.pdf", "second")]],
            ),
            patch(
                "kompend.compilation.flows.generation.segments.decide_error_action",
                return_value=ErrorActionOutcome(state=DecisionState.ANSWERED, action=ErrorAction.RETRY),
            ),
            patch(
                "kompend.compilation.flows.generation.segments.decide_unit_adaptation",
                return_value=UnitAdaptationOutcome(
                    state=DecisionState.ANSWERED,
                    decision=ErrorDecision(action=ErrorAction.SKIP, shrink=False),
                ),
            ),
            caplog.at_level(logging.WARNING),
        ):
            wiki = compiler.compile(_chunks())

        assert [node.title for node in wiki.nodes] == ["Rainfall Accumulation"]
        assert "the decision model skipped it" in caplog.text
        assert "verdict=skip" in caplog.text
        assert "passes=2, cap=2" in caplog.text

    def test_a_run_where_every_segment_is_skipped_still_fails(self) -> None:
        """A wiki with no page at all is a failure, not a quiet success."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = EmptyAnswerError("the model returned nothing")

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            patch(
                "kompend.compilation.flows.generation.segments.decide_error_action",
                return_value=ErrorActionOutcome(state=DecisionState.ANSWERED, action=ErrorAction.RETRY),
            ),
            patch(
                "kompend.compilation.flows.generation.segments.decide_unit_adaptation",
                return_value=UnitAdaptationOutcome(
                    state=DecisionState.ANSWERED,
                    decision=ErrorDecision(action=ErrorAction.SKIP, shrink=False),
                ),
            ),
            pytest.raises(LLMOutputError, match=r"Generation failed: 0 of"),
        ):
            compiler.compile(_chunks())

    def test_the_failure_message_names_the_verdict_and_the_pass_count(self) -> None:
        """A reader greps the message to learn why a segment gave up."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("boom")

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            pytest.raises(LLMOutputError, match=r"verdict=gave-up"),
        ):
            compiler.compile(_chunks())

    def test_the_deterministic_fallback_still_fails_loudly(self) -> None:
        """Without a decision model, an empty answer still ends the run."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = EmptyAnswerError("the model returned nothing")

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            pytest.raises(LLMOutputError, match=r"produced no page\(s\)"),
        ):
            compiler.compile(_chunks())

        assert agent.run.call_count == 2


class TestSourceIndexLifecycle:
    """The run's search index owns judge threads, so the flow releases it."""

    def test_the_flow_releases_the_index_after_a_successful_run(self) -> None:
        compiler = _compiler()
        index, closed = _fake_index()

        with patch("kompend.compilation.flows.generation.flow.SourceIndex") as factory:
            factory.from_segments.return_value = index
            with (
                patch.object(compiler, "extract", return_value=_extraction_set()),
                patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=_answering_agent()),
            ):
                compiler.compile(_chunks())

        assert closed == [index], "the index was never released"

    def test_the_flow_releases_the_index_when_generation_fails(self) -> None:
        """An exception must not leave a compile's judge threads behind."""
        compiler = _compiler()
        index, closed = _fake_index()
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("boom")

        with patch("kompend.compilation.flows.generation.flow.SourceIndex") as factory:
            factory.from_segments.return_value = index
            with (
                patch.object(compiler, "extract", return_value=_extraction_set()),
                patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
                pytest.raises(LLMOutputError, match=r"produced no page\(s\)"),
            ):
                compiler.compile(_chunks())

        assert closed == [index]


class TestEntityPageSourceGate:
    """A single mention does not earn an entity its own page."""

    @pytest.mark.parametrize(
        "entity_type",
        ["Function", "Struct", "Path", "Organization", "Standard", "Product", "Term"],
    )
    def test_single_source_entity_is_folded(self, entity_type: str) -> None:
        """One source stays in the source summary; no standalone page."""
        concepts = prepare_concepts(_compiler(), _entity_extraction_set(entity_type, sources=1))
        assert concepts == []

    @pytest.mark.parametrize(
        "entity_type",
        ["Function", "Struct", "Path", "Organization", "Standard", "Product", "Term"],
    )
    def test_multi_source_entity_gets_a_page(self, entity_type: str) -> None:
        """Two sources clear the default gate and keep the entity page."""
        concepts = prepare_concepts(_compiler(), _entity_extraction_set(entity_type, sources=2))
        assert [(c["name"], c["kind"]) for c in concepts] == [(f"FUN_{entity_type.upper()}", "entity")]

    def test_threshold_boundary_uses_the_configured_minimum(self) -> None:
        """The gate is inclusive: exactly ``entity_min_sources`` is enough."""
        compiler = _compiler(budget=BudgetConfig(entity_min_sources=3))
        assert prepare_concepts(compiler, _entity_extraction_set("Function", sources=2)) == []

        concepts = prepare_concepts(compiler, _entity_extraction_set("Function", sources=3))
        assert [c["name"] for c in concepts] == ["FUN_FUNCTION"]

    def test_per_type_override_beats_the_default(self) -> None:
        """Organization may be promoted without promoting every entity type."""
        compiler = _compiler(budget=BudgetConfig(entity_min_sources_by_type={"Organization": 1}))

        organization = prepare_concepts(compiler, _entity_extraction_set("Organization", sources=1))
        function = prepare_concepts(compiler, _entity_extraction_set("Function", sources=1))

        assert [c["name"] for c in organization] == ["FUN_ORGANIZATION"]
        assert function == []


class TestResumedHistoryPricing:
    """The segment fit guard prices a resumed transcript (knox-egxf).

    A recovered segment re-threads its old transcript on every pass, but the
    guard sized only the new content — the thread grew past the window while
    each segment "fit". The priced request is content plus the resumed thread
    in the same density, so an over-budget segment splits before the first
    pass instead of stalling mid-stream.
    """

    def test_no_resumed_thread_prices_nothing(self) -> None:
        total, history_tokens = gen_flow._priced_request_tokens("x" * 1000, None, chars_per_token=2.5)
        assert history_tokens == 0
        assert total == _token_estimate("x" * 1000, 2.5)

    def test_a_resumed_transcript_is_priced_with_the_sizing_density(self) -> None:
        history = [
            ModelRequest(parts=[UserPromptPart(content="q" * 20_000)]),
            ModelResponse(parts=[TextPart(content="a" * 20_000)]),
        ]
        total, history_tokens = gen_flow._priced_request_tokens("x" * 1000, history, chars_per_token=2.5)
        assert history_tokens == _estimate_history_tokens(history, chars_per_token=2.5) > 0
        assert total == _token_estimate("x" * 1000, 2.5) + history_tokens

    def test_over_thread_segment_splits_before_the_first_pass(self, tmp_path: Path) -> None:
        """An over-thread segment splits before the first pass (knox-egxf).

        A resumed transcript that no longer fits beside the segment content
        splits the segment instead of aborting the run: the halves page
        under fresh transcript prefixes and the compile finishes.
        """
        nodes, agent = _run_over_thread_segment(
            tmp_path,
            [[("a.pdf", "Source body text about attention. " * 20), ("b.pdf", "More source text. " * 20)]],
        )
        assert [node.title for node in nodes] == ["Listening Effort"]
        assert agent.run.call_count >= 2

    def test_over_thread_halves_start_without_the_oversized_transcript(self, tmp_path: Path) -> None:
        """The split halves run fresh: the oversized thread dies with the parent.

        Every pass after the pre-pass split carries no resumed history, so
        the guard never trips twice on the same thread.
        """
        _nodes, agent = _run_over_thread_segment(
            tmp_path,
            [[("a.pdf", "Source body text about attention. " * 20), ("b.pdf", "More source text. " * 20)]],
        )
        assert agent.run.call_count >= 2
        for call in agent.run.call_args_list:
            assert call.kwargs["message_history"] is None

    def test_an_unsplittable_over_thread_segment_keeps_the_loud_failure(self, tmp_path: Path) -> None:
        """A segment that cannot shrink keeps the guard's loud failure.

        A single-character source cannot halve, so the pre-pass split is
        declined and the run aborts instead of silently dropping the thread.
        """
        compiler, store, budget = _over_thread_setup(tmp_path)
        agent = _answering_agent()
        with (
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            pytest.raises(LLMOutputError, match="Agentic generation for"),
        ):
            generate_all_pages_agentic(
                compiler,
                [{"name": "attention", "kind": "concept", "importance": "high", "definition": "def"}],
                [[("a.pdf", "x")]],
                [],
                pinned_facts=[],
                known_titles=[],
                budget=budget,
                step_store=store,
            )
        assert agent.run.call_count == 0


class TestFailureActionDecisionState:
    """The verdict carries whether the decision model spoke.

    The flow tail needs the difference between "the model said retry" and
    "nobody answered, so the fallback retries": only the second degrades
    the run.
    """

    def test_no_sidecar_marks_the_verdict_not_configured(self) -> None:
        verdict = _failure_action(self._compiler(), ValueError("bad"), stage="s", attempt=1, max_attempts=3, transient=False)
        assert verdict.action is ErrorAction.SKIP
        assert verdict.from_model is False
        assert verdict.decision_state is DecisionState.NOT_CONFIGURED

    def test_unavailable_sidecar_rides_on_the_verdict(self) -> None:
        outcome = ErrorActionOutcome(state=DecisionState.UNAVAILABLE)
        with patch("kompend.compilation.flows.generation.segments.decide_error_action", return_value=outcome):
            verdict = _failure_action(self._compiler(), ValueError("bad"), stage="s", attempt=1, max_attempts=3, transient=True)
        assert verdict.action is ErrorAction.RETRY
        assert verdict.from_model is False
        assert verdict.decision_state is DecisionState.UNAVAILABLE

    def test_answered_verdict_marks_the_model(self) -> None:
        outcome = ErrorActionOutcome(state=DecisionState.ANSWERED, action=ErrorAction.RETRY)
        with patch("kompend.compilation.flows.generation.segments.decide_error_action", return_value=outcome):
            verdict = _failure_action(self._compiler(), ValueError("bad"), stage="s", attempt=1, max_attempts=3, transient=True)
        assert verdict.action is ErrorAction.RETRY
        assert verdict.from_model is True
        assert verdict.decision_state is DecisionState.ANSWERED

    def _compiler(self) -> MagicMock:
        compiler = MagicMock()
        compiler._config.decision = None
        compiler._config.llm.fallback_chain = []
        compiler._model = "test/model"
        return compiler


class TestUnjudgedDecisionTier:
    """A dead decision tier degrades the run instead of failing it.

    When every pass ends without a judgment (the tier never answered) and
    the budget is spent, the segment becomes a recorded hole: the run does
    not raise, and the segment is named. A request that dies on its own
    (no unjudged verdict anywhere) still fails loudly.
    """

    def test_unjudged_budget_becomes_a_hole_not_an_error(self, caplog: pytest.LogCaptureFixture) -> None:
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = TimeoutError("boom")
        with (
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            patch(
                "kompend.compilation.flows.generation.segments.decide_error_action",
                return_value=ErrorActionOutcome(state=DecisionState.UNAVAILABLE),
            ),
            caplog.at_level(logging.INFO),
        ):
            nodes, _slugs, _progress, report = generate_all_pages_agentic(
                compiler,
                [{"name": "Listening Effort", "kind": "concept", "definition": "d"}],
                [[("a", "x" * 500)]],
                [],
                pinned_facts=[],
                known_titles=[],
            )
        assert nodes == []
        assert report.engaged is True
        assert report.unjudged == ("segment 1/1",)
        assert "without a decision-model judgment" in caplog.text
        assert "segment 1/1" in caplog.text

    def test_answered_give_up_still_fails_loudly(self) -> None:
        """The model spoke and the budget is spent: no hole, an error."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = TimeoutError("boom")
        outcome = ErrorActionOutcome(state=DecisionState.ANSWERED, action=ErrorAction.SKIP)
        with (
            patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent),
            patch("kompend.compilation.flows.generation.segments.decide_error_action", return_value=outcome),
            pytest.raises(LLMOutputError, match=r"produced no page\(s\)"),
        ):
            generate_all_pages_agentic(
                compiler,
                [{"name": "Listening Effort", "kind": "concept", "definition": "d"}],
                [[("a", "x" * 500)]],
                [],
                pinned_facts=[],
                known_titles=[],
            )


def _run_over_thread_segment(
    tmp_path: Path,
    segments: list[list[tuple[str, str]]],
) -> tuple[list[WikiNode], MagicMock]:
    """Run splittable segments against the over-thread setup; return nodes + agent."""
    compiler, store, budget = _over_thread_setup(tmp_path)
    agent = _answering_agent()
    with patch("kompend.compilation.flows.generation.flow.make_agentic_generation_agent", return_value=agent):
        nodes, _slug_to_idx, _progress, _decision_report = generate_all_pages_agentic(
            compiler,
            [{"name": "attention", "kind": "concept", "importance": "high", "definition": "def"}],
            segments,
            [],
            pinned_facts=[],
            known_titles=[],
            budget=budget,
            step_store=store,
        )
    return nodes, agent


def _over_thread_setup(tmp_path: Path) -> tuple[TwoPhaseCompiler, SqliteStepStore, RequestBudget]:
    """A compiler, a seeded store, and a tight budget for the over-thread split.

    The store holds an interrupted segment-0 transcript whose thread dwarfs
    the window; the budget fits the segment content but not content plus
    the thread. Returns the compiler (resume enabled), the store, and the
    budget, so each test drives its own segment list through them.
    """
    compiler = _compiler()
    compiler._config.no_cache = False
    compiler._active_gen_deps = {"MODEL": "test/model", "EXTRACTIONS": "seed"}
    conversation = generation_conversation(compiler._active_gen_deps)
    store = SqliteStepStore(database=tmp_path / "steps.db")
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="q" * 60_000)]),
        ModelResponse(parts=[TextPart(content="a" * 60_000)]),
    ]
    run_id = pass_run_id(segment_run_prefix(conversation, 0), 1, 1)

    async def _seed() -> None:
        await store.register_run(RunRecord(run_id=run_id, conversation_id=conversation))
        await store.save_snapshot(ContinuableSnapshot(run_id=run_id, step_index=1, messages=history, state="interrupted"))

    asyncio.run(_seed())
    window = ContextWindow(max_input=8192, max_output=4096, source=ContextSource.FALLBACK)
    budget = RequestBudget(window=window, reserved={"system": 1000}, utilization=LLMConfig().request_utilization)
    return compiler, store, budget


def _extraction_set() -> ExtractionSet:
    ext_set = ExtractionSet()
    ext_set.add(
        Extraction(
            source="src.pdf",
            concepts=[Concept(name="Listening Effort", definition="The effort to understand speech.")],
        )
    )
    return ext_set


def _chunks(count: int = 3) -> list[ResolvedChunk]:
    return [
        ResolvedChunk(
            chunk=Chunk(
                id=f"chunk_{i:03d}",
                text=f"Content for chunk {i}.",
                element_types=[ElementType.TEXT],
                source_elements=[0],
                heading_context="Section 1",
            ),
        )
        for i in range(count)
    ]


def _compiler(*, budget: BudgetConfig | None = None) -> TwoPhaseCompiler:
    config = PipelineConfig(
        source_dir=make_tmp_vault(),
        output_dir=make_tmp_vault(),
        llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"),
        embedding=EmbeddingConfig(model="hash"),
        budget=budget or BudgetConfig(),
    )
    config.no_cache = True
    return TwoPhaseCompiler(config=config, base=GraphCompiler("model"))


def _fake_index() -> tuple[MagicMock, list[MagicMock]]:
    """A stand-in source index that records whether its run released it."""
    index = MagicMock()
    closed: list[MagicMock] = []
    index.close.side_effect = lambda: closed.append(index)
    return index, closed


def _answering_agent() -> MagicMock:
    agent = MagicMock()
    agent.run.return_value = _page_output()
    return agent


def _page_output(title: str = "Listening Effort") -> GenerationOutput:
    return GenerationOutput(
        pages=[GenerationPage(title=title, content="The effort to understand speech.", provenance=["sources/src.pdf"])],
    )


def _two_concept_extraction_set() -> ExtractionSet:
    """Two concepts in two sources, so each segment owns one of them."""
    ext_set = ExtractionSet()
    ext_set.add(
        Extraction(
            source="first.pdf",
            concepts=[Concept(name="Listening Effort", definition="The effort to understand speech.")],
        )
    )
    ext_set.add(
        Extraction(
            source="second.pdf",
            concepts=[Concept(name="Rainfall Accumulation", definition="The depth of rain over time.")],
        )
    )
    return ext_set


def _entity_extraction_set(entity_type: str, *, sources: int) -> ExtractionSet:
    ext_set = ExtractionSet()
    for index in range(sources):
        ext_set.add(
            Extraction(
                source=f"src{index}.pdf",
                entities=[Entity(name=f"FUN_{entity_type.upper()}", type=entity_type, context="mention")],
            )
        )
    return ext_set
