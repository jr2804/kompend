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

import logging
from unittest.mock import MagicMock, patch

import pytest

from kompend.agents.failure import EmptyAnswerError
from kompend.compilation import GraphCompiler
from kompend.compilation.flows.generation import prepare_concepts
from kompend.compilation.schema import (
    Concept,
    Entity,
    GenerationOutput,
    GenerationPage,
)
from kompend.compilation.two_phase import Extraction, ExtractionSet, TwoPhaseCompiler
from kompend.core.config import BudgetConfig, EmbeddingConfig, LLMConfig
from kompend.core.models import Chunk, ElementType, ResolvedChunk
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.decision import ErrorAction, ErrorDecision
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
            patch("kompend.compilation.flows.generation.segments.decide_error_action", return_value=ErrorAction.RETRY),
            patch(
                "kompend.compilation.flows.generation.segments.decide_unit_adaptation",
                return_value=ErrorDecision(action=ErrorAction.SKIP, shrink=False),
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
            patch("kompend.compilation.flows.generation.segments.decide_error_action", return_value=ErrorAction.RETRY),
            patch(
                "kompend.compilation.flows.generation.segments.decide_unit_adaptation",
                return_value=ErrorDecision(action=ErrorAction.SKIP, shrink=False),
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
            patch("kompend.compilation.flows.generation.segments.decide_error_action", return_value=ErrorAction.RETRY),
            patch(
                "kompend.compilation.flows.generation.segments.decide_unit_adaptation",
                return_value=ErrorDecision(action=ErrorAction.SKIP, shrink=False),
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
