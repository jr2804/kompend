"""Failure-path behaviour of the agentic generation pass loop.

One pass = one agent run. An empty answer ends that pass, not the
segment: the next pass asks again with the material it has, so a model
that returned nothing once does not cost the whole segment. Every other
failure keeps the old behaviour - the segment stops and the compile
fails loudly, so nothing downstream reports a partial wiki as a complete
one.
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


def _page_output() -> GenerationOutput:
    return GenerationOutput(
        pages=[GenerationPage(title="Listening Effort", content="The effort to understand speech.", provenance=["sources/src.pdf"])],
    )


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
