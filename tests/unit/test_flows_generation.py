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
from kompend.compilation._two_phase_core import Extraction, ExtractionSet
from kompend.compilation.schema import (
    Concept,
    GenerationOutput,
    GenerationPage,
)
from kompend.compilation.two_phase import TwoPhaseCompiler
from kompend.core.config import EmbeddingConfig, LLMConfig
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
            patch("kompend.compilation.flows.generation.make_agentic_generation_agent", return_value=agent),
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
            patch("kompend.compilation.flows.generation.make_agentic_generation_agent", return_value=agent),
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
            patch("kompend.compilation.flows.generation.make_agentic_generation_agent", return_value=agent),
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
            patch("kompend.compilation.flows.generation.make_agentic_generation_agent", return_value=agent),
            pytest.raises(LLMOutputError, match=r"produced no page\(s\)"),
        ):
            compiler.compile(_chunks())

        assert agent.run.call_count == 2


def _compiler() -> TwoPhaseCompiler:
    config = PipelineConfig(
        source_dir=make_tmp_vault(),
        output_dir=make_tmp_vault(),
        llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"),
        embedding=EmbeddingConfig(model="hash"),
    )
    config.no_cache = True
    return TwoPhaseCompiler(config=config, base=GraphCompiler("model"))


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
