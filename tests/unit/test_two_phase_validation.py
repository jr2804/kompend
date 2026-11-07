"""Placeholder-page rejection at the agentic generation ingestion gate.

The two-phase writer can self-test its ``final_result`` tool and submit
placeholder pages instead of summaries — bodies of filler text, bare
"x" titles, pages with no provenance at all. A reviewed wiki carried
exactly such pages (``concepts/x.md``, ``entities/entity/rs-polq-cli.md``),
so the gate refuses the whole submission the moment a placeholder page
enters the pipeline: the unit fails on the existing empty-answer path,
and the placeholder page is never written to the wiki.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, SqliteStepStore

from kompend.agents.failure import AgentFailureKind, classify_agent_failure
from kompend.compilation import GraphCompiler
from kompend.compilation._two_phase_core import Extraction, ExtractionSet
from kompend.compilation.flows.generation import (
    PlaceholderMarker,
    PlaceholderPageError,
    _placeholder_reasons,
    _reject_placeholder_pages,
    generate_all_pages_agentic,
)
from kompend.compilation.schema import (
    Concept,
    GenerationOutput,
    GenerationPage,
)
from kompend.compilation.step_recovery import generation_conversation, pass_run_id, segment_run_prefix
from kompend.compilation.two_phase import TwoPhaseCompiler
from kompend.core.config import EmbeddingConfig, LLMConfig
from kompend.core.models import Chunk, ElementType, ResolvedChunk
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.utils.exceptions import LLMOutputError
from tests.conftest import make_tmp_vault


class TestPlaceholderReasons:
    """Per-page checks: what makes a submitted page a placeholder."""

    def test_valid_page_has_no_reasons(self) -> None:
        """A page with provenance, real body and a real title is legitimate."""
        page = GenerationPage(
            title="Listening Effort",
            content="The effort to understand speech in noise.",
            provenance=["sources/src.pdf"],
        )
        assert _placeholder_reasons(page) == []

    def test_empty_provenance_is_rejected(self) -> None:
        """A page with no provenance had no source to summarize."""
        page = GenerationPage(title="Listening Effort", content="The effort to understand speech.")
        reasons = _placeholder_reasons(page)
        assert len(reasons) == 1
        assert "empty provenance" in reasons[0]

    def test_blank_provenance_entries_do_not_count(self) -> None:
        """Whitespace-only provenance entries are as good as none."""
        page = GenerationPage(title="Listening Effort", content="The effort to understand speech.", provenance=["  ", ""])
        reasons = _placeholder_reasons(page)
        assert len(reasons) == 1
        assert "empty provenance" in reasons[0]

    @pytest.mark.parametrize("marker", list(PlaceholderMarker))
    def test_each_marker_is_rejected(self, marker: PlaceholderMarker) -> None:
        """Every placeholder phrase rejects a body that carries it."""
        page = GenerationPage(
            title="Listening Effort",
            content=f"The model wrote: {marker.value} — nothing more.",
            provenance=["sources/src.pdf"],
        )
        reasons = _placeholder_reasons(page)
        assert any("filler body text" in reason for reason in reasons)

    def test_marker_match_is_case_insensitive(self) -> None:
        """Filler detection ignores case: "Placeholder Removed" is caught."""
        page = GenerationPage(title="Listening Effort", content="Placeholder Removed.", provenance=["sources/src.pdf"])
        reasons = _placeholder_reasons(page)
        assert any("placeholder removed" in reason for reason in reasons)

    def test_prose_mentioning_test_words_is_not_rejected(self) -> None:
        """A real summary may mention placeholders or tests — it passes."""
        page = GenerationPage(
            title="Listening Effort",
            content="The effort to understand speech. A placeholder in the app shows test values while listening.",
            provenance=["sources/src.pdf"],
        )
        assert _placeholder_reasons(page) == []

    def test_single_character_title_is_rejected(self) -> None:
        """A bare "x" title is a test identity, not a page name."""
        page = GenerationPage(title="x", content="Real prose body about the effort to understand speech.", provenance=["sources/src.pdf"])
        reasons = _placeholder_reasons(page)
        assert any("shorter than" in reason for reason in reasons)

    def test_two_character_title_passes(self) -> None:
        """Two characters are a real title: "3D" is legitimate."""
        page = GenerationPage(title="3D", content="Real prose body about the effort to understand speech.", provenance=["sources/src.pdf"])
        assert _placeholder_reasons(page) == []

    def test_empty_title_passes(self) -> None:
        """An empty title is not a placeholder — the absorber names the page."""
        page = GenerationPage(title="", content="Real prose body about the effort to understand speech.", provenance=["sources/src.pdf"])
        assert _placeholder_reasons(page) == []


class TestRejectPlaceholderPages:
    """The gate refuses a submission that carries any placeholder page."""

    def test_valid_submission_passes(self) -> None:
        """A submission of legitimate pages raises nothing."""
        gen_output = GenerationOutput(
            pages=[
                GenerationPage(title="Listening Effort", content="Real prose body about listening.", provenance=["sources/src.pdf"]),
                GenerationPage(title="Noise", content="More real prose about noise.", provenance=["sources/src.pdf"]),
            ]
        )
        _reject_placeholder_pages(gen_output, unit="src.pdf")

    def test_empty_submission_passes(self) -> None:
        """No pages — nothing to reject; the pass loop handles that its way."""
        _reject_placeholder_pages(GenerationOutput(pages=[]), unit="src.pdf")

    def test_mixed_submission_names_every_offender(self) -> None:
        """One placeholder page refuses the whole submission, naming it."""
        gen_output = GenerationOutput(
            pages=[
                GenerationPage(title="Listening Effort", content="Real prose body about listening.", provenance=["sources/src.pdf"]),
                GenerationPage(title="x", content="Placeholder removed."),
            ]
        )
        with pytest.raises(PlaceholderPageError) as excinfo:
            _reject_placeholder_pages(gen_output, unit="seg 'src.pdf'")
        message = str(excinfo.value)
        assert "1 placeholder page(s)" in message
        assert "'x'" in message
        assert "Listening Effort" not in message
        assert "filler body text" in message
        assert "empty provenance" in message
        assert "shorter than" in message

    def test_error_routes_through_the_empty_answer_path(self) -> None:
        """PlaceholderPageError classifies as an empty answer, not OTHER."""
        kind = classify_agent_failure(PlaceholderPageError("seg: rejected 1 placeholder page(s)"))
        assert kind is AgentFailureKind.EMPTY_ANSWER


class TestAgenticIngestionGate:
    """Loop level: a placeholder page never reaches the wiki."""

    def test_placeholder_pages_fail_the_segment_loudly(self, caplog: pytest.LogCaptureFixture) -> None:
        """Placeholder output on every pass runs out of passes, then fails."""
        compiler = _compiler()
        agent = MagicMock()
        # A list side_effect: a bare GenerationOutput is iterable (pydantic
        # BaseModel), so MagicMock would yield its field tuples instead.
        agent.run.side_effect = [_placeholder_output(), _placeholder_output()]

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.make_agentic_generation_agent", return_value=agent),
            caplog.at_level(logging.INFO),
            pytest.raises(LLMOutputError, match=r"produced no page\(s\)"),
        ):
            compiler.compile(_chunks())

        assert agent.run.call_count == 2
        assert "agentic-generation" in caplog.text
        assert "empty_answer" in caplog.text

    def test_placeholder_pass_then_good_pass_writes_only_the_good_page(self) -> None:
        """Pass 1 submits a placeholder, pass 2 a real page: only the real one lands."""
        compiler = _compiler()
        agent = MagicMock()
        agent.run.side_effect = [
            _placeholder_output(),
            _page_output(),
        ]

        with (
            patch.object(compiler, "extract", return_value=_extraction_set()),
            patch("kompend.compilation.flows.generation.make_agentic_generation_agent", return_value=agent),
        ):
            wiki = compiler.compile(_chunks())

        assert agent.run.call_count == 2
        titles = [node.title for node in wiki.nodes]
        assert "Listening Effort" in titles
        assert "X" not in titles


class TestRecoveryIngestionGate:
    """Recovery level: a poisoned transcript cannot replay its placeholder."""

    def test_replayed_placeholder_transcript_reruns_fresh(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A finished transcript whose final_result carries a placeholder
        page is rejected and recorded; the segment re-runs fresh.
        """
        store = SqliteStepStore(database=tmp_path / "steps.db")
        compiler = _compiler_no_cache()
        deps = {"MODEL": "ollama/test"}
        compiler._active_gen_deps = deps
        conversation = generation_conversation(deps)
        run_id = pass_run_id(segment_run_prefix(conversation, 0), 1, 1)
        _seed_run(store, run_id, conversation, _snapshot(run_id, _placeholder_payload()))

        agent = MagicMock()
        agent.result_type = GenerationOutput
        calls: list[GenerationOutput] = []

        def _run(fresh_deps: object, **run_kwargs: object) -> GenerationOutput:
            # Drain the broker like the real model would, or the pass loop
            # runs a second pass for the unconsumed concept.
            broker = getattr(fresh_deps, "concept_broker", None)
            if broker is not None:
                broker.fetch(getattr(fresh_deps, "source_id", ""), limit=10)
            calls.append(_page_output())
            return calls[-1]

        agent.run = _run

        with (
            patch("kompend.compilation.flows.generation.make_agentic_generation_agent", return_value=agent),
            caplog.at_level(logging.INFO),
        ):
            nodes, _, _ = generate_all_pages_agentic(
                compiler,
                [{"name": "Listening Effort", "kind": "concept"}],
                [[("a", "x" * 500)]],
                [],
                pinned_facts=[],
                known_titles=[],
                step_store=store,
            )

        # The poisoned transcript was rejected and recorded; the fresh rerun
        # produced the page and the placeholder never reached the wiki.
        assert "agentic-generation-recovery" in caplog.text
        assert len(calls) == 1
        titles = [node.title for node in nodes]
        assert "Listening Effort" in titles
        assert "X" not in titles


def _compiler() -> TwoPhaseCompiler:
    config = PipelineConfig(
        source_dir=make_tmp_vault(),
        output_dir=make_tmp_vault(),
        llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"),
        embedding=EmbeddingConfig(model="hash"),
    )
    config.no_cache = True
    return TwoPhaseCompiler(config=config, base=GraphCompiler("model"))


def _compiler_no_cache() -> TwoPhaseCompiler:
    """A compiler with the step-store resume enabled (no_cache off)."""
    vault = make_tmp_vault()
    config = PipelineConfig(
        base_dir=vault,
        source_dir=vault / "sources",
        output_dir=vault / "wiki",
        llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"),
        embedding=EmbeddingConfig(model="hash"),
    )
    return TwoPhaseCompiler(config=config)


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
    """A legitimate page: real title, real body, cited provenance."""
    return GenerationOutput(
        pages=[GenerationPage(title="Listening Effort", content="The effort to understand speech.", provenance=["sources/src.pdf"])],
    )


def _placeholder_output() -> GenerationOutput:
    """The reviewed-wiki artifact: bare "x" title, filler body, no provenance."""
    return GenerationOutput(
        pages=[GenerationPage(title="X", content="Placeholder removed.")],
    )


def _placeholder_payload() -> list:
    """A settled final_result round whose page is the reviewed-wiki artifact."""
    payload = json.dumps(
        {"pages": [{"title": "X", "content": "Placeholder removed."}]},
    )
    return [
        ModelResponse(parts=[ToolCallPart(tool_name="final_result", args=payload, tool_call_id="call-1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="final_result", content="ok", tool_call_id="call-1")]),
    ]


def _snapshot(run_id: str, messages: list) -> ContinuableSnapshot:
    return ContinuableSnapshot(run_id=run_id, step_index=1, messages=messages, state="complete")


def _seed_run(store: SqliteStepStore, run_id: str, conversation: str, snapshot: ContinuableSnapshot) -> None:
    async def _seed() -> None:
        await store.register_run(RunRecord(run_id=run_id, conversation_id=conversation))
        await store.save_snapshot(snapshot)

    asyncio.run(_seed())


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
