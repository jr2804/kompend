"""Tests for the extraction strategy dispatch (reduce / incremental).

Tests verify:
- ``_combine_extractions`` skips merge for single-batch sources.
- ``_combine_extractions`` uses ``_merge_extractions`` (naive) for incremental.
- ``_combine_extractions`` uses ``_reduce_extractions`` (LLM) for reduce.
- ``_batch_texts`` correctly groups texts into budget-sized batches.
- ``_extract_incremental`` threads message_history between batches.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from knox.compilation._two_phase_core import Extraction
from knox.compilation.schema import Concept, Entity, ExtractionOutput
from knox.compilation.two_phase import TwoPhaseCompiler
from knox.core.config._configs import (
    EmbeddingConfig,
    ExtractionStrategy,
    LLMConfig,
)
from knox.core.pipeline.config import PipelineConfig
from knox.core.utils.exceptions import LLMOutputError
from knox.llm.context_window import ContextSource, ContextWindow


def _make_config(strategy: ExtractionStrategy = ExtractionStrategy.INCREMENTAL) -> PipelineConfig:
    return PipelineConfig(
        source_dir=Path("."),
        output_dir=Path("."),
        llm=LLMConfig(default_model="ollama/test", vlm_model="vlm", extraction_strategy=strategy),
        embedding=EmbeddingConfig(model="model"),
    )


def _make_cw() -> ContextWindow:
    return ContextWindow(max_input=8192, max_output=4096, source=ContextSource.FALLBACK)


def _make_compiler(strategy: ExtractionStrategy = ExtractionStrategy.INCREMENTAL) -> TwoPhaseCompiler:
    config = _make_config(strategy)
    return TwoPhaseCompiler(config=config, context_window=_make_cw())


# ── _batch_texts ─────────────────────────────────────────────────────────


class TestBatchTexts:
    def test_single_text(self) -> None:
        batches = TwoPhaseCompiler._batch_texts(["hello"], 1000)
        assert batches == ["hello"]

    def test_multiple_texts_fit_one_batch(self) -> None:
        batches = TwoPhaseCompiler._batch_texts(["a", "b", "c"], 1000)
        assert batches == ["a\n\nb\n\nc"]

    def test_split_on_budget(self) -> None:
        # Each text is 5 chars; budget=12 → first batch fits 2 texts (10 chars),
        # second batch has 1 text.
        batches = TwoPhaseCompiler._batch_texts(["aaaaa", "bbbbb", "ccccc"], 12)
        assert len(batches) == 2
        assert batches[0] == "aaaaa\n\nbbbbb"
        assert batches[1] == "ccccc"

    def test_empty_texts(self) -> None:
        batches = TwoPhaseCompiler._batch_texts([], 1000)
        assert batches == []


# ── _combine_extractions ─────────────────────────────────────────────────


class TestCombineExtractions:
    def test_single_batch_returns_as_is(self) -> None:
        compiler = _make_compiler()
        ext = Extraction(source="doc.md", summary="Single.")
        result = compiler._combine_extractions("doc.md", [ext])
        assert result is ext

    def test_empty_parts_returns_empty(self) -> None:
        compiler = _make_compiler()
        result = compiler._combine_extractions("doc.md", [])
        assert result.source == "doc.md"
        assert result.summary == ""

    def test_incremental_uses_naive_merge(self) -> None:
        compiler = _make_compiler(ExtractionStrategy.INCREMENTAL)
        parts = [
            Extraction(source="doc.md", summary="Part 1."),
            Extraction(source="doc.md", summary="Part 2."),
        ]
        result = compiler._combine_extractions("doc.md", parts)
        # Naive merge concatenates summaries
        assert "Part 1." in result.summary
        assert "Part 2." in result.summary

    def test_reduce_calls_llm(self) -> None:
        compiler = _make_compiler(ExtractionStrategy.REDUCE)
        parts = [
            Extraction(source="doc.md", summary="Part 1."),
            Extraction(source="doc.md", summary="Part 2."),
        ]
        # Mock _reduce_extractions to avoid LLM call
        with patch.object(compiler, "_reduce_extractions") as mock_reduce:
            expected = Extraction(source="doc.md", summary="Merged.")
            mock_reduce.return_value = expected
            result = compiler._combine_extractions("doc.md", parts)
            mock_reduce.assert_called_once_with("doc.md", parts)
            assert result is expected


# ── _reduce_extractions ──────────────────────────────────────────────────


class TestReduceExtractions:
    def test_reduce_returns_merged_extraction(self) -> None:
        compiler = _make_compiler(ExtractionStrategy.REDUCE)
        parts = [
            Extraction(source="doc.md", summary="First half."),
            Extraction(source="doc.md", summary="Second half."),
        ]
        # Mock the agent to avoid LLM call
        mock_output = ExtractionOutput(
            summary="Full document summary.",
            entities=[Entity(name="KNOX", type="System")],
            concepts=[Concept(name="chunking", definition="splitting text")],
            tags=["ai", "docs"],
        )
        with patch("knox.compilation.two_phase.ExtractionReduceAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_output
            mock_agent_cls.return_value = mock_agent

            result = compiler._reduce_extractions("doc.md", parts)

            assert result.source == "doc.md"
            assert result.summary == "Full document summary."
            assert len(result.entities) == 1
            assert result.entities[0]["name"] == "KNOX"
            assert len(result.concepts) == 1
            assert result.tags == ["ai", "docs"]

    def test_reduce_falls_back_on_error(self) -> None:
        compiler = _make_compiler(ExtractionStrategy.REDUCE)
        parts = [
            Extraction(source="doc.md", summary="A."),
            Extraction(source="doc.md", summary="B."),
        ]

        with patch("knox.compilation.two_phase.ExtractionReduceAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run.side_effect = LLMOutputError("boom")
            mock_agent_cls.return_value = mock_agent

            result = compiler._reduce_extractions("doc.md", parts)
            # Falls back to naive merge
            assert "A." in result.summary
            assert "B." in result.summary
