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

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from knox.compilation._two_phase_core import Extraction
from knox.compilation.flows import extraction as ext_module
from knox.compilation.flows.extraction import (
    _compact_history_inplace,
    _estimate_history_tokens,
    _estimate_tokens,
    _synthetic_history_response,
)
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
        with patch("knox.compilation.flows.extraction.ExtractionReduceAgent") as mock_agent_cls:
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

        with patch("knox.compilation.flows.extraction.ExtractionReduceAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run.side_effect = LLMOutputError("boom")
            mock_agent_cls.return_value = mock_agent

            result = compiler._reduce_extractions("doc.md", parts)
            # Falls back to naive merge
            assert "A." in result.summary
            assert "B." in result.summary


# ── History compaction helpers (extract_incremental regression guard) ────


class TestHistoryCompaction:
    """Regression tests for the message-history compaction added to fix
    the empty-response bug in long-document incremental extraction.

    These tests exercise the helpers directly — no LLM call, no agent.
    """

    def test_estimate_tokens_is_chars_div_4(self) -> None:

        assert _estimate_tokens("") == 1  # min 1
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("abcdefgh") == 2

    def test_estimate_history_tokens_sums_part_contents(self) -> None:


        history = [
            ModelRequest(parts=[UserPromptPart(content="hello world!")]),  # 12 chars → 3 tokens
            ModelResponse(parts=[TextPart(content="reply here")]),  # 10 chars → 2 tokens
        ]
        # 22 chars / 4 = 5 tokens
        assert _estimate_history_tokens(history) == 5

    def test_compact_drops_oldest_pair_until_under_budget(self) -> None:


        # Two pairs of (request, response). Budget set so the oldest pair
        # must be dropped but the most-recent pair survives.
        history = [
            ModelRequest(parts=[UserPromptPart(content="x" * 400)]),  # 100 tokens
            ModelResponse(parts=[TextPart(content="y" * 400)]),  # 100 tokens
            ModelRequest(parts=[UserPromptPart(content="a" * 40)]),  # 10 tokens
            ModelResponse(parts=[TextPart(content="b" * 40)]),  # 10 tokens
        ]
        # Total = 220 tokens. Budget = 50 → must drop the first pair.
        _compact_history_inplace(history, max_tokens=50)
        assert len(history) == 2
        # The most-recent pair is retained.
        first_content = getattr(history[0].parts[0], "content", "")
        assert isinstance(first_content, str)
        assert first_content.startswith("a")

    def test_compact_keeps_at_least_last_pair_when_still_over_budget(self) -> None:


        # One pair that alone exceeds the budget. Compaction must not
        # delete everything — the agent needs at least one pair to continue.
        history = [
            ModelRequest(parts=[UserPromptPart(content="x" * 1000)]),
            ModelResponse(parts=[TextPart(content="y" * 1000)]),
        ]
        _compact_history_inplace(history, max_tokens=10)
        assert len(history) == 2  # nothing dropped

    def test_compact_empty_history_is_noop(self) -> None:

        history: list[object] = []
        _compact_history_inplace(history, max_tokens=10)
        assert history == []

    def test_synthetic_history_response_summarises_extraction(self) -> None:

        ext = Extraction(
            source="doc.md",
            summary="Doc is about X.",
            entities=[{"name": "Entity1"}, {"name": "Entity2"}],
            concepts=[{"name": "ConceptA"}, {"name": "ConceptB"}],
            tags=["x"],
        )
        msg = _synthetic_history_response(ext)
        assert isinstance(msg, ModelResponse)
        text = getattr(msg.parts[0], "content", "")
        assert isinstance(text, str)
        assert "Doc is about X." in text
        assert "ConceptA" in text
        assert "Entity1" in text


class TestExtractIncrementalHistoryCompaction:
    """End-to-end check: extract_incremental invokes the compaction helper
    between batches and on cache hits. Mocks the agent so no real LLM call.
    """

    def test_cache_hit_appends_synthetic_response(self) -> None:
        """When a batch is served from cache, history gains a synthetic summary."""
        compiler = _make_compiler()
        compiler._config.no_cache = False

        cached_ext = Extraction(
            source="doc.md",
            summary="Cached summary.",
            entities=[],
            concepts=[{"name": "CachedConcept"}],
            tags=[],
        )
        # Patch the cache store to always hit on the first batch.
        compiler._cache_store.get = MagicMock(return_value=cached_ext)

        # Patch _synthetic_history_response so we can assert it was called.
        with patch.object(ext_module, "_synthetic_history_response") as synth_mock, \
             patch.object(ext_module, "_compact_history_inplace") as compact_mock:
            results, stopped = ext_module.extract_incremental(
                compiler,
                source="doc.md",
                batches_list=["batch 1 text"],
                extraction_system="sys",
            )

        assert stopped is False
        assert len(results) == 1
        assert results[0] is cached_ext
        synth_mock.assert_called_once_with(cached_ext)
        compact_mock.assert_called_once()
