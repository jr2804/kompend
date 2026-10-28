"""Tests for the extraction strategy dispatch (reduce / incremental).

Tests verify:
- ``_combine_extractions`` skips merge for single-batch sources.
- ``_combine_extractions`` uses ``_merge_extractions`` (naive) for incremental.
- ``_combine_extractions`` uses ``_reduce_extractions`` (LLM) for reduce.
- ``_batch_texts`` correctly groups texts into budget-sized batches.
- ``_extract_incremental`` threads message_history between batches.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from kompend.compilation._two_phase_core import Extraction, ExtractionSet, chars_to_tokens, context_safety_tokens
from kompend.compilation.flows import extraction as ext_module
from kompend.compilation.flows import generation as gen_module
from kompend.compilation.flows.extraction import (
    _compact_history_inplace,
    _estimate_history_tokens,
    _synthetic_history_response,
)
from kompend.compilation.schema import Concept, Entity, ExtractionOutput
from kompend.compilation.two_phase import TwoPhaseCompiler
from kompend.core.cache.store import make_cache_key
from kompend.core.config._configs import (
    EmbeddingConfig,
    ExtractionStrategy,
    LLMConfig,
)
from kompend.core.models.enums import CacheDeps
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.utils.exceptions import LLMOutputError
from kompend.core.utils.tokens import estimate_tokens
from kompend.llm.context_window import ContextSource, ContextWindow
from tests.conftest import make_tmp_vault

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
            entities=[Entity(name="kompend", type="System")],
            concepts=[Concept(name="chunking", definition="splitting text")],
            tags=["ai", "docs"],
        )
        with patch("kompend.compilation.flows.extraction.ExtractionReduceAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_output
            mock_agent_cls.return_value = mock_agent

            result = compiler._reduce_extractions("doc.md", parts)

            assert result.source == "doc.md"
            assert result.summary == "Full document summary."
            assert len(result.entities) == 1
            assert result.entities[0].name == "kompend"
            assert len(result.concepts) == 1
            assert result.tags == ["ai", "docs"]

    def test_reduce_falls_back_on_error(self) -> None:
        compiler = _make_compiler(ExtractionStrategy.REDUCE)
        parts = [
            Extraction(source="doc.md", summary="A."),
            Extraction(source="doc.md", summary="B."),
        ]

        with patch("kompend.compilation.flows.extraction.ExtractionReduceAgent") as mock_agent_cls:
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

        assert estimate_tokens("") == 1  # min 1
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcdefgh") == 2

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

        history: list[ModelMessage] = []
        _compact_history_inplace(history, max_tokens=10)
        assert history == []

    def test_synthetic_history_response_summarises_extraction(self) -> None:

        ext = Extraction(
            source="doc.md",
            summary="Doc is about X.",
            entities=[Entity(name="Entity1"), Entity(name="Entity2")],
            concepts=[Concept(name="ConceptA"), Concept(name="ConceptB")],
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
            concepts=[Concept(name="CachedConcept")],
            tags=[],
        )
        # Patch the cache store to always hit on the first batch.
        compiler._cache_store.get = MagicMock(return_value=cached_ext.to_dict())

        # Patch _synthetic_history_response so we can assert it was called.
        with patch.object(ext_module, "_synthetic_history_response") as synth_mock, patch.object(ext_module, "_compact_history_inplace") as compact_mock:
            results, stopped = ext_module.extract_incremental(
                compiler,
                source="doc.md",
                batches_list=["batch 1 text"],
                extraction_system="sys",
            )

        assert stopped is False
        assert len(results) == 1
        # Cache serves a JSON round-trip (dict -> from_dict), so compare by value.
        assert results[0] == cached_ext
        synth_mock.assert_called_once_with(results[0])
        compact_mock.assert_called_once()

    def test_progress_is_reported_for_every_batch(self) -> None:
        """A long document must move the bar inside, not jump at the end."""
        compiler = _make_compiler()
        reports: list[tuple[int, int, str]] = []

        with patch.object(ext_module, "extract_one", return_value=Extraction(source="doc.md")):
            results, stopped = ext_module.extract_incremental(
                compiler,
                source="doc.md",
                batches_list=["batch 1", "batch 2", "batch 3"],
                extraction_system="sys",
                progress=lambda done, total, activity: reports.append((done, total, activity)),
            )

        assert stopped is False
        assert len(results) == 3
        assert [(done, total) for done, total, _ in reports] == [(1, 3), (2, 3), (3, 3)]


class TestExtractionProgressDetail:
    """The extraction line names the document and how far inside it we are."""

    def test_names_the_document_and_the_unit_count(self) -> None:
        detail = ext_module.extraction_detail(3, 100, "PolqaOemLibManualV3.18.pdf", 12, 234, "sections")
        assert detail == "extracting [3/100] PolqaOemLibManualV3.18.pdf (12/234 sections)"

    def test_names_the_running_call(self) -> None:
        detail = ext_module.extraction_detail(3, 100, "doc.pdf", 12, 234, "sections", "consolidating context")
        assert detail == "extracting [3/100] doc.pdf (12/234 sections), consolidating context"

    def test_the_bar_pays_the_document_share_out_in_units(self) -> None:
        """The bar counts characters: a document's share is paid out as its
        units complete, so the bar moves inside a long document.
        """
        compiler = _make_compiler()
        reports: list[tuple[int, int, str]] = []
        compiler.on_progress = lambda current, total, description: reports.append((current, total, description))

        progress = ext_module._source_progress(
            compiler,
            source_idx=1,
            num_sources=2,
            source_name="doc.pdf",
            chars_before=100,
            source_chars=200,
            total_chars_all=600,
            weights=[100, 100],
            unit="sections",
        )

        progress(1, 2, "")
        # Half of the document's share is paid out.
        assert reports[-1][0] == 200
        progress(2, 2, "")
        # The document's whole share is paid out.
        assert reports[-1][0] == 300

    def test_a_source_without_units_is_paid_out_at_once(self) -> None:
        compiler = _make_compiler()
        reports: list[tuple[int, int, str]] = []
        compiler.on_progress = lambda current, total, description: reports.append((current, total, description))

        progress = ext_module._source_progress(
            compiler,
            source_idx=1,
            num_sources=1,
            source_name="empty.pdf",
            chars_before=0,
            source_chars=100,
            total_chars_all=100,
            weights=[],
            unit="sections",
        )

        progress(0, 0, "")
        assert reports[-1][0] == 100


class TestGenerationProgressDetail:
    """The generation line carries counters, not labels.

    A counter without its denominator ("segment 2") and a second unit
    next to the main one ("4/12 concepts covered") are noise — the
    numbers must carry the meaning on their own.
    """

    def test_the_line_carries_two_counters(self) -> None:
        detail = gen_module.generation_detail(57, 338, 4, 12)
        assert detail == "generating [57/338] (4/12)"

    def test_the_line_carries_no_labels(self) -> None:
        detail = gen_module.generation_detail(57, 338, 4, 12)
        assert "segment" not in detail
        assert "concept" not in detail

    def test_a_segment_is_not_announced(self) -> None:
        """No number may sit outside the two counters.

        A bare "segment 2" has no denominator, so it can only confuse.
        """
        detail = gen_module.generation_detail(57, 338, 4, 12)
        residue = detail.replace("[57/338]", "").replace("(4/12)", "")
        assert not any(character.isdigit() for character in residue)

    def test_the_throughput_suffix_stays_last(self) -> None:
        detail = gen_module.generation_detail(57, 338, 4, 12, rate=" · 12.3 tok/s")
        assert detail == "generating [57/338] (4/12) · 12.3 tok/s"

    def test_a_note_names_a_state_not_a_count(self) -> None:
        detail = gen_module.generation_detail(57, 338, note="(recovered)")
        assert detail == "generating [57/338] (recovered)"

    def test_without_a_segment_there_is_one_counter(self) -> None:
        detail = gen_module.generation_detail(57, 338)
        assert detail == "generating [57/338]"


# ── extraction batch sizing ────────────────────────────────────────────


class TestContextSafetyTokens:
    """Shared safety margin used by extraction batching and generation segments."""

    def test_fraction_of_window(self) -> None:
        assert context_safety_tokens(65536) == round(0.08 * 65536)

    def test_floor_wins_on_narrow_windows(self) -> None:
        # 8% of 2048 is 164 — the floor keeps the margin meaningful.
        assert context_safety_tokens(2048) == 512


class TestExtractCharBudget:
    """The batch budget must leave headroom for the safety margin.

    Regression: an exact-fit batch on a formula-dense standard overflowed a
    65536-token local model (65762 tokens), because the chars-per-token
    heuristic undercounts token-dense text.
    """

    def test_reserves_safety_margin(self) -> None:
        compiler = _make_big_window_compiler()
        cpt = compiler._config.llm.default_chars_per_token
        cw = compiler._context_window.max_input

        budget = ext_module._extract_char_budget(compiler)
        content_tokens = chars_to_tokens(budget, chars_per_token=cpt)
        safety = context_safety_tokens(cw)

        # Everything that lands in the request, plus the margin, must fit.
        assert content_tokens + compiler._extract_system_tokens + compiler._context_window.answer_reserve + safety <= cw

    def test_budget_is_smaller_than_without_margin(self) -> None:
        compiler = _make_big_window_compiler()
        cpt = compiler._config.llm.default_chars_per_token
        cw = compiler._context_window.max_input

        with_margin = ext_module._extract_char_budget(compiler)
        without = ext_module.tokens_to_chars(
            cw - compiler._extract_system_tokens - compiler._context_window.answer_reserve,
            chars_per_token=cpt,
        )

        assert with_margin < without
        # The gap is the safety margin, converted to chars (±2 for int truncation).
        expected_gap = ext_module.tokens_to_chars(context_safety_tokens(cw), chars_per_token=cpt)
        assert abs((without - with_margin) - expected_gap) <= 2


# ── extract cache key ───────────────────────────────────────────────────


class TestExtractCacheDeps:
    """The ``extract`` cache key must identify the model.

    Extraction output is model-specific, so an entry written by a previous
    model must never be replayed after switching models (regression: the
    key held only the source text and chunk sizes).
    """

    def test_key_records_model_and_text(self) -> None:
        deps = ext_module._extract_cache_deps(_make_compiler(), "hello")
        assert deps[CacheDeps.MODEL] == "ollama/test"
        assert deps[CacheDeps.SOURCE_TEXT] == hashlib.sha256(b"hello").hexdigest()
        assert deps[CacheDeps.CHUNK_SIZE_MIN] == "2048"
        assert deps[CacheDeps.CHUNK_SIZE_MAX] == "131072"

    def test_switching_model_changes_the_key(self) -> None:
        text = "same text"
        first = _make_compiler()
        second = _make_compiler()
        second._model = "ollama/other"

        deps_a = ext_module._extract_cache_deps(first, text)
        deps_b = ext_module._extract_cache_deps(second, text)

        assert deps_a[CacheDeps.SOURCE_TEXT] == deps_b[CacheDeps.SOURCE_TEXT]
        assert make_cache_key(deps_a) != make_cache_key(deps_b)

    def test_extract_one_looks_up_with_model_dep(self) -> None:
        """A cache read that omits the model cannot invalidate on model change."""
        compiler = _make_compiler()
        compiler._cache_store = MagicMock()
        compiler._cache_store.get_dict.return_value = {
            "source": "s",
            "summary": "",
            "entities": [],
            "concepts": [],
            "tags": [],
        }

        ext_module.extract_one(compiler, "s", "text", "system")

        deps = compiler._cache_store.get_dict.call_args.args[1]
        assert deps[CacheDeps.MODEL] == "ollama/test"

    def test_ignoring_model_drops_it_from_the_key(self) -> None:
        """knox-wflz: ``compile.cache_ignore_model`` keeps extractions across a model switch."""
        compiler = _make_compiler()
        compiler._config.cache_ignore_model = True

        deps = ext_module._extract_cache_deps(compiler, "hello")

        assert CacheDeps.MODEL not in deps
        assert deps[CacheDeps.SOURCE_TEXT] == hashlib.sha256(b"hello").hexdigest()


class TestGenCacheDeps:
    """The ``generate`` cache key must identify the model — unless the vault
    opted out via ``compile.cache_ignore_model`` (mixed-model continuation,
    knox-wflz).
    """

    def test_key_records_model(self) -> None:
        deps = gen_module.gen_cache_deps(_make_compiler(), ExtractionSet())
        assert deps[CacheDeps.MODEL] == "ollama/test"

    def test_ignoring_model_drops_it_from_the_key(self) -> None:
        compiler = _make_compiler()
        compiler._config.cache_ignore_model = True

        deps = gen_module.gen_cache_deps(compiler, ExtractionSet())

        assert CacheDeps.MODEL not in deps
        assert deps[CacheDeps.PINNED_FACTS] == hashlib.sha256(b"").hexdigest()

    def test_switching_model_changes_the_key(self) -> None:
        first = _make_compiler()
        second = _make_compiler()
        second._model = "ollama/other"

        deps_a = gen_module.gen_cache_deps(first, ExtractionSet())
        deps_b = gen_module.gen_cache_deps(second, ExtractionSet())

        assert make_cache_key(deps_a) != make_cache_key(deps_b)


def _make_big_window_compiler() -> TwoPhaseCompiler:
    """Compiler with a realistic cloud-scale window.

    Batch sizing is only meaningful when the window comfortably exceeds the
    reserved output — the 8192-token fixture window is smaller than
    the answer reserve and collapses every budget to the 1-token floor.
    A 65536-token window mirrors the local model this was observed on.
    """
    compiler = _make_compiler()
    compiler._context_window = ContextWindow(max_input=65536, max_output=32768, source=ContextSource.OLLAMA)
    compiler._extract_system_tokens = 1000
    return compiler


def _make_compiler(strategy: ExtractionStrategy = ExtractionStrategy.INCREMENTAL) -> TwoPhaseCompiler:
    config = _make_config(strategy)
    return TwoPhaseCompiler(config=config, context_window=_make_cw())


def _make_config(strategy: ExtractionStrategy = ExtractionStrategy.INCREMENTAL) -> PipelineConfig:
    # Absolute vault root: a CWD-relative source_dir/output_dir would resolve the
    # vault store against the process CWD and write .kompend/ into the repo root.
    vault = make_tmp_vault()
    return PipelineConfig(
        source_dir=vault / "sources",
        output_dir=vault / "wiki",
        llm=LLMConfig(default_model="ollama/test", vlm_model="vlm", extraction_strategy=strategy),
        embedding=EmbeddingConfig(model="model"),
    )


def _make_cw() -> ContextWindow:
    return ContextWindow(max_input=8192, max_output=4096, source=ContextSource.FALLBACK)
