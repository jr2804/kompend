"""Tests for the Q20-N3-N8-N4-Q21 throughput feature chain."""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from kompend.agents.types import VerificationDeps, VerificationVerdict
from kompend.agents.verification import AgenticVerificationAgent
from kompend.compilation._two_phase_core import _FAILURE_THRESHOLD_MIN, normalize_concept_key
from kompend.compilation.flows.generation import (
    _agentic_reserved_tokens,
    generate_all_pages_agentic,
    prepare_source_segments,
)
from kompend.compilation.schema import Concept, Entity, GenerationOutput
from kompend.compilation.two_phase import (
    Extraction,
    ExtractionSet,
    TwoPhaseCompiler,
    cosine_similarity,
    merge_similar_concepts,
    segment_sources,
)
from kompend.core.config import EmbeddingConfig, LLMConfig
from kompend.core.models import Chunk, Citation, CompiledWiki, ElementType, ResolvedChunk, WikiNode, extract_citations, merge_duplicate_citations
from kompend.core.paths import KOMP_DIR, REVIEW_DIR, STATE_FILE
from kompend.core.pipeline.config import Tier
from kompend.core.pipeline.orchestrator import (
    ALL_TIERS,
    TIER_CHUNK,
    TIER_COMPILE,
    TIER_INDEX,
    TIER_OUTPUT,
    TIER_RESOLVE,
    PipelineConfig,
    WikiPipeline,
)
from kompend.core.pipeline.state import (
    SourceState,
    TierState,
    TierStateStore,
    auto_promote_demote,
    classify_compile_status,
)
from kompend.core.store import open_store
from kompend.core.store.schema import PRIVATE_PAGES_TABLE, PROVENANCE_TABLE, VERIFICATION_TABLE
from kompend.provenance import annotate_wiki_with_provenance
from kompend.review import ReviewQueue, ReviewStatus, slugify
from kompend.trust import (
    LIFECYCLE_DRAFT,
    LIFECYCLE_STALE,
    LifecycleEntry,
    LifecycleState,
    LifecycleStore,
    filter_sensitive,
    has_sensitive_content,
    mark_stale,
    mark_wiki_private,
)
from kompend.verification import (
    verify_node,
    verify_wiki,
)
from tests.conftest import make_tmp_vault


class TestTierLevelRanges:
    """TIER_LEVEL and tier range sanity."""

    def test_all_tiers_ordered(self) -> None:
        assert ALL_TIERS == (TIER_INDEX, TIER_CHUNK, TIER_RESOLVE, TIER_COMPILE, TIER_OUTPUT)

    def test_should_run_default_runs_all(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        pipeline = WikiPipeline(config=config, state_store=TierStateStore(Path("/dev/null")))
        for tier in Tier:
            assert pipeline._should_run(tier)

    def test_should_run_output_only(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
            start_at_tier=Tier.OUTPUT,
        )
        pipeline = WikiPipeline(config=config, state_store=TierStateStore(Path("/dev/null")))
        assert not pipeline._should_run(Tier.INDEX)
        assert not pipeline._should_run(Tier.CHUNK)
        assert not pipeline._should_run(Tier.RESOLVE)
        assert not pipeline._should_run(Tier.COMPILE)
        assert pipeline._should_run(Tier.OUTPUT)


class TestTierStateStore:
    """Tier state persistence."""

    def test_empty_state_on_missing_file(self, tmp_path: Path) -> None:
        store = TierStateStore(tmp_path / "missing.json")
        state = store.load()
        assert state.entries == {}

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / STATE_FILE
        store = TierStateStore(path)
        state = TierState(entries={"a": SourceState(stem="a", source_file="a", content_hash="x")})
        store.save(state)
        loaded = store.load()
        assert "a" in loaded.entries
        assert loaded.entries["a"].content_hash == "x"

    def test_record_creates_entry(self, tmp_path: Path) -> None:
        store = TierStateStore(tmp_path / STATE_FILE)
        state = TierState()
        entry = store.record(state, "doc1", Path("doc1.pdf"), "abc123", tier=1)
        assert entry.current_tier == 1
        assert entry.content_hash == "abc123"
        assert state.entries["doc1"].last_compile

    def test_record_updates_existing_entry(self, tmp_path: Path) -> None:
        store = TierStateStore(tmp_path / STATE_FILE)
        state = TierState()
        store.record(state, "doc1", Path("doc1.pdf"), "abc", tier=1)
        store.record(state, "doc1", Path("doc1.pdf"), "abc", tier=3)
        assert state.entries["doc1"].current_tier == 3
        assert state.entries["doc1"].search_hits == 0

    def test_record_hit_increments(self, tmp_path: Path) -> None:
        store = TierStateStore(tmp_path / STATE_FILE)
        state = TierState()
        store.record(state, "doc1", Path("d.pdf"), "h", tier=1)
        store.record_hit(state, "doc1")
        store.record_hit(state, "doc1")
        assert state.entries["doc1"].search_hits == 2

    def test_record_sets_ingested_by(self, tmp_path: Path) -> None:
        store = TierStateStore(tmp_path / STATE_FILE)
        state = TierState()
        store.record(state, "doc1", Path("doc1.pdf"), "abc", tier=1, ingested_by="alice")
        assert state.entries["doc1"].ingested_by == "alice"

    def test_record_preserves_ingested_by_when_unset(self, tmp_path: Path) -> None:
        # The compile path does not pass ingested_by — it must preserve the
        # value recorded during ingest rather than clobbering it.
        store = TierStateStore(tmp_path / STATE_FILE)
        state = TierState()
        store.record(state, "doc1", Path("doc1.pdf"), "abc", tier=1, ingested_by="alice")
        store.record(state, "doc1", Path("doc1.pdf"), "abc", tier=2)
        assert state.entries["doc1"].ingested_by == "alice"
        assert state.entries["doc1"].current_tier == 2

    def test_record_empty_ingested_when_disabled(self, tmp_path: Path) -> None:
        store = TierStateStore(tmp_path / STATE_FILE)
        state = TierState()
        store.record(state, "doc1", Path("doc1.pdf"), "abc", tier=1, ingested_by="")
        assert state.entries["doc1"].ingested_by == ""

    def test_ingested_by_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / STATE_FILE
        store = TierStateStore(path)
        state = TierState()
        store.record(state, "doc1", Path("d.pdf"), "h", tier=1, ingested_by="bob")
        store.save(state)
        loaded = store.load()
        assert loaded.entries["doc1"].ingested_by == "bob"

    def test_legacy_state_file_is_not_read(self, tmp_path: Path) -> None:
        # Clean-cut policy (B3): legacy .kompend JSON is never read. A pre-existing
        # compile_state.json is ignored; the store is the single source.
        path = tmp_path / STATE_FILE
        path.write_text(
            json.dumps({"entries": {"doc1": {"stem": "doc1", "source_file": "d", "content_hash": "h", "current_tier": 1}}}),
            encoding="utf-8",
        )
        assert TierStateStore(path).load().entries == {}

    def test_legacy_corrupt_state_file_is_ignored(self, tmp_path: Path) -> None:
        # Clean-cut policy (B3): a corrupt legacy file is not read, so it can
        # neither raise nor silently reset — it is simply inert.
        path = tmp_path / STATE_FILE
        path.write_text("not json", encoding="utf-8")
        assert TierStateStore(path).load().entries == {}

    def test_tier_distribution(self) -> None:
        state = TierState(
            entries={
                "a": SourceState(stem="a", source_file="", content_hash="", current_tier=1),
                "b": SourceState(stem="b", source_file="", content_hash="", current_tier=3),
                "c": SourceState(stem="c", source_file="", content_hash="", current_tier=3),
            }
        )
        assert state.tier_distribution() == {1: 1, 3: 2}


class TestClassifyCompileStatus:
    """classify_compile_status() classification logic."""

    def test_not_initialized(self) -> None:
        info = classify_compile_status(
            state_dir_exists=False,
            sources_count=0,
            tier_state=None,
            wiki_exists=False,
        )
        assert info.status.value == "not-initialized"

    def test_no_sources(self) -> None:
        info = classify_compile_status(
            state_dir_exists=True,
            sources_count=0,
            tier_state=None,
            wiki_exists=False,
        )
        assert info.status.value == "no-sources"

    def test_not_indexed(self) -> None:
        info = classify_compile_status(
            state_dir_exists=True,
            sources_count=3,
            tier_state=None,
            wiki_exists=False,
        )
        assert info.status.value == "not-indexed"
        assert info.total == 3

    def test_in_progress(self) -> None:
        state = TierState(
            entries={
                "a": SourceState(stem="a", source_file="", content_hash="", current_tier=1),
                "b": SourceState(stem="b", source_file="", content_hash="", current_tier=2),
            }
        )
        info = classify_compile_status(
            state_dir_exists=True,
            sources_count=2,
            tier_state=state,
            wiki_exists=False,
        )
        assert info.status.value == "in-progress"
        assert info.indexed == 2
        assert info.compiled == 0

    def test_partially_compiled_by_tier(self) -> None:
        state = TierState(
            entries={
                "a": SourceState(stem="a", source_file="", content_hash="", current_tier=4),
                "b": SourceState(stem="b", source_file="", content_hash="", current_tier=1),
            }
        )
        info = classify_compile_status(
            state_dir_exists=True,
            sources_count=2,
            tier_state=state,
            wiki_exists=False,
        )
        assert info.status.value == "partially-compiled"
        assert info.compiled == 1

    def test_partially_compiled_by_wiki(self) -> None:
        # Wiki exists but not all entries at tier 4
        state = TierState(
            entries={
                "a": SourceState(stem="a", source_file="", content_hash="", current_tier=3),
                "b": SourceState(stem="b", source_file="", content_hash="", current_tier=3),
            }
        )
        info = classify_compile_status(
            state_dir_exists=True,
            sources_count=2,
            tier_state=state,
            wiki_exists=True,
        )
        assert info.status.value == "partially-compiled"

    def test_compiled(self) -> None:
        state = TierState(
            entries={
                "a": SourceState(stem="a", source_file="", content_hash="", current_tier=4),
                "b": SourceState(stem="b", source_file="", content_hash="", current_tier=4),
            }
        )
        info = classify_compile_status(
            state_dir_exists=True,
            sources_count=2,
            tier_state=state,
            wiki_exists=True,
        )
        assert info.status.value == "compiled"
        assert info.compiled == 2

    def test_broken(self) -> None:
        info = classify_compile_status(
            state_dir_exists=True,
            sources_count=5,
            tier_state=None,
            wiki_exists=False,
            error="Invalid JSON at line 1",
        )
        assert info.status.value == "broken"
        assert "Invalid JSON" in info.error
        assert "broken" in info.label()

    def test_label_formats(self) -> None:
        """label() returns human-readable strings."""
        info = classify_compile_status(
            state_dir_exists=False,
            sources_count=0,
            tier_state=None,
            wiki_exists=False,
        )
        assert info.label() == "not initialized"

        state = TierState(
            entries={
                "a": SourceState(stem="a", source_file="", content_hash="", current_tier=4),
            }
        )
        info = classify_compile_status(
            state_dir_exists=True,
            sources_count=1,
            tier_state=state,
            wiki_exists=True,
        )
        assert "compiled" in info.label()


class TestAutoPromoteDemote:
    """Auto-promote / auto-demote rules."""

    def test_promote_after_threshold(self) -> None:
        state = TierState(
            entries={
                "a": SourceState(
                    stem="a",
                    source_file="",
                    content_hash="",
                    current_tier=1,
                    search_hits=5,
                )
            }
        )
        changed = auto_promote_demote(state, promote_threshold=3, demote_days=90)
        assert changed == 1
        assert state.entries["a"].current_tier == 3

    def test_no_promote_below_threshold(self) -> None:
        state = TierState(
            entries={
                "a": SourceState(
                    stem="a",
                    source_file="",
                    content_hash="",
                    current_tier=1,
                    search_hits=1,
                )
            }
        )
        changed = auto_promote_demote(state, promote_threshold=3)
        assert changed == 0
        assert state.entries["a"].current_tier == 1

    def test_demote_old_entries(self) -> None:
        old = (datetime.now(UTC) - timedelta(days=200)).isoformat(timespec="seconds")
        state = TierState(
            entries={
                "a": SourceState(
                    stem="a",
                    source_file="",
                    content_hash="",
                    current_tier=3,
                    last_compile=old,
                )
            }
        )
        changed = auto_promote_demote(state, promote_threshold=3, demote_days=90)
        assert changed == 1
        assert state.entries["a"].current_tier == 1


class TestWikiPipelineTiered:
    """WikiPipeline.run() and run_tier() with tier range filtering."""

    def test_pipeline_config_defaults(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        assert config.start_at_tier == "index"
        assert config.stop_at_tier == "output"

    def test_start_level_default(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        pipeline = WikiPipeline(config=config, state_store=TierStateStore(Path("/dev/null")))
        assert pipeline.start_level() == 0
        assert pipeline.stop_level() == 4

    def test_start_level_output_only(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
            start_at_tier=Tier.OUTPUT,
        )
        pipeline = WikiPipeline(config=config, state_store=TierStateStore(Path("/dev/null")))
        assert pipeline.start_level() == 4
        assert pipeline.stop_level() == 4

    def test_validate_prerequisites_empty_state_raises(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
            start_at_tier=Tier.RESOLVE,
        )
        pipeline = WikiPipeline(config=config, state_store=TierStateStore(Path("/dev/null")))
        with pytest.raises(ValueError, match="no previous run"):
            pipeline._validate_prerequisites()

    def test_run_tier_unknown_raises(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        pipeline = WikiPipeline(config=config, state_store=TierStateStore(Path("/dev/null")))
        with pytest.raises(ValueError, match="Unknown tier"):
            pipeline.run_tier("nonsense")

    def test_list_tiers_returns_all(self, tmp_path: Path) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=tmp_path,
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        pipeline = WikiPipeline(
            config=config,
            state_store=TierStateStore(tmp_path / STATE_FILE),
        )
        tiers = pipeline.list_tiers()
        tier_names = {t["tier"] for t in tiers}
        assert {TIER_INDEX, TIER_CHUNK, TIER_RESOLVE, TIER_COMPILE, TIER_OUTPUT} <= tier_names


# ===========================================================================
#  N3: Two-phase compiler
# ===========================================================================


class TestExtractionData:
    """Extraction / ExtractionSet behaviour."""

    def test_concept_pages_merges_by_name(self) -> None:
        ext = ExtractionSet(
            extractions=[
                Extraction(
                    source="a.pdf",
                    concepts=[Concept(name="attention", definition="x")],
                ),
                Extraction(
                    source="b.pdf",
                    concepts=[Concept(name="attention", definition="y")],
                ),
            ]
        )
        pages = ext.concept_pages()
        assert len(pages) == 1
        assert pages[0]["name"] == "attention"
        assert sorted(pages[0]["sources"]) == ["a.pdf", "b.pdf"]
        assert sorted(pages[0]["definitions"]) == ["x", "y"]

    def test_entity_index(self) -> None:
        ext = ExtractionSet(
            extractions=[
                Extraction(source="a.pdf", entities=[Entity(name="kompend", type="PROJECT")]),
                Extraction(source="b.pdf", entities=[Entity(name="kompend", type="PROJECT")]),
            ]
        )
        idx = ext.entity_index()
        assert idx["kompend"] == ["a.pdf", "b.pdf"]


class TestNormalizeConceptKey:
    """Normalization for concept merge-key dedup (Option A)."""

    def test_strip_acronym(self) -> None:

        assert normalize_concept_key("Deep Neural Networks (DNNs)") == "deep neural network"
        assert normalize_concept_key("Short-Time Objective Intelligibility (STOI)") == "short time objective intelligibility"
        assert normalize_concept_key("Binaural Unmasking (BU)") == "binaural unmasking"

    def test_strip_mixed_case_acronym(self) -> None:

        assert normalize_concept_key("Mean Temporal Distance (M-Measure)") == "mean temporal distance"
        assert normalize_concept_key("Mean Temporal Distance (MTD)") == "mean temporal distance"

    def test_keep_year_in_parens(self) -> None:

        assert normalize_concept_key("ITU-T P.58 (2023)") == "itu t p.58 (2023)"

    def test_hyphen_to_space(self) -> None:

        assert normalize_concept_key("Short-Time") == "short time"

    def test_case_fold(self) -> None:

        assert normalize_concept_key("Deep Neural Network") == "deep neural network"
        assert normalize_concept_key("Deep neural network") == "deep neural network"

    def test_plural_collapse(self) -> None:

        # -s → singular
        assert normalize_concept_key("Networks") == "network"
        assert normalize_concept_key("Posteriorgrams") == "posteriorgram"


class TestConceptPagesMerge:
    """concept_pages() merges variants via normalized key."""

    def test_merge_acronym_variant(self) -> None:
        ext = ExtractionSet(
            extractions=[
                Extraction(
                    source="a.pdf",
                    concepts=[Concept(name="Deep Neural Network", definition="x")],
                ),
                Extraction(
                    source="b.pdf",
                    concepts=[Concept(name="Deep Neural Networks (DNNs)", definition="y")],
                ),
            ]
        )
        pages = ext.concept_pages()
        assert len(pages) == 1
        assert pages[0]["name"] == "Deep Neural Networks (DNNs)"  # longest variant wins

    def test_merge_case_diff(self) -> None:
        ext = ExtractionSet(
            extractions=[
                Extraction(
                    source="a.pdf",
                    concepts=[Concept(name="Automatic Speech Recognition (ASR)", definition="x")],
                ),
                Extraction(
                    source="b.pdf",
                    concepts=[Concept(name="Automatic speech recognition (ASR)", definition="y")],
                ),
            ]
        )
        pages = ext.concept_pages()
        assert len(pages) == 1
        assert len(pages[0]["sources"]) == 2
        assert len(pages[0]["definitions"]) == 2

    def test_merge_hyphen_variant(self) -> None:
        ext = ExtractionSet(
            extractions=[
                Extraction(
                    source="a.pdf",
                    concepts=[Concept(name="Short-Time Objective Intelligibility (STOI)", definition="x")],
                ),
                Extraction(
                    source="b.pdf",
                    concepts=[Concept(name="Short Time Objective Intelligibility (STOI)", definition="y")],
                ),
            ]
        )
        pages = ext.concept_pages()
        assert len(pages) == 1

    def test_distinct_concepts_not_merged(self) -> None:
        ext = ExtractionSet(
            extractions=[
                Extraction(
                    source="a.pdf",
                    concepts=[Concept(name="Listening Effort", definition="x")],
                ),
                Extraction(
                    source="b.pdf",
                    concepts=[Concept(name="Speech Intelligibility", definition="y")],
                ),
            ]
        )
        pages = ext.concept_pages()
        assert len(pages) == 2


class TestCosineSimilarity:
    """_cosine_similarity helper."""

    def test_identical(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_similar(self) -> None:
        sim = cosine_similarity([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
        assert sim > 0.99

    def test_zero_vector(self) -> None:
        assert cosine_similarity([1.0], [0.0]) == 0.0

    def test_empty_vectors_raise(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            cosine_similarity([], [])


class TestMergeSimilarConcepts:
    """_merge_similar_concepts with mocked embedding provider."""

    def test_skip_when_hash(self) -> None:
        """'hash' model skips embedding merge."""
        concepts = [{"name": "A", "sources": ["a.pdf"], "definitions": ["x"]}]
        result = merge_similar_concepts(concepts, "hash", threshold=0.82, string_threshold=0.70, jaccard_min_token_len=1)
        assert result == concepts  # unchanged

    def test_skip_when_empty_model(self) -> None:
        result = merge_similar_concepts([], "", threshold=0.82, string_threshold=0.70, jaccard_min_token_len=1)
        assert result == []

    def test_provider_failure_returns_unchanged(self) -> None:
        concepts = [{"name": "A", "sources": ["a.pdf"], "definitions": ["x"]}]
        result = merge_similar_concepts(
            concepts,
            "nonexistent-model-xyz",
            threshold=0.82,
            string_threshold=0.70,
            jaccard_min_token_len=1,
        )
        assert result == concepts

    def test_merge_similar_names(self) -> None:
        """Two concepts with semantically similar names get merged."""
        concepts = [
            {"name": "Posteriorgram", "sources": ["a.pdf"], "definitions": ["x"]},
            {"name": "Phoneme Posteriorgram", "sources": ["b.pdf"], "definitions": ["y"]},
        ]
        with patch("kompend.compilation.flows.assembly.make_provider") as mock_factory:
            mock_provider = mock_factory.return_value
            mock_provider.embed.side_effect = [
                [1.0, 0.0, 0.0],  # "Posteriorgram"
                [0.9, 0.1, 0.0],  # "Phoneme Posteriorgram" — very similar
            ]
            result = merge_similar_concepts(
                concepts,
                "any",
                threshold=0.82,
                string_threshold=0.70,
                jaccard_min_token_len=1,
            )

        assert len(result) == 1
        # Longest name wins as canonical
        assert result[0]["name"] == "Phoneme Posteriorgram"

    def test_keep_distinct_concepts(self) -> None:
        """Dissimilar names stay separate."""
        concepts = [
            {"name": "Listening Effort", "sources": ["a.pdf"], "definitions": ["x"]},
            {"name": "Speech Intelligibility", "sources": ["b.pdf"], "definitions": ["y"]},
        ]
        with patch("kompend.compilation.flows.assembly.make_provider") as mock_factory:
            mock_provider = mock_factory.return_value
            mock_provider.embed.side_effect = [
                [1.0, 0.0, 0.0],  # "Listening Effort"
                [0.0, 1.0, 0.0],  # "Speech Intelligibility" — orthogonal
            ]
            result = merge_similar_concepts(
                concepts,
                "any",
                threshold=0.82,
                string_threshold=0.70,
                jaccard_min_token_len=1,
            )

        assert len(result) == 2

    def test_empty_sources_returns_empty(self) -> None:
        assert segment_sources([], 1000) == []

    def test_under_segment_size_returns_single_segment(self) -> None:
        result = segment_sources([("a", "short")], 100)
        assert len(result) == 1
        assert result[0] == [("a", "short")]

    def test_large_source_split_across_segments(self) -> None:
        result = segment_sources([("a", "x" * 100)], 30)
        total_content = "".join(content for segment in result for _, content in segment)
        assert len(total_content) == 100  # no data loss

    def test_multiple_sources_packed_efficiently(self) -> None:
        result = segment_sources(
            [("a", "x" * 20), ("b", "y" * 20)],
            30,
        )
        # Both sources should be in one or two segments, no data lost
        total = sum(len(content) for seg in result for _, content in seg)
        assert total == 40  # no data loss


class TestAgenticSegmentSizing:
    """Agentic-aware segment sizing and the request-fit guard."""

    def test_reserved_tokens_measured_not_hardcoded(self) -> None:
        """The agentic reservation measures the real templates + variables."""
        compiler = self._compiler()
        reserved = _agentic_reserved_tokens(compiler, ["Alpha", "Beta"], ["fact one"])
        # Templates alone are ~1K tokens; known_titles + pinned add more. The
        # reservation must be a few thousand — never zero and never negative.
        assert reserved > 500
        assert reserved < 20000

    def test_reserved_tokens_shrink_segments_with_known_titles(self) -> None:
        """A big known_titles list reserves more context, so segments shrink.

        Content preservation is guaranteed by segment_sources splitting — the
        total content across segments stays identical, never cropped.
        """
        compiler = self._compiler()
        chunks = [
            ResolvedChunk(
                chunk=Chunk(
                    id="c1",
                    text="x" * 100_000,
                    element_types=[ElementType.TEXT],
                    source_elements=[0],
                    heading_context="H",
                    source_file=Path("a.pdf"),
                )
            )
        ]
        empty = prepare_source_segments(compiler, chunks, reserved_tokens=0)
        reserved = _agentic_reserved_tokens(compiler, [f"Title {i}" for i in range(300)], [])
        sized = prepare_source_segments(compiler, chunks, reserved_tokens=reserved)
        empty_chars = sum(len(c) for seg in empty for _, c in seg)
        sized_chars = sum(len(c) for seg in sized for _, c in seg)
        # No data loss either way — the content is fully preserved (the
        # +2 per source is the "\n\n" separator the segmenter adds).
        assert empty_chars == sized_chars
        # The reservation must actually shrink the per-segment budget.
        assert len(sized) >= len(empty)

    def test_request_fit_guard_raises_on_overflow(self) -> None:
        """When segment content + overhead + output exceeds the window, the
        agentic loop must fail loudly instead of letting Ollama stall.
        """
        compiler = self._compiler(context_window=8192, gen_output_max=8192)
        # Tiny window: even a small segment overflows once reserved + output
        # are added. The guard must raise LLMOutputError before any agent call.
        chunks = [
            ResolvedChunk(
                chunk=Chunk(
                    id="c1",
                    text="y" * 40_000,
                    element_types=[ElementType.TEXT],
                    source_elements=[0],
                    heading_context="H",
                    source_file=Path("a.pdf"),
                )
            )
        ]
        reserved = _agentic_reserved_tokens(compiler, [], [])
        segments = prepare_source_segments(compiler, chunks, reserved_tokens=reserved)
        concepts = [{"name": "Concept One", "kind": "concept", "importance": "high", "definition": "def"}]
        with pytest.raises(Exception, match="context window"):
            generate_all_pages_agentic(
                compiler,
                concepts,
                segments,
                [],
                pinned_facts=[],
                known_titles=[],
            )

    def test_correctly_sized_segment_does_not_trip_guard(self) -> None:
        """A segment sized with the label-overhead reservation must pass the
        request-fit guard — the rendered block (content + labels) + reserved
        + output budget fits the window.
        """
        compiler = self._compiler(context_window=16384, gen_output_max=4096)
        # Content sized to fit: 16K window - 4K output - reserved - labels.
        chunks = [
            ResolvedChunk(
                chunk=Chunk(
                    id="c1",
                    text="y" * 20_000,
                    element_types=[ElementType.TEXT],
                    source_elements=[0],
                    heading_context="H",
                    source_file=Path("a.pdf"),
                )
            )
        ]
        reserved = _agentic_reserved_tokens(compiler, [], [])
        segments = prepare_source_segments(compiler, chunks, reserved_tokens=reserved)
        # The guard must not raise for a correctly-sized segment: the
        # rendered content + reserved + output budget fits the window.
        # (We only assert the sizing math here — the guard itself is
        # exercised end-to-end in the overflow test above.)
        rendered = sum(len(f"### {label}\n{content}") for seg in segments for label, content in seg)
        rendered_tokens = rendered / compiler._config.llm.default_chars_per_token
        assert rendered_tokens + reserved + 4096 <= 16384

    def _compiler(self, *, context_window: int = 65536, gen_output_max: int = 16384) -> TwoPhaseCompiler:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(
                default_model="test/model",
                vlm_model="test/vlm",
                timeout=120.0,
            ),
            embedding=EmbeddingConfig(model="hash"),
        )
        config.budget.gen_output_max = gen_output_max
        config.llm.context_window = context_window
        compiler = TwoPhaseCompiler(config=config)
        return compiler


class TestTwoPhaseCompiler:
    """TwoPhaseCompiler behaviour."""

    def test_empty_input(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        compiler = TwoPhaseCompiler(config=config)
        wiki = compiler.compile([])
        assert wiki.metadata["two_phase"] is True
        assert wiki.nodes == []

    def test_extraction_failure_skips_source(self) -> None:
        """When extraction LLM fails for a source, the source is skipped — pipeline continues."""
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm", batch_retry_attempts=0),
            embedding=EmbeddingConfig(model="hash"),
        )
        config.no_cache = True
        compiler = TwoPhaseCompiler(config=config)
        chunks = [
            ResolvedChunk(
                chunk=Chunk(
                    id="c1",
                    text="Some text",
                    element_types=[ElementType.TEXT],
                    source_elements=[0],
                    heading_context="H",
                    source_file=Path("a.pdf"),
                )
            )
        ]
        with patch("kompend.compilation.flows.extraction.try_pydanticai_extraction", side_effect=_validated_side_effect([None])):
            wiki = compiler.compile(chunks)
        # Extraction failed → no concepts → generation has nothing → empty wiki
        assert wiki.metadata.get("generation_skipped") is True
        assert len(wiki.nodes) == 0

    def test_compile_flags_incomplete_on_high_failure_rate(self) -> None:
        """When most batches fail (after retries), compile stops early and flags incomplete."""
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="hash"),
        )
        config.no_cache = True
        compiler = TwoPhaseCompiler(config=config)
        # Enough distinct sources to clear the threshold minimum.
        n_sources = _FAILURE_THRESHOLD_MIN + 1
        chunks = [
            ResolvedChunk(
                chunk=Chunk(
                    id=f"c{i}",
                    text=f"text {i}",
                    element_types=[ElementType.TEXT],
                    source_elements=[0],
                    heading_context="H",
                    source_file=Path(f"src{i}.pdf"),
                )
            )
            for i in range(n_sources)
        ]
        # Every extraction batch fails.
        with patch("kompend.compilation.flows.extraction.try_pydanticai_extraction", return_value=None):
            wiki = compiler.compile(chunks)
        # Threshold fired → flagged incomplete, failures recorded.
        assert wiki.metadata.get("compile_incomplete") is True
        assert wiki.metadata.get("batch_failures", 0) >= _FAILURE_THRESHOLD_MIN
        assert wiki.metadata.get("batch_total", 0) >= _FAILURE_THRESHOLD_MIN

    def test_extract_returns_extractions(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        compiler = TwoPhaseCompiler(config=config)
        chunks = [
            ResolvedChunk(
                chunk=Chunk(
                    id="c1",
                    text="Hello",
                    element_types=[ElementType.TEXT],
                    source_elements=[0],
                    heading_context="H",
                    source_file=Path("a.pdf"),
                )
            )
        ]
        with patch(
            "kompend.compilation.flows.extraction.try_pydanticai_extraction",
            side_effect=_validated_side_effect(
                [
                    {
                        "summary": "x",
                        "entities": [{"name": "kompend", "type": "PROJECT"}],
                        "concepts": [{"name": "attention", "definition": "y"}],
                        "tags": ["ml"],
                    },
                ]
            ),
        ):
            ext = compiler.extract(chunks)
        assert ext.extractions[0].summary == "x"
        assert ext.extractions[0].concepts[0].name == "attention"

    def test_extract_populates_extraction_stats(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        compiler = TwoPhaseCompiler(config=config)
        chunks = [
            ResolvedChunk(
                chunk=Chunk(
                    id="c1",
                    text="Hello world this is a test",
                    element_types=[ElementType.TEXT],
                    source_elements=[0],
                    heading_context="H",
                    source_file=Path("a.pdf"),
                )
            )
        ]
        with patch(
            "kompend.compilation.flows.extraction.try_pydanticai_extraction",
            side_effect=_validated_side_effect(
                [
                    {
                        "summary": "one two three four",
                        "entities": [{"name": "kompend", "type": "PROJECT"}],
                        "concepts": [{"name": "attention", "definition": "y"}],
                        "tags": ["ml"],
                    },
                ]
            ),
        ):
            compiler.extract(chunks)
        stats = compiler._extraction_stats
        assert len(stats) == 1
        s = stats[0]
        assert s.source == "a.pdf"
        assert s.duration_s >= 0.0
        assert s.summary_words == 4
        assert s.summary_chars == len("one two three four")
        assert s.entity_count == 1
        assert s.concept_count == 1
        assert s.tag_count == 1
        assert s.batches == 1

    def test_compile_writes_extraction_stats_to_metadata(self) -> None:
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="test/model", vlm_model="test/vlm"),
            embedding=EmbeddingConfig(model="test/model"),
        )
        config.no_cache = True
        compiler = TwoPhaseCompiler(config=config)
        chunks = [
            ResolvedChunk(
                chunk=Chunk(
                    id="c1",
                    text="Hello world",
                    element_types=[ElementType.TEXT],
                    source_elements=[0],
                    heading_context="H",
                    source_file=Path("a.pdf"),
                )
            )
        ]
        extract_side_effect = _validated_side_effect(
            [
                {
                    "summary": "a summary",
                    "entities": [],
                    "concepts": [{"name": "attention", "definition": "y"}],
                    "tags": [],
                },
            ]
        )
        gen_side_effect = _validated_side_effect(
            [
                {
                    "pages": [
                        {
                            "title": "Attention",
                            "content": (
                                "Attention is the cognitive process of selectively focusing "
                                "on discrete information while ignoring other perceivable input. "
                                "It plays a critical role in learning and memory."
                            ),
                        }
                    ]
                },
            ]
        )

        # The agentic generation path builds a real agent per segment; patch the
        # factory to return a stub whose run() validates the side-effect dict
        # against GenerationOutput (the agentic generation seam).
        def _fake_agent_factory(**kwargs: object) -> object:
            agent = MagicMock()
            agent.result_type = GenerationOutput

            def _run(_deps: object) -> object:
                # Drain the broker so the coverage loop terminates normally
                # (the real agent would call fetch_concepts).
                broker = getattr(_deps, "concept_broker", None)
                if broker is not None:
                    broker.fetch(getattr(_deps, "source_id", ""), limit=10)
                return gen_side_effect(agent, _deps)

            agent.run = _run
            return agent

        with (
            patch("kompend.compilation.flows.extraction.try_pydanticai_extraction", side_effect=extract_side_effect),
            patch("kompend.compilation.flows.generation.make_agentic_generation_agent", side_effect=_fake_agent_factory),
        ):
            wiki = compiler.compile(chunks)
        assert "extraction_stats" in wiki.metadata
        stats = wiki.metadata["extraction_stats"]
        assert len(stats) == 1
        assert stats[0]["source"] == "a.pdf"
        assert stats[0]["summary_words"] == 2


# ===========================================================================
#  N8: Review queue
# ===========================================================================


class TestReviewQueue:
    """ReviewQueue behaviour."""

    def test_enqueue_creates_pending_file(self, tmp_path: Path) -> None:
        queue = ReviewQueue(review_dir=tmp_path / REVIEW_DIR)
        node = WikiNode(id="n1", title="Hello World", content="body", node_type="section")
        queue.enqueue(node)
        assert (tmp_path / REVIEW_DIR / "hello-world.md").exists()
        assert (tmp_path / REVIEW_DIR / "_state.json").exists()

    def test_approve_moves_file(self, tmp_path: Path) -> None:
        queue = ReviewQueue(review_dir=tmp_path / REVIEW_DIR)
        node = WikiNode(id="n1", title="My Page", content="b", node_type="section")
        queue.enqueue(node)
        approved = queue.approve("my-page")
        assert approved
        assert approved[0].status == ReviewStatus.APPROVED
        assert (tmp_path / REVIEW_DIR / "my-page.approved.md").exists()
        assert not (tmp_path / REVIEW_DIR / "my-page.md").exists()

    def test_approve_all(self, tmp_path: Path) -> None:
        queue = ReviewQueue(review_dir=tmp_path / REVIEW_DIR)
        for i in range(3):
            queue.enqueue(WikiNode(id=f"n{i}", title=f"Page {i}", content="x", node_type="section"))
        approved = queue.approve("", approve_all=True)
        assert len(approved) == 3

    def test_reject_with_reason(self, tmp_path: Path) -> None:
        queue = ReviewQueue(review_dir=tmp_path / REVIEW_DIR)
        queue.enqueue(WikiNode(id="n1", title="Bad", content="x", node_type="section"))
        rejected = queue.reject("bad", reason="too long")
        assert rejected[0].reason == "too long"
        assert (tmp_path / REVIEW_DIR / "bad.rejected.md").exists()

    def test_lock_and_unlock(self, tmp_path: Path) -> None:
        queue = ReviewQueue(review_dir=tmp_path / REVIEW_DIR)
        queue.enqueue(WikiNode(id="n1", title="L", content="x", node_type="section"))
        queue.approve("l")
        assert queue.lock("l") is True
        assert queue.is_locked("l") is True

    def test_list_entries(self, tmp_path: Path) -> None:
        queue = ReviewQueue(review_dir=tmp_path / REVIEW_DIR)
        queue.enqueue(WikiNode(id="n1", title="A", content="x", node_type="section"))
        entries = queue.list_entries()
        assert len(entries) == 1
        assert entries[0].status == ReviewStatus.PENDING

    def test_slugify(self) -> None:
        assert slugify("Hello, World!") == "hello-world"
        assert slugify("---") == "untitled"


# ===========================================================================
#  N4: Provenance
# ===========================================================================


class TestExtractCitations:
    """extract_citations parses [source:42-58] markers."""

    def test_single_citation(self) -> None:
        text = "Some claim. [paper.pdf:42-58]"
        out = extract_citations(text)
        assert len(out) == 1
        assert out[0].source == "paper.pdf"
        assert out[0].start == 42
        assert out[0].end == 58

    def test_multiple_citations(self) -> None:
        text = "First. [a.pdf:1-3] Second. [b.pdf:5-9]"
        out = extract_citations(text)
        assert len(out) == 2
        assert [c.source for c in out] == ["a.pdf", "b.pdf"]

    def test_no_citations(self) -> None:
        assert extract_citations("no markers here") == []

    def test_to_marker_round_trip(self) -> None:
        c = Citation(source="x.pdf", start=1, end=5)
        assert extract_citations(c.to_marker())[0] == c


class TestMergeDuplicates:
    """merge_duplicate_citations removes duplicate (source, start, end) keys."""

    def test_dedup_preserves_order(self) -> None:
        c1 = Citation(source="a.pdf", start=1, end=5)
        c2 = Citation(source="a.pdf", start=1, end=5)
        c3 = Citation(source="a.pdf", start=10, end=15)
        out = merge_duplicate_citations([c1, c2, c3])
        assert len(out) == 2
        assert out[0] == c1
        assert out[1] == c3


class TestAnnotateWikiWithProvenance:
    """annotate_wiki_with_provenance persists per-node citations and updates metadata."""

    def test_annotates_and_persists(self, tmp_path: Path) -> None:
        wiki = _wiki(
            nodes=[
                WikiNode(
                    id="n1",
                    title="A",
                    content="Claim. ^[a.pdf:1-3]",
                    node_type="section",
                )
            ]
        )
        out = annotate_wiki_with_provenance(wiki, tmp_path)
        assert out.metadata["provenance_annotated"] is True
        assert out.nodes[0].metadata["citation_count"] == 1
        with open_store(tmp_path / KOMP_DIR / "kompend.db") as conn:
            row = conn.execute(f"SELECT title, citations FROM {PROVENANCE_TABLE} WHERE node_id = ?", ("n1",)).fetchone()
        assert row is not None
        assert row["title"] == "A"
        assert json.loads(row["citations"]) == ["[a.pdf:1-3]"]


# ===========================================================================
#  Q21: Trust / privacy
# ===========================================================================


class TestPrivacyFilter:
    """filter_sensitive / has_sensitive_content behaviour."""

    @pytest.mark.parametrize(
        ("text", "expected_token"),
        [
            ("API key is sk-12345678901234567890abc", "[REDACTED]"),
            ("Email me at user@example.com", "[REDACTED]"),
            ("SSN 123-45-6789 is sensitive", "[REDACTED]"),
            ("AWS AKIAIOSFODNN7EXAMPLE key", "[REDACTED]"),
        ],
    )
    def test_redacts_pattern(self, text: str, expected_token: str) -> None:
        out, n = filter_sensitive(text)
        assert expected_token in out
        assert n >= 1

    def test_clean_text_unchanged(self) -> None:
        out, n = filter_sensitive("nothing to redact here")
        assert n == 0
        assert out == "nothing to redact here"

    def test_has_sensitive_detects(self) -> None:
        assert has_sensitive_content("sk-12345678901234567890")
        assert not has_sensitive_content("plain text")


class TestMarkWikiPrivate:
    """mark_wiki_private writes a private.json index."""

    def test_creates_private_index(self, tmp_path: Path) -> None:
        wiki = _wiki(nodes=[WikiNode(id="n1", title="A", content="x", node_type="section")])
        mark_wiki_private(wiki, tmp_path)
        with open_store(tmp_path / KOMP_DIR / "kompend.db") as conn:
            rows = {r["id"] for r in conn.execute(f"SELECT id FROM {PRIVATE_PAGES_TABLE}")}
        assert "n1" in rows


class TestLifecycleStore:
    """Lifecycle persistence and transitions."""

    def test_load_empty(self, tmp_path: Path) -> None:
        store = LifecycleStore(tmp_path / "life.json")
        assert store.load().entries == {}

    def test_round_trip(self, tmp_path: Path) -> None:
        store = LifecycleStore(tmp_path / "life.json")
        state = LifecycleState()
        store.transition(state, "n1", LIFECYCLE_DRAFT)
        store.save(state)
        loaded = store.load()
        assert loaded.entries["n1"].state == LIFECYCLE_DRAFT

    def test_invalid_state_raises(self, tmp_path: Path) -> None:
        store = LifecycleStore(tmp_path / "life.json")
        state = LifecycleState()
        with pytest.raises(ValueError, match="Invalid lifecycle state"):
            store.transition(state, "n1", "bogus")

    def test_mark_stale(self) -> None:
        old = (datetime.now(UTC) - timedelta(days=200)).isoformat(timespec="seconds")
        state = LifecycleState(
            entries={
                "n1": LifecycleEntry(node_id="n1", state="reviewed", last_transition=old),
            }
        )
        changed = mark_stale(state, stale_after_days=90)
        assert changed == 1
        assert state.entries["n1"].state == LIFECYCLE_STALE


# ===========================================================================
#  Q21: Grounding verification
# ===========================================================================


class TestVerificationUnavailable:
    """verify_node / verify_wiki behaviour when LLM is unavailable."""

    def test_claim_without_citation_unsupported(self, tmp_path: Path) -> None:
        node = WikiNode(
            id="n1",
            title="T",
            content="This claim has no citation.",
            node_type="section",
        )
        result = verify_node(node, tmp_path, agent=self._fake_client())
        assert result[0].verdict == "unsupported"
        assert result[0].confidence < 0.5

    def test_claim_with_missing_source_unsupported(self, tmp_path: Path) -> None:
        node = WikiNode(
            id="n1",
            title="T",
            content="A claim with a missing source citation. ^[missing.pdf:1-3]",
            node_type="section",
        )
        result = verify_node(node, tmp_path, agent=self._fake_client())
        assert result[0].verdict == "unsupported"

    def test_llm_unavailable_marks_unverified(self, tmp_path: Path) -> None:
        """When LLM returns None, claim is 'unverified' — no substring matching."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "a.pdf").write_text(
            "The transformer uses self attention to model sequences.",
            encoding="utf-8",
        )
        node = WikiNode(
            id="n1",
            title="T",
            content="The transformer uses self attention to model sequences. ^[a.pdf:1-2]",
            node_type="section",
        )
        result = verify_node(node, sources, agent=self._fake_client())
        assert result[0].verdict == "unverified"
        assert result[0].confidence == 0.0

    def test_verify_wiki_writes_report(self, tmp_path: Path) -> None:
        wiki = _wiki(
            nodes=[
                WikiNode(id="n1", title="A", content="x" * 50, node_type="section"),
            ]
        )
        result = verify_wiki(wiki, tmp_path, agent=self._fake_client())
        assert result.total >= 1
        with open_store(tmp_path / KOMP_DIR / "kompend.db") as conn:
            row = conn.execute(f"SELECT total FROM {VERIFICATION_TABLE} ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        assert row["total"] == result.total

    @staticmethod
    def _fake_client() -> Mock:
        """Agent whose run returns None (forces fallback)."""
        c = Mock()
        c.run.return_value = None
        c.timeout = 120
        return c


class TestAgenticVerification:
    """Agentic verification: one multi-turn session over all claims."""

    def test_single_session_all_claims(self, tmp_path: Path) -> None:
        wiki = _wiki(
            nodes=[
                WikiNode(id="n1", title="A", content="Claim about transformers. ^[a.pdf:1-2]" * 5, node_type="section"),
                WikiNode(id="n2", title="B", content="Claim about attention. ^[b.pdf:1-2]" * 5, node_type="section"),
            ]
        )
        # _split_claims produces 5 claims per node (4 from repeated citation pattern)
        verdicts = {0: "supported", 5: "contradicted"}
        agent = self._fake_agentic_client(verdicts)
        result = verify_wiki(wiki, tmp_path, agent=agent)
        assert agent.run.call_count == 1
        assert result.total == 10

    def test_agentic_failure_returns_unverified(self, tmp_path: Path) -> None:
        wiki = _wiki(nodes=[WikiNode(id="n1", title="A", content="Some claim text. ^[a.pdf:1-2]" * 5, node_type="section")])
        agent = MagicMock(spec=AgenticVerificationAgent)
        agent.run.side_effect = RuntimeError("LLM failed")
        result = verify_wiki(wiki, tmp_path, agent=agent)
        assert result.total >= 1
        assert all(c.verdict == "unverified" for c in result.claims)

    def _fake_agentic_client(self, verdicts: dict[int, str]) -> MagicMock:
        agent = MagicMock(spec=AgenticVerificationAgent)

        def _run(deps: VerificationDeps) -> None:
            for cid, v in verdicts.items():
                deps.results.append(VerificationVerdict(claim_id=cid, verdict=v, evidence="test evidence", sources=[]))

        agent.run.side_effect = _run
        return agent


# ── helpers ─────────────────────────────────────────────────────────────


def _wiki(nodes: list[WikiNode] | None = None) -> CompiledWiki:
    return CompiledWiki(
        nodes=nodes or [],
        edges=[],
        toc=[{"title": n.title, "id": n.id, "level": 1} for n in nodes or []],
        metadata={},
    )


def _validated_side_effect(dicts: list) -> Callable:
    """Mock agent helper that validates each dict via the agent's result_type.

    Each call returns the validated model; a ``None`` entry yields ``None``
    (a failed batch). Mirrors the agent helper contract so tests can supply
    raw LLM-output dicts without constructing models by hand.
    """
    it = iter(dicts)

    def _impl(*args: object, **_kwargs: object) -> object | None:
        data = next(it)
        if data is None:
            return None
        agent_obj = args[0]
        rt = getattr(agent_obj, "result_type", None)
        return rt.model_validate(data) if rt else None

    return _impl
