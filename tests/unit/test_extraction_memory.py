"""Tests for ExtractionMemory — structured concept/entity/tag digest."""

from __future__ import annotations

from knox.compilation._two_phase_core import Extraction
from knox.compilation.extraction_memory import ExtractionMemory


def _extraction(
    *,
    summary: str = "",
    entities: list[dict] | None = None,
    concepts: list[dict] | None = None,
    tags: list[str] | None = None,
) -> Extraction:
    return Extraction(
        source="test",
        summary=summary,
        entities=entities or [],
        concepts=concepts or [],
        tags=tags or [],
    )


# ── ingest ────────────────────────────────────────────────────────────────


def test_ingest_adds_concepts() -> None:
    memory = ExtractionMemory()
    ext = _extraction(
        concepts=[{"name": "foo", "definition": "bar"}],
    )
    memory.ingest(ext)
    assert "foo" in memory.concepts
    assert memory.concepts["foo"].definition == "bar"
    assert memory.concepts["foo"].ref_count == 1


def test_ingest_increments_ref_count_for_duplicate() -> None:
    memory = ExtractionMemory()
    memory.ingest(_extraction(concepts=[{"name": "foo", "definition": "v1"}]))
    memory.ingest(_extraction(concepts=[{"name": "foo", "definition": "v2"}]))
    assert memory.concepts["foo"].ref_count == 2
    # Last definition wins
    assert memory.concepts["foo"].definition == "v2"


def test_ingest_adds_entities() -> None:
    memory = ExtractionMemory()
    memory.ingest(
        _extraction(
            entities=[{"name": "Alice", "type": "person"}],
        )
    )
    assert "alice" in memory.entities  # normalised key
    assert memory.entities["alice"].entity_type == "person"


def test_ingest_aggregates_tags() -> None:
    memory = ExtractionMemory()
    memory.ingest(_extraction(tags=["ai", "ml"]))
    memory.ingest(_extraction(tags=["ml", "nlp"]))
    assert memory.tags == {"ai", "ml", "nlp"}


def test_ingest_appends_section_summary() -> None:
    memory = ExtractionMemory()
    memory.ingest(_extraction(summary="first section"))
    memory.ingest(_extraction(summary="second section"))
    assert len(memory.section_summaries) == 2


def test_ingest_deduplicates_entities_by_name() -> None:
    memory = ExtractionMemory()
    memory.ingest(_extraction(entities=[{"name": "Bob", "type": "person", "context": "ctx"}]))
    memory.ingest(_extraction(entities=[{"name": "Bob", "type": "person", "context": "ctx2"}]))
    assert "bob" in memory.entities
    assert memory.entities["bob"].ref_count == 2


# ── summarize ─────────────────────────────────────────────────────────────


def test_summarize_empty_returns_empty() -> None:
    memory = ExtractionMemory()
    assert memory.summarize() == ""


def test_summarize_includes_top_concepts() -> None:
    memory = ExtractionMemory()
    memory.ingest(
        _extraction(
            concepts=[
                {"name": "ai", "definition": "artificial intelligence"},
                {"name": "ml", "definition": "machine learning"},
            ],
            tags=["tech"],
        )
    )
    summary = memory.summarize()
    assert "ai" in summary
    assert "ml" in summary
    assert "tech" in summary


def test_summarize_respects_max_tokens() -> None:
    memory = ExtractionMemory()
    for i in range(10):
        memory.ingest(
            _extraction(
                concepts=[{"name": f"concept{i}", "definition": "x" * 200}],
                tags=[f"tag{i}"],
            )
        )
    unlimited = memory.summarize()
    limited = memory.summarize(max_tokens=20)
    assert len(limited) < len(unlimited)


# ── compact ───────────────────────────────────────────────────────────────


def test_compact_keeps_last_three_section_summaries() -> None:
    """Compact keeps at most 3 most recent section summaries."""
    memory = ExtractionMemory()
    for i in range(5):
        memory.ingest(
            _extraction(
                concepts=[{"name": f"c{i}", "definition": "x"}],
                summary=f"section {i}",
            )
        )
    assert len(memory.section_summaries) == 5
    memory.compact()
    assert len(memory.section_summaries) == 3
    # Last three (most recent) are kept
    assert memory.section_summaries == ["section 2", "section 3", "section 4"]


def test_compact_drops_singleton_concepts() -> None:
    """Compact drops concepts with ref_count == 1."""
    memory = ExtractionMemory()
    memory.ingest(_extraction(concepts=[{"name": "freq", "definition": "x"}]))
    memory.ingest(_extraction(concepts=[{"name": "freq", "definition": "x"}]))
    memory.ingest(_extraction(concepts=[{"name": "rare", "definition": "y"}]))
    memory.compact()
    assert "freq" in memory.concepts
    assert "rare" not in memory.concepts


def test_compact_reduces_to_title_only() -> None:
    """Compact clears definitions and context on remaining digests."""
    memory = ExtractionMemory()
    memory.ingest(
        _extraction(
            concepts=[{"name": "keep", "definition": "definition text"}],
            entities=[{"name": "entity", "type": "person", "context": "context text"}],
        )
    )
    memory.ingest(
        _extraction(
            concepts=[{"name": "keep", "definition": "definition text"}],
            entities=[{"name": "entity", "type": "person", "context": "context text"}],
        )
    )
    memory.compact()
    assert memory.concepts["keep"].definition == ""
    assert memory.entities["entity"].context == ""


# ── to_extraction ─────────────────────────────────────────────────────────


def test_to_extraction_returns_valid_extraction() -> None:
    memory = ExtractionMemory()
    memory.ingest(
        _extraction(
            summary="test summary",
            concepts=[{"name": "foo", "definition": "bar"}],
            entities=[{"name": "Alice", "type": "person", "context": "ctx"}],
            tags=["test"],
        )
    )
    result = memory.to_extraction("test_source")
    assert result.source == "test_source"
    assert "foo" in [c["name"] for c in result.concepts]
    assert "Alice" in [e["name"] for e in result.entities]
    assert "test" in result.tags


def test_to_extraction_empty_memory_returns_empty() -> None:
    memory = ExtractionMemory()
    result = memory.to_extraction("empty")
    assert result.source == "empty"
    assert result.summary == ""
    assert result.concepts == []
    assert result.entities == []
    assert result.tags == []


# ── token_estimate ────────────────────────────────────────────────────────


def test_token_estimate_grows_with_content() -> None:
    memory = ExtractionMemory()
    assert memory._estimate_tokens() == 0
    memory.ingest(
        _extraction(
            concepts=[{"name": "foo", "definition": "bar"}],
        )
    )
    assert memory._estimate_tokens() > 0
