"""Tests for ExtractionMemory — structured concept/entity/tag digest."""

from __future__ import annotations

from kompend.compilation._two_phase_core import Extraction
from kompend.compilation.extraction_memory import ExtractionMemory
from kompend.compilation.schema import Concept, Entity

# ── ingest ────────────────────────────────────────────────────────────────


def test_ingest_adds_concepts() -> None:
    memory = ExtractionMemory()
    ext = _extraction(
        concepts=[Concept(name="foo", definition="bar")],
    )
    memory.ingest(ext)
    assert "foo" in memory.concepts
    assert memory.concepts["foo"].definition == "bar"
    assert memory.concepts["foo"].ref_count == 1


def test_ingest_increments_ref_count_for_duplicate() -> None:
    memory = ExtractionMemory()
    memory.ingest(_extraction(concepts=[Concept(name="foo", definition="v1")]))
    memory.ingest(_extraction(concepts=[Concept(name="foo", definition="v2")]))
    assert memory.concepts["foo"].ref_count == 2
    # Last definition wins
    assert memory.concepts["foo"].definition == "v2"


def test_ingest_adds_entities() -> None:
    memory = ExtractionMemory()
    memory.ingest(
        _extraction(
            entities=[Entity(name="Alice", type="person")],
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
    memory.ingest(_extraction(entities=[Entity(name="Bob", type="person", context="ctx")]))
    memory.ingest(_extraction(entities=[Entity(name="Bob", type="person", context="ctx2")]))
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
            concepts=[Concept(name="ai", definition="artificial intelligence"), Concept(name="ml", definition="machine learning")],
            tags=["tech"],
        )
    )
    summary = memory.summarize()
    assert "ai" in summary
    assert "ml" in summary
    assert "tech" in summary


def test_summarize_keeps_everything() -> None:
    """The digest is handed to the model in full - no crop, no truncation."""
    memory = ExtractionMemory()
    definitions = [f"definition-{i}-" + "x" * 200 for i in range(10)]
    for i, definition in enumerate(definitions):
        memory.ingest(_extraction(concepts=[Concept(name=f"concept{i}", definition=definition)], tags=[f"tag{i}"]))

    digest = memory.summarize()

    for definition in definitions:
        assert definition in digest
    assert "truncated" not in digest


# ── to_extraction ─────────────────────────────────────────────────────────


def test_to_extraction_returns_valid_extraction() -> None:
    memory = ExtractionMemory()
    memory.ingest(
        _extraction(
            summary="test summary",
            concepts=[Concept(name="foo", definition="bar")],
            entities=[Entity(name="Alice", type="person", context="ctx")],
            tags=["test"],
        )
    )
    result = memory.to_extraction("test_source")
    assert result.source == "test_source"
    assert "foo" in [c.name for c in result.concepts]
    assert "Alice" in [e.name for e in result.entities]
    assert "test" in result.tags


def test_to_extraction_empty_memory_returns_empty() -> None:
    memory = ExtractionMemory()
    result = memory.to_extraction("empty")
    assert result.source == "empty"
    assert result.summary == ""
    assert result.concepts == []
    assert result.entities == []
    assert result.tags == []


def _extraction(
    *,
    summary: str = "",
    entities: list[Entity] | None = None,
    concepts: list[Concept] | None = None,
    tags: list[str] | None = None,
) -> Extraction:
    return Extraction(
        source="test",
        summary=summary,
        entities=entities or [],
        concepts=concepts or [],
        tags=tags or [],
    )
