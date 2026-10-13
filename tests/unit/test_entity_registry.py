"""Tests for entity_registry.py — programmatic entity extraction and dedup."""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import (
    Chunk,
    ChunkNode,
    ChunkTree,
    ElementType,
    EquationVariable,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)
from kompend.core.resolution.entity_registry import (
    EntityLocation,
    EntityRecord,
    EntityRegistry,
)


class TestEntityRegistry:
    def test_register_new_entity(self) -> None:
        reg = EntityRegistry()
        rec = reg.register("SNR", definition="Signal-to-Noise Ratio")
        assert rec.name == "SNR"
        assert rec.canonical_name == "SNR"
        assert rec.definition == "Signal-to-Noise Ratio"
        assert reg.entity_count == 1

    def test_case_insensitive_query(self) -> None:
        reg = EntityRegistry()
        reg.register("SNR", definition="Signal-to-Noise Ratio")
        assert reg.query("snr") is not None
        assert reg.query("SNR") is not None
        assert reg.query("Snr") is not None
        assert reg.query("X") is None

    def test_acronym_dedup_snr_to_full(self) -> None:
        """Registering SNR as alias of Signal-to-Noise Ratio → single record."""
        reg = EntityRegistry()
        reg.register("Signal-to-Noise Ratio", aliases=["SNR"])
        assert reg.entity_count == 1
        rec = reg.query("SNR")
        assert rec is not None
        assert rec.canonical_name == "Signal-to-Noise Ratio"

    def test_acronym_dedup_reverse(self) -> None:
        """Registering Signal-to-Noise Ratio as alias of SNR → single record."""
        reg = EntityRegistry()
        reg.register("SNR", aliases=["Signal-to-Noise Ratio"])
        assert reg.entity_count == 1
        assert reg.query("Signal-to-Noise Ratio") is not None
        rec = reg.query("signal-to-noise ratio")
        assert rec is not None
        assert rec.canonical_name == "SNR"

    def test_acronym_dedup_separate_registrations(self) -> None:
        """Two separate .register() calls with acronym relationship → merged."""
        reg = EntityRegistry()
        reg.register("Signal-to-Noise Ratio", aliases=["SNR"])
        reg.register("SNR", definition="Signal power to noise power ratio")
        assert reg.entity_count == 1
        rec = reg.query("SNR")
        assert rec is not None
        assert rec.definition == "Signal power to noise power ratio"
        assert "Signal-to-Noise Ratio" in rec.aliases or rec.canonical_name == "Signal-to-Noise Ratio"

    def test_definition_accumulated(self) -> None:
        reg = EntityRegistry()
        reg.register("SNR", definition="Signal to Noise Ratio")
        reg.register("SNR", definition="Signal-to-Noise Ratio")
        assert reg.entity_count == 1
        rec = reg.query("SNR")
        assert rec is not None
        assert len(rec.all_definitions) == 2

    def test_location_tracked(self) -> None:
        loc = EntityLocation(source_file="doc.md", heading_chain=("Section 1",))
        reg = EntityRegistry()
        reg.register("ABC", location=loc)
        assert reg.query("ABC") is not None
        rec = reg.query("ABC")
        assert rec is not None
        assert rec.locations == [loc]

    def test_text_extraction_acronyms(self) -> None:
        reg = EntityRegistry()
        loc = EntityLocation(source_file="doc.md", heading_chain=())
        reg._extract_from_text("The Signal-to-Noise Ratio (SNR) is critical.", loc)
        assert reg.entity_count >= 1
        assert reg.query("SNR") is not None

    def test_text_extraction_reverse_acronym(self) -> None:
        reg = EntityRegistry()
        loc = EntityLocation(source_file="doc.md", heading_chain=())
        reg._extract_from_text("The SNR (Signal-to-Noise Ratio) is important.", loc)
        assert reg.entity_count >= 1
        rec = reg.query("SNR")
        assert rec is not None

    def test_text_extraction_definition(self) -> None:
        reg = EntityRegistry()
        loc = EntityLocation(source_file="doc.md", heading_chain=())
        reg._extract_from_text(
            "Precedence Effect is defined as the auditory illusion where the first arriving sound dominates localization.",
            loc,
        )
        assert reg.query("Precedence Effect") is not None

    def test_text_extraction_means(self) -> None:
        reg = EntityRegistry()
        loc = EntityLocation(source_file="doc.md", heading_chain=())
        reg._extract_from_text("Psychoacoustics means the study of how humans perceive sound.", loc)
        assert reg.query("Psychoacoustics") is not None

    def test_text_extraction_also_known_as(self) -> None:
        reg = EntityRegistry()
        loc = EntityLocation(source_file="doc.md", heading_chain=())
        reg._extract_from_text("Haas effect (also known as Precedence Effect) is important.", loc)
        rec = reg.query("Haas effect")
        assert rec is not None
        assert "Precedence Effect" in rec.aliases

    def test_structured_equation_variables(self) -> None:
        reg = EntityRegistry()
        loc = EntityLocation(source_file="doc.md", heading_chain=())
        eq = StructuredEquation(
            latex=r"E = mc^2",
            variables=[
                EquationVariable(name="E", definition="energy", units="J"),
                EquationVariable(name="m", definition="mass", units="kg"),
            ],
        )
        reg._extract_from_structured(eq, loc)
        assert reg.query("E") is not None
        assert reg.query("m") is not None
        assert reg.query("E") is not None
        eq_def = reg.query("E")
        assert eq_def is not None
        assert eq_def.definition == "energy"

    def test_structured_table_columns(self) -> None:
        reg = EntityRegistry()
        loc = EntityLocation(source_file="doc.md", heading_chain=())
        tbl = StructuredTable(
            columns=["Frequency", "Threshold dB"],
            rows=[
                {"Frequency": "1000", "Threshold dB": "20"},
            ],
        )
        reg._extract_from_structured(tbl, loc)
        assert reg.query("Frequency") is not None
        assert reg.query("Threshold dB") is not None

    def test_scan_tree_integration(self) -> None:
        reg = EntityRegistry()
        nodes = [
            _text_node("t1", "root", "Bark scale (also known as Critical-Band Rate) is important."),
            _text_node(
                "t2",
                "root",
                "SNR is defined as Signal-to-Noise Ratio.",
                heading=("Section 1",),
            ),
        ]
        tree = _leaf_tree(nodes)
        reg.scan_tree(tree)

        bark = reg.query("Bark scale")
        assert bark is not None
        assert "Critical-Band Rate" in bark.aliases
        snr = reg.query("SNR")
        assert snr is not None

    def test_find_conflicts_detects_divergent_definitions(self) -> None:
        reg = EntityRegistry()
        reg.register("SNR", definition="Signal to Noise Ratio")
        reg.register("SNR", definition="Signal-to-Noise power Ratio")
        conflicts = reg.find_conflicts()
        assert len(conflicts) >= 1
        assert conflicts[0].issue_type == "conflict"
        assert "SNR" in conflicts[0].entity

    def test_find_conflicts_no_issues_for_identical_definitions(self) -> None:
        reg = EntityRegistry()
        reg.register("SNR", definition="Signal to Noise Ratio")
        reg.register("SNR", definition="Signal to Noise Ratio")
        conflicts = reg.find_conflicts()
        assert len(conflicts) == 0

    def test_find_duplicates_detects_shared_names(self) -> None:
        reg = EntityRegistry()
        reg.register("Precedence Effect", definition="Auditory localization illusion")
        reg.register("Haas Effect", aliases=["Precedence Effect"])
        dups = reg.find_duplicates()
        assert len(dups) >= 1
        assert dups[0].issue_type == "duplicate"

    def test_find_duplicates_no_issues_when_no_overlap(self) -> None:
        reg = EntityRegistry()
        reg.register("SNR", definition="Signal to Noise Ratio")
        reg.register("ABC", definition="Alphabet")
        dups = reg.find_duplicates()
        assert len(dups) == 0

    def test_entity_record_all_names(self) -> None:
        rec = EntityRecord(
            name="SNR",
            canonical_name="Signal-to-Noise Ratio",
            aliases=["S/N Ratio", "Signal/Noise Ratio"],
        )
        names = rec.all_names
        assert "Signal-to-Noise Ratio" in names
        assert "SNR" in names
        assert "S/N Ratio" in names
        assert "Signal/Noise Ratio" in names
        assert len(names) == 4  # no duplicates

    def test_entity_record_duplicate_names_deduped(self) -> None:
        rec = EntityRecord(
            name="snr",
            canonical_name="SNR",
            aliases=["snr"],
        )
        names = rec.all_names
        assert len(names) == 1  # case-insensitive dedup: SNR = snr


def _text_node(nid: str, parent: str, text: str, heading: tuple[str, ...] = ()) -> ChunkNode:
    return ChunkNode(
        id=nid,
        parent_id=parent,
        depth=len(heading) + 1,
        heading_chain=heading,
        chunk=_make_chunk(nid, text),
        token_count=max(1, len(text) // 4),
    )


def _make_chunk(cid: str, text: str, source: str = "doc.md") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[],
        source_file=Path(source),
        token_count=max(1, len(text) // 4),
    )


def _structured_node(
    nid: str, parent: str, structured: StructuredTable | StructuredEquation | StructuredFigure | None, heading: tuple[str, ...] = ()
) -> ChunkNode:
    return ChunkNode(
        id=nid,
        parent_id=parent,
        depth=len(heading) + 1,
        heading_chain=heading,
        structured=structured,
        token_count=1,
    )


def _leaf_tree(nodes: list[ChunkNode], source: str = "doc.md") -> ChunkTree:
    root = ChunkNode(id="root", depth=0, children_ids=[n.id for n in nodes])
    nodes.insert(0, root)
    return _make_tree(nodes, source)


def _make_tree(nodes: list[ChunkNode], source: str = "doc.md") -> ChunkTree:
    nodes_by_id = {n.id: n for n in nodes}
    return ChunkTree(
        root_id=nodes[0].id,
        nodes=nodes_by_id,
        source_file=Path(source),
    )
