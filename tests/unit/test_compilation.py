"""Tests for KNOX Tier 4 compilers — GraphCompiler and WiCERCompiler."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knox.agents.wicer import WicerDiagnoseAgent, WicerEvaluateAgent, WicerProbeAgent
from knox.compilation import GraphCompiler, WiCERCompiler
from knox.compilation._two_phase_core import (
    _FAILURE_THRESHOLD_MIN,
    _SLUG_MAX_LENGTH,
    BatchFailureTracker,
    Extraction,
    ExtractionSet,
    _merge_node,
    slugify,
)
from knox.compilation.base import BaseCompiler
from knox.compilation.schema import (
    Concept,
    Entity,
    LLMOutputError,
    WicerDiagnoseOutput,
    WicerEvaluateOutput,
    WicerProbeOutput,
)
from knox.compilation.two_phase import TwoPhaseCompiler
from knox.core.config import EmbeddingConfig, LLMConfig
from knox.core.models import Chunk, ChunkTree, CompiledWiki, ElementType, ResolvedChunk, WikiNode
from knox.core.pipeline.config import PipelineConfig
from knox.llm.context_window import ContextSource, ContextWindow

# ===========================================================================
#  GraphCompiler
# ===========================================================================


class TestGraphCompiler:
    """GraphCompiler behaviour."""

    def test_empty_input_returns_empty_wiki(self) -> None:
        compiler = GraphCompiler("model")
        wiki = compiler.compile([])
        assert isinstance(wiki, CompiledWiki)
        assert wiki.nodes == []
        assert wiki.edges == []
        assert wiki.toc == []
        assert wiki.metadata["chunk_count"] == 0
        assert wiki.metadata["source_count"] == 0
        assert wiki.metadata["source_names"] == []

    def test_heading_groups_form_nodes(self) -> None:
        chunks = _make_chunks(count=4, heading="Section A")
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        # All chunks under same heading → one node
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].title == "Section A"
        assert len(wiki.nodes[0].source_chunks or []) == 4  # all 4 merged

    def test_multiple_headings_multiple_nodes(self) -> None:
        chunks = _make_chunks(count=4, heading="Section A")
        # Add one chunk with a different heading
        c = _chunk(
            text="Different section.",
            heading="Section B",
            chunk_id="chunk_extra",
        )
        chunks.append(ResolvedChunk(chunk=c))
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 2
        titles = {n.title for n in wiki.nodes}
        assert titles == {"Section A", "Section B"}

    def test_root_heading_becomes_overview(self) -> None:
        c = _chunk(
            text="No heading context.",
            heading="",
            chunk_id="chunk_root",
        )
        chunks = [ResolvedChunk(chunk=c)]
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].title == "Overview"

    def test_content_is_merged(self) -> None:
        chunks = _make_chunks(count=3, heading="Intro")
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 1
        # All three texts merged with double newlines
        assert "Content for chunk 0" in wiki.nodes[0].content
        assert "Content for chunk 1" in wiki.nodes[0].content
        assert "Content for chunk 2" in wiki.nodes[0].content

    def test_merge_node_dict_valued_provenance_no_crash(self) -> None:
        """Regression: _merge_node must not crash when provenance/related hold
        dicts (malformed LLM output). Previously raised 'unhashable type: dict'.
        """
        a = WikiNode(
            id="neural-net",
            title="Neural Net",
            content="c1",
            node_type="concept",
            source_chunks=[],
            metadata={"provenance": [{"chunk": "c1", "quote": "x"}], "related": [{"id": "r1"}]},
        )
        b = WikiNode(
            id="neural-net",
            title="Neural Net",
            content="c2",
            node_type="concept",
            source_chunks=[],
            metadata={"provenance": [{"chunk": "c2"}, {"chunk": "c1", "quote": "x"}]},
        )
        merged = _merge_node(a, b)  # must not raise
        # dedup: c1 in both -> 2 unique
        assert len(merged.metadata["provenance"]) == 2
        assert len(merged.metadata["related"]) == 1

    def test_entities_create_cross_section_edges(self) -> None:
        chunks = _make_diverse_chunks()
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        # All sections mention KNOX and WiCER — expect edges between them
        assert len(wiki.edges) > 0
        # Each edge should have source, target, type, entity
        for edge in wiki.edges:
            assert "source" in edge
            assert "target" in edge
            assert edge["type"] == "entity_reference"
            assert "entity" in edge

    def test_toc_from_headings(self) -> None:
        chunks = _make_diverse_chunks()
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        assert len(wiki.toc) == len(wiki.nodes)
        for entry in wiki.toc:
            assert "title" in entry
            assert "id" in entry
            assert "level" in entry

    def test_metadata_contains_counts(self) -> None:
        chunks = _make_chunks(count=5)
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        assert wiki.metadata["chunk_count"] == 5
        assert wiki.metadata["node_count"] == 1  # all same heading
        assert "model" in wiki.metadata

    def test_source_count_and_names_from_chunks(self) -> None:
        chunks = [
            ResolvedChunk(chunk=_chunk(text="A", heading="Sec", chunk_id="c1", source_file=Path("paper1.pdf"))),
            ResolvedChunk(chunk=_chunk(text="B", heading="Sec", chunk_id="c2", source_file=Path("paper1.pdf"))),
            ResolvedChunk(chunk=_chunk(text="C", heading="Sec", chunk_id="c3", source_file=Path("paper2.pdf"))),
        ]
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        assert wiki.metadata["source_count"] == 2
        assert wiki.metadata["source_names"] == ["paper1.pdf", "paper2.pdf"]

    def test_no_entities_no_edges(self) -> None:
        c = _chunk(text="Simple text.", heading="Docs")
        rc = ResolvedChunk(chunk=c, entities=[])
        compiler = GraphCompiler("model")
        wiki = compiler.compile([rc])
        assert wiki.edges == []

    def test_typed_relations_create_edges(self) -> None:
        c1 = _chunk(text="KNOX overview.", heading="Overview", chunk_id="c1")
        c2 = _chunk(text="WiCER details.", heading="Methods", chunk_id="c2")
        rc1 = ResolvedChunk(
            chunk=c1,
            entities=[{"name": "KNOX", "type": "project"}],
            relations=[{"source": "KNOX", "target": "WiCER", "type": "uses", "confidence": 0.85}],
        )
        rc2 = ResolvedChunk(
            chunk=c2,
            entities=[{"name": "WiCER", "type": "method"}],
            relations=[],
        )
        compiler = GraphCompiler("model")
        wiki = compiler.compile([rc1, rc2])
        typed = [e for e in wiki.edges if e.get("type") == "uses"]
        assert len(typed) == 1
        assert typed[0]["source"] != typed[0]["target"]
        assert typed[0]["confidence"] == pytest.approx(0.85)

    def test_typed_relations_skip_missing_entities(self) -> None:
        c1 = _chunk(text="KNOX overview.", heading="Overview", chunk_id="c1")
        rc1 = ResolvedChunk(
            chunk=c1,
            entities=[{"name": "KNOX", "type": "project"}],
            relations=[{"source": "KNOX", "target": "Missing", "type": "uses", "confidence": 0.9}],
        )
        compiler = GraphCompiler("model")
        wiki = compiler.compile([rc1])
        assert wiki.edges == []

    def test_overview_kind_inferred(self) -> None:
        c = _chunk(text="Overview content.", heading="", chunk_id="c1")
        rc = ResolvedChunk(chunk=c)
        wiki = GraphCompiler("model").compile([rc])
        assert wiki.nodes[0].node_type == "overview"
        assert wiki.nodes[0].metadata["frontmatter"]["kind"] == "overview"

    def test_comparison_kind_inferred(self) -> None:
        c = _chunk(text="Comparison text.", heading="Comparison of Methods", chunk_id="c1")
        rc = ResolvedChunk(chunk=c)
        wiki = GraphCompiler("model").compile([rc])
        assert wiki.nodes[0].node_type == "comparison"

    def test_entity_kind_inferred_when_title_matches_entity(self) -> None:
        c = _chunk(text="KNOX details.", heading="KNOX", chunk_id="c1")
        rc = ResolvedChunk(chunk=c, entities=[{"name": "KNOX", "type": "project"}])
        wiki = GraphCompiler("model").compile([rc])
        assert wiki.nodes[0].node_type == "entity"

    def test_concept_kind_default(self) -> None:
        c = _chunk(text="Some concept.", heading="Signal Theory", chunk_id="c1")
        rc = ResolvedChunk(chunk=c, entities=[{"name": "KNOX", "type": "project"}])
        wiki = GraphCompiler("model").compile([rc])
        assert wiki.nodes[0].node_type == "concept"

    def test_kind_distribution_in_metadata(self) -> None:
        chunks = [
            ResolvedChunk(chunk=_chunk(text="Overview.", heading="", chunk_id="c1")),
            ResolvedChunk(chunk=_chunk(text="Comparison.", heading="Comparison", chunk_id="c2")),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        assert wiki.metadata["kinds"] == {"overview": 1, "comparison": 1}

    def test_duplicate_titles_disambiguated(self) -> None:
        """knox-wgt.5: headings differing only in case get disambiguated."""
        chunks = [
            ResolvedChunk(chunk=_chunk(text="Refs A.", heading="References", chunk_id="c1", source_file=Path("paper-a.pdf"))),
            ResolvedChunk(chunk=_chunk(text="Refs B.", heading="REFERENCES", chunk_id="c2", source_file=Path("paper-b.pdf"))),
        ]
        compiler = GraphCompiler("model")
        wiki = compiler.compile(chunks)
        titles = {n.title for n in wiki.nodes}
        assert len(titles) == 2, f"Expected 2 unique titles, got {titles}"
        # Both should have source context in the title
        assert any("paper-a" in t for t in titles)
        assert any("paper-b" in t for t in titles)

    def test_duplicate_titles_same_source_get_counter(self) -> None:
        """Duplicates from the same source get a numeric counter."""
        chunks = [
            ResolvedChunk(chunk=_chunk(text="Refs.", heading="References", chunk_id="c1", source_file=Path("paper.pdf"))),
            ResolvedChunk(chunk=_chunk(text="More refs.", heading="REFERENCES", chunk_id="c2", source_file=Path("paper.pdf"))),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        titles = {n.title for n in wiki.nodes}
        assert len(titles) == 2

    def test_unique_titles_not_modified(self) -> None:
        """Nodes with unique titles are not renamed."""
        chunks = [
            ResolvedChunk(chunk=_chunk(text="Intro.", heading="Introduction", chunk_id="c1")),
            ResolvedChunk(chunk=_chunk(text="Methods.", heading="Methods", chunk_id="c2")),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        titles = {n.title for n in wiki.nodes}
        assert titles == {"Introduction", "Methods"}

    def test_disambiguated_titles_in_toc(self) -> None:
        """TOC entries reflect the disambiguated titles."""
        chunks = [
            ResolvedChunk(chunk=_chunk(text="Refs A.", heading="References", chunk_id="c1", source_file=Path("paper-a.pdf"))),
            ResolvedChunk(chunk=_chunk(text="Refs B.", heading="REFERENCES", chunk_id="c2", source_file=Path("paper-b.pdf"))),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        toc_titles = {entry["title"] for entry in wiki.toc}
        node_titles = {n.title for n in wiki.nodes}
        assert toc_titles == node_titles


class TestCrossDocumentSeparation:
    """Ensure generic headings from different sources don't mix."""

    def test_same_heading_different_sources_creates_separate_nodes(self) -> None:
        """Two papers with 'Introduction' headings produce two nodes."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="Introduction from paper A.",
                    heading="Introduction",
                    chunk_id="c1",
                    source_file=Path("paper-a.pdf"),
                ),
            ),
            ResolvedChunk(
                chunk=_chunk(
                    text="Introduction from paper B.",
                    heading="Introduction",
                    chunk_id="c2",
                    source_file=Path("paper-b.pdf"),
                ),
            ),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        assert len(wiki.nodes) == 2
        contents = {n.content for n in wiki.nodes}
        assert "Introduction from paper A." in contents
        assert "Introduction from paper B." in contents

    def test_same_heading_same_source_keeps_single_node(self) -> None:
        """Two chunks from the same source and heading merge into one node."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="Part one.",
                    heading="Methods",
                    chunk_id="c1",
                    source_file=Path("paper-a.pdf"),
                ),
            ),
            ResolvedChunk(
                chunk=_chunk(
                    text="Part two.",
                    heading="Methods",
                    chunk_id="c2",
                    source_file=Path("paper-a.pdf"),
                ),
            ),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        assert len(wiki.nodes) == 1
        assert "Part one." in wiki.nodes[0].content
        assert "Part two." in wiki.nodes[0].content

    def test_node_metadata_stores_source_file(self) -> None:
        """Each node records which source file it came from."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="Content.",
                    heading="Results",
                    chunk_id="c1",
                    source_file=Path("paper-x.pdf"),
                ),
            ),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        assert wiki.nodes[0].metadata["source_file"] == str(Path("paper-x.pdf"))

    def test_no_source_file_chunks_group_together(self) -> None:
        """Chunks without source_file still group by heading alone."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(text="Text A.", heading="Abstract", chunk_id="c1"),
            ),
            ResolvedChunk(
                chunk=_chunk(text="Text B.", heading="Abstract", chunk_id="c2"),
            ),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        assert len(wiki.nodes) == 1


class TestSiblingEdges:
    """Cross-linking via same-source sibling edges (knox-wgt.8)."""

    def test_multi_section_source_gets_sibling_edges(self) -> None:
        """Consecutive same-source nodes are linked bidirectionally."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text=f"Section {i} body content here.",
                    heading=f"Section {i}",
                    chunk_id=f"c{i}",
                    source_file=Path("paper.pdf"),
                ),
            )
            for i in range(1, 4)
        ]
        wiki = GraphCompiler("model").compile(chunks)
        assert len(wiki.nodes) == 3
        sibling_edges = [e for e in wiki.edges if e.get("type") == "sibling"]
        # 2 consecutive pairs × 2 directions = 4 edges
        assert len(sibling_edges) == 4
        targets = {e["target"] for e in sibling_edges}
        sources = {e["source"] for e in sibling_edges}
        # Every node should appear as both source and target
        for node in wiki.nodes:
            assert node.id in targets
            assert node.id in sources

    def test_single_section_source_no_sibling_edges(self) -> None:
        """A source with only one section gets no sibling edges."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="Only section.",
                    heading="Intro",
                    chunk_id="c1",
                    source_file=Path("paper.pdf"),
                ),
            ),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        sibling_edges = [e for e in wiki.edges if e.get("type") == "sibling"]
        assert len(sibling_edges) == 0

    def test_no_source_file_no_sibling_edges(self) -> None:
        """Nodes without source_file do not get sibling edges."""
        chunks = _make_chunks(3, heading="Shared")
        wiki = GraphCompiler("model").compile(chunks)
        sibling_edges = [e for e in wiki.edges if e.get("type") == "sibling"]
        assert len(sibling_edges) == 0

    def test_sibling_edges_dedup_with_entity_edges(self) -> None:
        """Sibling edges are not duplicated if an entity edge already exists."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="Section 1 references Foo.",
                    heading="Section 1",
                    chunk_id="c1",
                    source_file=Path("paper.pdf"),
                ),
                entities=[{"name": "Foo", "type": "CONCEPT"}],
            ),
            ResolvedChunk(
                chunk=_chunk(
                    text="Section 2 also mentions Foo.",
                    heading="Section 2",
                    chunk_id="c2",
                    source_file=Path("paper.pdf"),
                ),
                entities=[{"name": "Foo", "type": "CONCEPT"}],
            ),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        # Entity edge creates one direction; sibling fills the other
        pair_edges = [e for e in wiki.edges if {e["source"], e["target"]} == {"node_0", "node_1"}]
        # Should have entity_reference + sibling in both directions = 3 max
        assert len(pair_edges) <= 3


class TestConceptPages:
    """Cross-cutting concept page generation (knox-wgt.7)."""

    def test_concept_page_created_for_cross_source_entity(self) -> None:
        """Entity in 2+ source documents gets a concept page."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="The MTD metric is widely used for evaluation.",
                    heading="Methods",
                    chunk_id="c1",
                    source_file=Path("paper-a.pdf"),
                ),
                entities=[{"name": "MTD", "type": "METRIC"}],
            ),
            ResolvedChunk(
                chunk=_chunk(
                    text="We employed MTD as our primary measure.",
                    heading="Results",
                    chunk_id="c2",
                    source_file=Path("paper-b.pdf"),
                ),
                entities=[{"name": "MTD", "type": "METRIC"}],
            ),
        ]
        compiler = GraphCompiler("model", min_concept_sources=2, max_concept_pages=50)
        wiki = compiler.compile(chunks)
        concepts = self._generated_concepts(wiki)
        assert len(concepts) == 1
        assert concepts[0].title == "MTD"
        assert concepts[0].metadata.get("source_count") == 2

    def test_no_concept_page_for_single_source_entity(self) -> None:
        """Entity in only 1 source does not get a concept page."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="The MTD metric.",
                    heading="Methods",
                    chunk_id="c1",
                    source_file=Path("paper-a.pdf"),
                ),
                entities=[{"name": "MTD", "type": "METRIC"}],
            ),
            ResolvedChunk(
                chunk=_chunk(
                    text="Also MTD.",
                    heading="Results",
                    chunk_id="c2",
                    source_file=Path("paper-a.pdf"),
                ),
                entities=[{"name": "MTD", "type": "METRIC"}],
            ),
        ]
        compiler = GraphCompiler("model", min_concept_sources=2, max_concept_pages=50)
        wiki = compiler.compile(chunks)
        assert len(self._generated_concepts(wiki)) == 0

    def test_concept_page_has_bidirectional_edges(self) -> None:
        """Concept page links to and from source sections."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="STOI is a standard metric.",
                    heading="Intro",
                    chunk_id="c1",
                    source_file=Path("paper-a.pdf"),
                ),
                entities=[{"name": "STOI", "type": "METRIC"}],
            ),
            ResolvedChunk(
                chunk=_chunk(
                    text="We compare against STOI.",
                    heading="Methods",
                    chunk_id="c2",
                    source_file=Path("paper-b.pdf"),
                ),
                entities=[{"name": "STOI", "type": "METRIC"}],
            ),
        ]
        compiler = GraphCompiler("model", min_concept_sources=2, max_concept_pages=50)
        wiki = compiler.compile(chunks)
        concept = next(n for n in wiki.nodes if n.metadata.get("cross_cutting"))
        concept_edges = [e for e in wiki.edges if e.get("type") == "concept_reference"]
        outgoing = [e for e in concept_edges if e["source"] == concept.id]
        incoming = [e for e in concept_edges if e["target"] == concept.id]
        assert len(outgoing) == 2
        assert len(incoming) == 2

    def test_concept_pages_disabled_by_default(self) -> None:
        """Without min_concept_sources config, no concept pages."""
        chunks = [
            ResolvedChunk(
                chunk=_chunk(
                    text="Entity A.",
                    heading="S1",
                    chunk_id="c1",
                    source_file=Path("a.pdf"),
                ),
                entities=[{"name": "EntityA", "type": "X"}],
            ),
            ResolvedChunk(
                chunk=_chunk(
                    text="Entity A again.",
                    heading="S2",
                    chunk_id="c2",
                    source_file=Path("b.pdf"),
                ),
                entities=[{"name": "EntityA", "type": "X"}],
            ),
        ]
        wiki = GraphCompiler("model").compile(chunks)
        assert not any(n.metadata.get("cross_cutting") for n in wiki.nodes)

    def test_max_concept_pages_caps_count(self) -> None:
        """No more than max_concept_pages concept nodes are created."""
        chunks: list[ResolvedChunk] = []
        for i in range(5):
            chunks.append(
                ResolvedChunk(
                    chunk=_chunk(
                        text=f"Concept {i} is discussed here.",
                        heading=f"Section {i}",
                        chunk_id=f"c{i}a",
                        source_file=Path(f"paper-{i}.pdf"),
                    ),
                    entities=[{"name": f"Concept{i}", "type": "X"}],
                ),
            )
            chunks.append(
                ResolvedChunk(
                    chunk=_chunk(
                        text=f"Concept {i} appears again.",
                        heading="Other",
                        chunk_id=f"c{i}b",
                        source_file=Path("shared.pdf"),
                    ),
                    entities=[{"name": f"Concept{i}", "type": "X"}],
                ),
            )
        compiler = GraphCompiler("model", min_concept_sources=2, max_concept_pages=2)
        wiki = compiler.compile(chunks)
        assert len(self._generated_concepts(wiki)) == 2

    @staticmethod
    def _generated_concepts(wiki: CompiledWiki) -> list[WikiNode]:
        return [n for n in wiki.nodes if n.metadata.get("cross_cutting")]


# ===========================================================================
#  WiCERCompiler
# ===========================================================================


class TestWiCERCompilerUnavailable:
    """WiCERCompiler behaviour when the LLM is unavailable.

    Without an LLM client, WiCER skips probe generation and evaluation
    entirely — no heuristic fallbacks. The base compiler output is
    returned as-is with zero WiCER iterations.
    """

    def test_empty_input(self) -> None:
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 3
        compiler = WiCERCompiler(config=config)
        wiki = compiler.compile([])
        assert isinstance(wiki, CompiledWiki)
        assert wiki.nodes == []
        assert wiki.metadata["source_count"] == 0
        assert wiki.metadata["source_names"] == []
        assert wiki.metadata["wicer_iterations"] == 0

    def test_no_llm_skips_wicer(self) -> None:
        """Without LLM, WiCER skips — base compiler output only, 0 iterations."""
        chunks = _make_chunks(count=2, heading="Test Section")
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 3
        compiler = WiCERCompiler(config=config)
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].title == "Test Section"
        assert wiki.metadata["wicer_iterations"] == 0

    def test_no_llm_no_probes(self) -> None:
        """Without LLM, no probes are generated (no heuristic fallback)."""
        chunks = _make_chunks(count=2, heading="Test")
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 1
        compiler = WiCERCompiler(config=config)
        wiki = compiler.compile(chunks)
        assert isinstance(wiki, CompiledWiki)
        assert wiki.metadata["wicer_iterations"] == 0

    def test_no_crash_on_empty_chunks(self) -> None:
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 3
        compiler = WiCERCompiler(config=config)
        wiki = compiler.compile([])
        assert wiki.metadata["chunk_count"] == 0

    def test_refine_recompiles_with_pinned_facts(self) -> None:
        """Refinement calls the base compiler with the cumulative pin set.

        Per Algorithm 1 of arXiv:2605.07068v1, refinement is a full
        re-compile of the source with the dropped facts injected as a
        hard preservation constraint — not a post-hoc node append.
        """

        class _RecordingCompiler(BaseCompiler):
            def __init__(self) -> None:
                self.calls: list[list[str] | None] = []

            def compile(
                self,
                chunks: list[ResolvedChunk],
                *,
                trees: list[ChunkTree] | None = None,
                pinned_facts: list[str] | None = None,
            ) -> CompiledWiki:
                del chunks, trees
                self.calls.append(list(pinned_facts) if pinned_facts else None)
                return CompiledWiki(
                    nodes=[WikiNode(id="n0", title="N", content="X", node_type="section")],
                    edges=[],
                    toc=[{"title": "N", "id": "n0", "level": 1}],
                    metadata={"chunk_count": 1},
                )

        chunks = _make_chunks(count=1, heading="Test")
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 0
        rec = _RecordingCompiler()
        compiler = WiCERCompiler(config=config, base_compiler=rec)

        # Seed pinned facts as a public-API caller would, so we exercise
        # the same flow the loop uses on iteration 1+.
        wiki = compiler.compile(chunks, pinned_facts=["KNOX is a knowledge base project.", "WiCER is iterative."])

        assert rec.calls == [["KNOX is a knowledge base project.", "WiCER is iterative."]]
        # No stray "Pinned Refinement Facts" node — refinement happens
        # via re-compile, not by appending to the wiki.
        assert all(n.id != "node_refinement_pinned_facts" for n in wiki.nodes)
        assert wiki.metadata["wicer_pinned_facts"] == 2

    def test_refine_accumulates_pinned_facts_monotonically(self) -> None:
        """F_cumulative grows across iterations; dedup preserves order."""

        class _RecordingCompiler(BaseCompiler):
            def __init__(self) -> None:
                self.calls: list[list[str] | None] = []

            def compile(
                self,
                chunks: list[ResolvedChunk],
                *,
                trees: list[ChunkTree] | None = None,
                pinned_facts: list[str] | None = None,
            ) -> CompiledWiki:
                del chunks, trees
                self.calls.append(list(pinned_facts) if pinned_facts else None)
                return CompiledWiki(
                    nodes=[WikiNode(id="n0", title="N", content="X", node_type="section")],
                    edges=[],
                    toc=[{"title": "N", "id": "n0", "level": 1}],
                    metadata={"chunk_count": 1},
                )

        chunks = _make_chunks(count=1, heading="Test")
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 0
        compiler = WiCERCompiler(config=config, base_compiler=_RecordingCompiler())

        # Seed F_cumulative; with max_iterations=0 the loop body does not
        # run, so we observe only the seed-dedup behavior of the initial
        # blind compile call.
        wiki = compiler.compile(chunks, pinned_facts=["Fact one.", "Fact one.", "Fact two."])

        # Dedup (preserve order) on the seed.
        assert wiki.metadata["wicer_pinned_facts"] == 2
        recording = compiler._compiler
        assert isinstance(recording, _RecordingCompiler)
        assert recording.calls == [["Fact one.", "Fact two."]]

    @pytest.fixture(autouse=True)
    def _no_llm(self) -> None:  # noqa: ANN202 — generator fixture
        return


class TestWiCERCompilerWithMock:
    """WiCERCompiler with a mocked LLM client for full loop coverage."""

    @pytest.fixture(autouse=True)
    def mock_llm(self) -> Generator[MagicMock]:
        """Patch the three WiCER PydanticAI agents used by WiCERCompiler."""
        controller = MagicMock()
        controller.probe = MagicMock(return_value=self._probe_output([]))
        controller.evaluate = MagicMock(return_value=self._evaluate_output([]))
        controller.diagnose = MagicMock(return_value=self._diagnose_output([]))
        with (
            patch.object(WicerProbeAgent, "run", controller.probe),
            patch.object(WicerEvaluateAgent, "run", controller.evaluate),
            patch.object(WicerDiagnoseAgent, "run", controller.diagnose),
        ):
            yield controller

    def test_llm_generated_probes_used(self, mock_llm: MagicMock) -> None:
        """When the LLM agents return probes, they are used for evaluation."""
        mock_llm.probe.return_value = self._probe_output(
            [
                {"question": "What is KNOX?", "answer": "A knowledge base project."},
                {"question": "What is WiCER?", "answer": "An iterative compilation loop."},
            ]
        )
        mock_llm.evaluate.return_value = self._evaluate_output(
            [
                {
                    "question": "What is KNOX?",
                    "expected_answer": "A knowledge base project.",
                    "found_answer": "A knowledge base project.",
                    "correct": True,
                },
                {
                    "question": "What is WiCER?",
                    "expected_answer": "An iterative compilation loop.",
                    "found_answer": "An iterative compilation loop.",
                    "correct": True,
                },
            ]
        )
        mock_llm.diagnose.return_value = self._diagnose_output([])
        chunks = _make_chunks(count=2, heading="Test")
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 1
        compiler = WiCERCompiler(config=config, context_window=_make_test_cw())
        compiler._has_llm = True
        wiki = compiler.compile(chunks)
        assert wiki.metadata["wicer_probes_generated"] == 2

    def test_llm_evaluation_detects_failures(self, mock_llm: MagicMock) -> None:
        """When evaluation says probes fail, refinement happens."""
        mock_llm.probe.return_value = self._probe_output([{"question": "What is KNOX?", "answer": "A knowledge base project."}])
        mock_llm.evaluate.return_value = self._evaluate_output(
            [{"question": "What is KNOX?", "expected_answer": "A knowledge base project.", "found_answer": "", "correct": False}]
        )
        mock_llm.diagnose.return_value = self._diagnose_output(["KNOX is a knowledge base project."])
        chunks = _make_chunks(count=1, heading="Test")
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 1
        config.budget.min_content_length = 0
        compiler = WiCERCompiler(config=config, context_window=_make_test_cw())
        compiler._has_llm = True
        wiki = compiler.compile(chunks)
        assert wiki.metadata["wicer_refinements"] == 1

    def test_all_probes_pass_loop_ends(self, mock_llm: MagicMock) -> None:
        """When all probes pass, the loop ends early."""
        mock_llm.probe.return_value = self._probe_output([{"question": "Q1?", "answer": "A1."}])
        mock_llm.evaluate.return_value = self._evaluate_output([{"question": "Q1?", "expected_answer": "A1.", "found_answer": "A1.", "correct": True}])
        mock_llm.diagnose.return_value = self._diagnose_output([])
        chunks = _make_chunks(count=1, heading="Test")
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 3
        compiler = WiCERCompiler(config=config, context_window=_make_test_cw())
        compiler._has_llm = True
        wiki = compiler.compile(chunks)
        assert wiki.metadata["wicer_iterations"] == 1
        assert wiki.metadata["wicer_refinements"] == 0

    def test_llm_failure_skips_wicer(self) -> None:
        """When the WiCER agents fail, WiCER skips (0 iterations)."""
        chunks = _make_chunks(count=2, heading="Fallback")
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 2
        compiler = WiCERCompiler(config=config, context_window=_make_test_cw())
        with (
            patch("knox.compilation.wicer.WicerProbeAgent.run", side_effect=LLMOutputError("boom")),
            patch("knox.compilation.wicer.WicerEvaluateAgent.run", side_effect=LLMOutputError("boom")),
            patch("knox.compilation.wicer.WicerDiagnoseAgent.run", side_effect=LLMOutputError("boom")),
        ):
            wiki = compiler.compile(chunks)
        assert len(wiki.nodes) >= 1
        assert wiki.metadata["wicer_iterations"] == 0

    def test_custom_base_compiler(self) -> None:
        """A custom BaseCompiler can be injected."""

        class DummyCompiler(BaseCompiler):
            def compile(
                self,
                chunks: list[ResolvedChunk],
                *,
                trees: list[ChunkTree] | None = None,
                pinned_facts: list[str] | None = None,
            ) -> CompiledWiki:
                del chunks, trees, pinned_facts
                return CompiledWiki(
                    nodes=[WikiNode(id="dummy", title="Dummy", content="X", node_type="section")],
                    edges=[],
                    toc=[{"title": "Dummy", "id": "dummy", "level": 1}],
                    metadata={"custom": True},
                )

        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 3
        compiler = WiCERCompiler(config=config, base_compiler=DummyCompiler())
        c = _chunk(text="test", heading="H")
        wiki = compiler.compile([ResolvedChunk(chunk=c)])
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].id == "dummy"
        assert wiki.metadata.get("custom") is True

    def _probe_output(self, items: list[dict]) -> WicerProbeOutput:
        return WicerProbeOutput(probes=list(items))

    def _evaluate_output(self, items: list[dict]) -> WicerEvaluateOutput:
        return WicerEvaluateOutput(results=list(items))

    def _diagnose_output(self, facts: list[str]) -> WicerDiagnoseOutput:
        return WicerDiagnoseOutput(critical_facts=list(facts))

    @staticmethod
    def _llm_result(payload: dict | list, content: str = "") -> MagicMock:
        """Build a fake LLMResult with json_data=payload."""
        result = MagicMock()
        if isinstance(payload, dict):
            result.json_data = payload
            result.content = content or json.dumps(payload)
        else:
            # When the LLM returns a bare list, json_data is None and the
            # client falls back to parsing content as JSON.
            result.json_data = None
            result.content = json.dumps(payload)
        return result


# ===========================================================================
#  Integration: WiCER with real GraphCompiler
# ===========================================================================


class TestWiCERWithGraphCompiler:
    """WiCERCompiler wrapping a real GraphCompiler."""

    def test_compiles_multi_section_doc(self) -> None:
        chunks = _make_diverse_chunks()
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 1
        compiler = WiCERCompiler(config=config, context_window=_make_test_cw())
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 4  # 4 sections
        assert "KNOX" in str(wiki.edges)
        assert wiki.metadata["chunk_count"] == 8

    def test_default_initializer(self) -> None:
        """WiCERCompiler() creates a GraphCompiler internally."""
        config = PipelineConfig(
            source_dir=Path("."), output_dir=Path("."), llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"), embedding=EmbeddingConfig(model="model")
        )
        config.budget.wicer_max_iterations = 3
        compiler = WiCERCompiler(config=config, context_window=_make_test_cw())
        assert compiler._compiler is not None
        assert isinstance(compiler._compiler, GraphCompiler)

    @pytest.fixture(autouse=True)
    def _no_llm(self) -> None:  # noqa: ANN202 — generator fixture
        return


# ===========================================================================
#  Empty-node filtering (knox-wgt.4)
# ===========================================================================


class TestEmptyNodeFiltering:
    """Title-only / heading-only pages are filtered by min_content_length."""

    def test_effective_content_length_strips_headings(self) -> None:
        assert BaseCompiler.effective_content_length("# Title\nBody text here.") == len("Body text here.")
        assert BaseCompiler.effective_content_length("# Title") == 0

    def test_filter_empty_groups_keeps_root_and_nonempty(self) -> None:
        groups = {
            "_root": [ResolvedChunk(chunk=_chunk("Overview text.", heading="", chunk_id="r"))],
            "Title Only": [ResolvedChunk(chunk=_chunk("# Title Only", heading="Title Only", chunk_id="t"))],
            "Has Body": [ResolvedChunk(chunk=_chunk("# Has Body\nSome useful content here.", heading="Has Body", chunk_id="b"))],
        }
        filtered = BaseCompiler.filter_empty_groups(groups, min_content_length=10)
        assert "_root" in filtered
        assert "Has Body" in filtered
        assert "Title Only" not in filtered

    def test_filter_empty_groups_disabled_when_zero(self) -> None:
        groups = {"Title Only": [ResolvedChunk(chunk=_chunk("# Title Only", heading="Title Only", chunk_id="t"))]}
        assert BaseCompiler.filter_empty_groups(groups, min_content_length=0) == groups

    def test_filter_empty_nodes_keeps_overview(self) -> None:
        nodes = [
            WikiNode(id="n1", title="Overview", content="Brief.", node_type="overview"),
            WikiNode(id="n2", title="Stub", content="# Stub", node_type="concept"),
        ]
        filtered = BaseCompiler.filter_empty_nodes(nodes, min_content_length=10)
        assert len(filtered) == 1
        assert filtered[0].id == "n1"

    def test_graph_compiler_filters_heading_only_nodes(self) -> None:
        chunks = [
            ResolvedChunk(chunk=_chunk("# Title Only", heading="Title Only", chunk_id="c1")),
            ResolvedChunk(chunk=_chunk("This section has enough body text to remain." * 3, heading="Real Section", chunk_id="c2")),
        ]
        compiler = GraphCompiler("model", min_content_length=20)
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].title == "Real Section"

    def test_dedup_content_duplicate_nodes_drops_identical_body(self) -> None:
        """knox-h7bm: singular/plural pages with identical content collapse to one."""
        body = "Phoneme posteriorgrams are dense representations of phoneme probabilities."
        nodes = [
            WikiNode(id="phoneme-posteriorgram", title="Phoneme Posteriorgram", content=body, node_type="concept"),
            WikiNode(id="phoneme-posteriorgrams", title="Phoneme Posteriorgrams", content=body, node_type="concept"),
        ]
        kept = BaseCompiler.dedup_content_duplicate_nodes(nodes)
        assert len(kept) == 1

    def test_dedup_content_duplicate_nodes_keeps_distinct_bodies(self) -> None:
        nodes = [
            WikiNode(id="a", title="A", content="Alpha concept body.", node_type="concept"),
            WikiNode(id="b", title="B", content="Beta concept body.", node_type="concept"),
        ]
        kept = BaseCompiler.dedup_content_duplicate_nodes(nodes)
        assert len(kept) == 2


class TestDeduplication:
    def test_repeated_heading_prefix_collapsed(self) -> None:
        c1 = _chunk(
            text="[2. Methods] First paragraph about methods.",
            heading="2. Methods",
            chunk_id="c1",
        )
        c2 = _chunk(
            text="[2. Methods] Second paragraph with more detail.",
            heading="2. Methods",
            chunk_id="c2",
        )
        rc1 = ResolvedChunk(chunk=c1)
        rc2 = ResolvedChunk(chunk=c2)
        wiki = GraphCompiler("model").compile([rc1, rc2])
        assert len(wiki.nodes) == 1
        content = wiki.nodes[0].content
        assert "[2. Methods]" not in content
        assert "First paragraph" in content
        assert "Second paragraph" in content

    def test_duplicate_paragraph_removed(self) -> None:
        para = "Posteriorgrams are used in the M-Measure."
        c1 = _chunk(text=para, heading="Methods", chunk_id="c1")
        c2 = _chunk(text=para, heading="Methods", chunk_id="c2")
        rc1 = ResolvedChunk(chunk=c1)
        rc2 = ResolvedChunk(chunk=c2)
        wiki = GraphCompiler("model").compile([rc1, rc2])
        assert wiki.nodes[0].content.count(para) == 1

    def test_reference_prefix_stripped(self) -> None:
        c1 = _chunk(
            text="[7. References] Smith, J. (2020). Important paper.",
            heading="7. References",
            chunk_id="c1",
        )
        c2 = _chunk(
            text="[7. References] Doe, A. (2021). Another study.",
            heading="7. References",
            chunk_id="c2",
        )
        rc1 = ResolvedChunk(chunk=c1)
        rc2 = ResolvedChunk(chunk=c2)
        wiki = GraphCompiler("model").compile([rc1, rc2])
        content = wiki.nodes[0].content
        assert "[7. References]" not in content
        assert "Smith, J. (2020)" in content
        assert "Doe, A. (2021)" in content

    def test_similar_but_different_paragraphs_kept(self) -> None:
        c1 = _chunk(text="The model was trained on data A.", heading="Training", chunk_id="c1")
        c2 = _chunk(text="The model was tested on data B.", heading="Training", chunk_id="c2")
        rc1 = ResolvedChunk(chunk=c1)
        rc2 = ResolvedChunk(chunk=c2)
        wiki = GraphCompiler("model").compile([rc1, rc2])
        content = wiki.nodes[0].content
        assert "trained on data A" in content
        assert "tested on data B" in content

    def test_deduplicate_group_text_empty(self) -> None:
        assert BaseCompiler.deduplicate_group_text([]) == ""


class TestTwoPhaseConceptSalvage:
    """knox-nbbx (concept-name salvage) + generation failure handling."""

    def test_concept_pages_skips_empty_names_without_silent_empty_result(self) -> None:
        """knox-nbbx regression: a bad extraction must not yield zero concept pages.

        Name salvage for ``concept``/``title`` drift keys now happens at
        PydanticAI validation (``schema.Concept.name`` is required), so
        ``Extraction.concepts`` only ever holds validated models. The remaining
        invariant: empty-name concepts are skipped, valid ones still produce
        pages (was: empty -> silent error).
        """
        ext_set = ExtractionSet()
        ext_set.add(
            Extraction(
                source="src.pdf",
                concepts=[
                    Concept(name="Phoneme Posteriorgram", definition="A dense representation."),
                    Concept(name="", definition="Attending to the better ear."),
                ],
            )
        )
        pages = ext_set.concept_pages()
        names = {p["name"] for p in pages}
        assert "Phoneme Posteriorgram" in names
        assert all(p.get("definitions") for p in pages)

    def test_entity_pages_merge_and_dedup(self) -> None:
        """Entities merge by normalized key with kind='entity' pre-assigned."""
        ext_set = ExtractionSet()
        ext_set.add(
            Extraction(
                source="a.pdf",
                entities=[
                    Entity(name="ITU-T", type="Organization", context="Published P.863."),
                    Entity(name="itu-t", type="Organization"),
                    Entity(name="", type="Organization"),
                ],
            )
        )
        ext_set.add(
            Extraction(
                source="b.pdf",
                entities=[Entity(name="ITU-T", type="", context="Standards body.")],
            )
        )
        pages = ext_set.entity_pages()
        assert len(pages) == 1
        page = pages[0]
        assert page["kind"] == "entity"
        assert page["name"] == "ITU-T"
        assert page["type"] == "Organization"
        assert sorted(page["sources"]) == ["a.pdf", "b.pdf"]
        assert page["definitions"] == ["Published P.863.", "Standards body."]

    def test_generation_failure_raises(self) -> None:
        """When generation LLM calls all fail, compile raises LLMOutputError — no silent empty wiki."""
        config = PipelineConfig(
            source_dir=Path("."),
            output_dir=Path("."),
            llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"),
            embedding=EmbeddingConfig(model="hash"),
        )
        config.no_cache = True
        compiler = TwoPhaseCompiler(config=config, base=GraphCompiler("model"))
        ext_set = ExtractionSet()
        ext_set.add(
            Extraction(
                source="src.pdf",
                concepts=[Concept(name="Listening Effort", definition="The effort to understand speech.")],
            )
        )
        chunks = _make_chunks(count=3, heading="Some Heading")
        # Extraction succeeds, generation returns empty (all calls fail)
        with (
            patch.object(compiler, "extract", return_value=ext_set),
            patch("knox.compilation.flows.generation.try_pydanticai_generation", return_value=None),
            pytest.raises(LLMOutputError, match=r"produced no page\(s\) for any of its 1 assigned concept\(s\)"),
        ):
            compiler.compile(chunks)


class TestVlmArtifactFiltering:
    """knox-ary0: vlm_filtered/vlm_skipped must never become concept pages."""

    def test_vlm_markers_not_in_graph_concept_pages(self) -> None:
        compiler = GraphCompiler("model", min_concept_sources=2, max_concept_pages=50)
        wiki = compiler.compile(self._make_vlm_chunks())
        concept_titles = {n.title for n in wiki.nodes if n.metadata.get("cross_cutting")}
        assert "vlm_filtered" not in concept_titles
        assert "vlm_skipped" not in concept_titles
        assert "Real Entity" in concept_titles

    def test_vlm_markers_not_in_extraction_concept_pages(self) -> None:
        ext_set = ExtractionSet()
        ext_set.add(
            Extraction(
                source="src.pdf",
                concepts=[
                    Concept(name="vlm_filtered", definition="Internal marker."),
                    Concept(name="vlm_skipped", definition="Internal marker."),
                    Concept(name="Listening Effort", definition="Real concept."),
                ],
            )
        )
        pages = ext_set.concept_pages()
        names = {p["name"] for p in pages}
        assert "vlm_filtered" not in names
        assert "vlm_skipped" not in names
        assert "Listening Effort" in names

    def _make_vlm_chunks(self) -> list[ResolvedChunk]:
        """Two chunks from different sources both carrying vlm_filtered entity."""
        c1 = ResolvedChunk(
            chunk=_chunk("Some text.", heading="Intro", chunk_id="c1", source_file=Path("a.pdf")),
            entities=[{"name": "vlm_filtered", "type": "Term"}, {"name": "Real Entity", "type": "Term"}],
        )
        c2 = ResolvedChunk(
            chunk=_chunk("More text.", heading="Intro", chunk_id="c2", source_file=Path("b.pdf")),
            entities=[{"name": "vlm_skipped", "type": "Term"}, {"name": "Real Entity", "type": "Term"}],
        )
        return [c1, c2]


class TestBatchFailureTracker:
    """_BatchFailureTracker: per-batch retry failure accounting → stop threshold."""

    def test_below_minimum_does_not_stop(self) -> None:
        tracker = BatchFailureTracker()
        # All fail but below the minimum sample size — must not stop (small compile)
        for _ in range(_FAILURE_THRESHOLD_MIN - 1):
            tracker.record(failed=True)
        assert tracker.should_stop() is False

    def test_clears_threshold_at_high_failure_rate(self) -> None:
        tracker = BatchFailureTracker()
        # 6 batches, 4 failed (>50%) → stop
        for failed in [True, True, False, True, True, False]:
            tracker.record(failed=failed)
        assert tracker.total == 6
        assert tracker.failed == 4
        assert tracker.should_stop() is True

    def test_low_failure_rate_does_not_stop(self) -> None:
        tracker = BatchFailureTracker()
        # 6 batches, 2 failed (33%) → under 50%, keep going
        for failed in [True, True, False, False, False, False]:
            tracker.record(failed=failed)
        assert tracker.should_stop() is False


class TestSlugifyLengthCap:
    """_slugify caps overly-long titles to stay within filesystem limits.

    Regression: the LLM sometimes emits a full sentence as a concept title,
    which slugified to 300+ chars and crashed write_text with
    ``OSError: [Errno 22] Invalid argument`` on Windows (255-char component limit).
    """

    def test_short_title_unchanged(self) -> None:
        assert slugify("Stochastic Gradient Descent") == "stochastic-gradient-descent"

    def test_long_title_is_capped(self) -> None:
        sentence = (
            "The training was performed in up to 20 epochs, stopping when the "
            "relative improvement was lower than 0.001. The starting learning "
            "rate was 0.008, halving it every time the relative improvement was "
            "lower than 0.01. These changes were the only modifications to an "
            "otherwise vanilla stochastic gradient descent optimizer."
        )
        slug = slugify(sentence)
        assert len(slug) <= _SLUG_MAX_LENGTH
        assert len(slug) > 0

    def test_distinct_long_titles_get_distinct_slugs(self) -> None:
        base = "word " * 50  # well over the cap
        a = slugify(base + " alpha ending")
        b = slugify(base + " beta ending")
        # Same prefix, different long titles → different hash suffixes
        assert a != b


# ===========================================================================
#  Helpers
# ===========================================================================


def _make_test_cw() -> ContextWindow:
    """Build a minimal ContextWindow for tests (no real LLM calls)."""
    return ContextWindow(max_input=128000, max_output=32768, source=ContextSource.FALLBACK)


def _make_chunks(
    count: int = 3,
    heading: str = "Section 1",
) -> list[ResolvedChunk]:
    """Create a list of ResolvedChunks with optional entities."""
    chunks: list[ResolvedChunk] = []
    for i in range(count):
        c = _chunk(
            text=f"Content for chunk {i}. Entity reference to KNOX.",
            heading=heading,
            chunk_id=f"chunk_{i:03d}",
        )
        rc = ResolvedChunk(
            chunk=c,
            entities=[Entity(name="KNOX", type="PROJECT", context="knowledge base").model_dump()] if i == 0 else [],
        )
        chunks.append(rc)
    return chunks


def _make_diverse_chunks() -> list[ResolvedChunk]:
    """Chunks across multiple headings for cross-section edge testing."""
    chunks: list[ResolvedChunk] = []
    for section in ["Introduction", "Methods", "Results", "Conclusion"]:
        for i in range(2):
            c = _chunk(
                text=f"{section} content paragraph {i}.",
                heading=section,
                chunk_id=f"chunk_{section.lower()}_{i}",
            )
            rc = ResolvedChunk(
                chunk=c,
                entities=[{"name": "KNOX", "type": "PROJECT", "context": "knowledge base"}, {"name": "WiCER", "type": "METHOD", "context": "compilation loop"}],
            )
            chunks.append(rc)
    return chunks


def _chunk(
    text: str = "Sample text.",
    heading: str = "",
    chunk_id: str = "chunk_001",
    source_file: Path | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[0],
        heading_context=heading,
        source_file=source_file,
    )
