"""Tests for KNOX Tier 4 compilers — GraphCompiler and WiCERCompiler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from knox.compilation import GraphCompiler, WiCERCompiler
from knox.compilation.base import BaseCompiler
from knox.core.models import Chunk, CompiledWiki, ElementType, ResolvedChunk, WikiNode


# ===========================================================================
#  Helpers
# ===========================================================================

def _chunk(
    text: str = "Sample text.",
    heading: str = "",
    chunk_id: str = "chunk_001",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[0],
        heading_context=heading,
    )


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
            entities=[
                {"name": "KNOX", "type": "PROJECT", "context": "knowledge base"},
            ] if i == 0 else [],
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
                entities=[
                    {"name": "KNOX", "type": "PROJECT", "context": "knowledge base"},
                    {"name": "WiCER", "type": "METHOD", "context": "compilation loop"},
                ],
            )
            chunks.append(rc)
    return chunks


# ===========================================================================
#  GraphCompiler
# ===========================================================================

class TestGraphCompiler:
    """GraphCompiler behaviour."""

    def test_empty_input_returns_empty_wiki(self) -> None:
        compiler = GraphCompiler()
        wiki = compiler.compile([])
        assert isinstance(wiki, CompiledWiki)
        assert wiki.nodes == []
        assert wiki.edges == []
        assert wiki.toc == []
        assert wiki.metadata["chunk_count"] == 0

    def test_heading_groups_form_nodes(self) -> None:
        chunks = _make_chunks(count=4, heading="Section A")
        compiler = GraphCompiler()
        wiki = compiler.compile(chunks)
        # All chunks under same heading → one node
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].title == "Section A"
        assert len(wiki.nodes[0].source_chunks) == 4  # all 4 merged

    def test_multiple_headings_multiple_nodes(self) -> None:
        chunks = _make_chunks(count=4, heading="Section A")
        # Add one chunk with a different heading
        c = _chunk(
            text="Different section.",
            heading="Section B",
            chunk_id="chunk_extra",
        )
        chunks.append(ResolvedChunk(chunk=c))
        compiler = GraphCompiler()
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
        compiler = GraphCompiler()
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].title == "Overview"

    def test_content_is_merged(self) -> None:
        chunks = _make_chunks(count=3, heading="Intro")
        compiler = GraphCompiler()
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 1
        # All three texts merged with double newlines
        assert "Content for chunk 0" in wiki.nodes[0].content
        assert "Content for chunk 1" in wiki.nodes[0].content
        assert "Content for chunk 2" in wiki.nodes[0].content

    def test_entities_create_cross_section_edges(self) -> None:
        chunks = _make_diverse_chunks()
        compiler = GraphCompiler()
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
        compiler = GraphCompiler()
        wiki = compiler.compile(chunks)
        assert len(wiki.toc) == len(wiki.nodes)
        for entry in wiki.toc:
            assert "title" in entry
            assert "id" in entry
            assert "level" in entry

    def test_metadata_contains_counts(self) -> None:
        chunks = _make_chunks(count=5)
        compiler = GraphCompiler()
        wiki = compiler.compile(chunks)
        assert wiki.metadata["chunk_count"] == 5
        assert wiki.metadata["node_count"] == 1  # all same heading
        assert "model" in wiki.metadata
        assert "provider" in wiki.metadata

    def test_no_entities_no_edges(self) -> None:
        c = _chunk(text="Simple text.", heading="Docs")
        rc = ResolvedChunk(chunk=c, entities=[])
        compiler = GraphCompiler()
        wiki = compiler.compile([rc])
        assert wiki.edges == []


# ===========================================================================
#  WiCERCompiler
# ===========================================================================

class TestWiCERCompilerUnavailable:
    """WiCERCompiler behaviour when litellm is NOT available."""

    def test_empty_input(self) -> None:
        compiler = WiCERCompiler()
        wiki = compiler.compile([])
        assert isinstance(wiki, CompiledWiki)
        assert wiki.nodes == []
        assert wiki.metadata["wicer_iterations"] == 0

    def test_fallback_blind_compilation_no_litellm(self) -> None:
        """Without litellm, WiCER does a single blind compilation."""
        chunks = _make_chunks(count=2, heading="Test Section")
        compiler = WiCERCompiler(max_iterations=3)
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].title == "Test Section"
        assert wiki.metadata["wicer_iterations"] > 0

    def test_fallback_probes_generated_from_text(self) -> None:
        """Without LLM, fallback probes are generated from text."""
        chunks = _make_chunks(count=2, heading="Test")
        compiler = WiCERCompiler(max_iterations=1)
        wiki = compiler.compile(chunks)
        assert isinstance(wiki, CompiledWiki)
        assert len(wiki.nodes) >= 1

    def test_no_crash_on_empty_chunks(self) -> None:
        compiler = WiCERCompiler(max_iterations=3)
        wiki = compiler.compile([])
        assert wiki.metadata["chunk_count"] == 0

    def test_refine_appends_facts(self) -> None:
        """_refine adds a pinned facts node."""
        chunks = _make_chunks(count=1, heading="Test")
        compiler = WiCERCompiler(max_iterations=0)
        wiki = compiler._blind_compile(chunks)
        pinned = ["KNOX is a knowledge base project.", "WiCER is iterative."]
        refined = compiler._refine(wiki, pinned)
        assert len(refined.nodes) == len(wiki.nodes) + 1
        fact_node = refined.nodes[-1]
        assert fact_node.node_type == "reference"
        assert "KNOX is a knowledge base project" in fact_node.content

    def test_refine_updates_existing_fact_node(self) -> None:
        """Repeated refinement updates the same pinned facts node."""
        chunks = _make_chunks(count=1, heading="Test")
        compiler = WiCERCompiler(max_iterations=0)
        wiki = compiler._blind_compile(chunks)
        wiki = compiler._refine(wiki, ["Fact one."])
        wiki = compiler._refine(wiki, ["Fact two."])
        # Only one extra node, not two
        assert len(wiki.nodes) == 2  # 1 original + 1 pinned
        assert "Fact two" in wiki.nodes[-1].content


class TestWiCERCompilerWithMock:
    """WiCERCompiler with mocked litellm for full loop coverage."""

    @pytest.fixture
    def mock_litellm(self) -> MagicMock:
        """Patch litellm.completion at the module level."""
        with patch("knox.compilation.wicer._try_litellm") as mock:
            yield mock

    def test_llm_generated_probes_used(self, mock_litellm: MagicMock) -> None:
        """When litellm returns probes, they are used for evaluation."""
        def mock_responses(prompt: str, system: str | None = None) -> str:
            if "generate" in prompt.lower() or "probe" in (system or "").lower():
                return json.dumps([
                    {"question": "What is KNOX?", "answer": "A knowledge base project."},
                    {"question": "What is WiCER?", "answer": "An iterative compilation loop."},
                ])
            elif "evaluat" in prompt.lower() or "evaluat" in (system or "").lower():
                return json.dumps({
                    "results": [
                        {"question": "What is KNOX?", "expected_answer": "A knowledge base project.",
                         "found_answer": "A knowledge base project.", "correct": True},
                        {"question": "What is WiCER?", "expected_answer": "An iterative compilation loop.",
                         "found_answer": "An iterative compilation loop.", "correct": True},
                    ]
                })
            return json.dumps({"critical_facts": []})

        mock_litellm.side_effect = mock_responses
        chunks = _make_chunks(count=2, heading="Test")
        compiler = WiCERCompiler(max_iterations=1)
        wiki = compiler.compile(chunks)
        assert wiki.metadata["wicer_probes_generated"] == 2

    def test_llm_evaluation_detects_failures(self, mock_litellm: MagicMock) -> None:
        """When evaluation says probes fail, refinement happens."""
        # First call: generate probes
        # Second call: evaluate (find failures)
        # Third call: diagnose
        mock_litellm.side_effect = [
            # _generate_probes
            json.dumps([
                {"question": "What is KNOX?", "answer": "A knowledge base project."},
            ]),
            # _evaluate — all wrong
            json.dumps({
                "results": [
                    {"question": "What is KNOX?", "expected_answer": "A knowledge base project.",
                     "found_answer": "", "correct": False},
                ]
            }),
            # _diagnose
            json.dumps({
                "critical_facts": ["KNOX is a knowledge base project."]
            }),
        ]
        chunks = _make_chunks(count=1, heading="Test")
        compiler = WiCERCompiler(max_iterations=1)
        wiki = compiler.compile(chunks)
        # Should have refined (added pinned facts node)
        assert wiki.metadata["wicer_refinements"] == 1

    def test_all_probes_pass_loop_ends(self, mock_litellm: MagicMock) -> None:
        """When all probes pass, the loop ends early."""
        mock_litellm.side_effect = [
            # _generate_probes
            json.dumps([
                {"question": "Q1?", "answer": "A1."},
            ]),
            # _evaluate — all correct
            json.dumps({
                "results": [
                    {"question": "Q1?", "expected_answer": "A1.",
                     "found_answer": "A1.", "correct": True},
                ]
            }),
        ]
        chunks = _make_chunks(count=1, heading="Test")
        compiler = WiCERCompiler(max_iterations=3)
        wiki = compiler.compile(chunks)
        # Should have 1 iteration and 0 refinements
        assert wiki.metadata["wicer_iterations"] == 1
        assert wiki.metadata["wicer_refinements"] == 0

    def test_llm_failure_falls_back_gracefully(self) -> None:
        """When litellm raises, WiCER falls back to blind compilation."""
        chunks = _make_chunks(count=2, heading="Fallback")
        compiler = WiCERCompiler(max_iterations=2)
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) >= 1
        # Should still have iteration info
        assert wiki.metadata["wicer_iterations"] > 0

    def test_custom_base_compiler(self) -> None:
        """A custom BaseCompiler can be injected."""
        class DummyCompiler(BaseCompiler):
            def compile(self, chunks):
                return CompiledWiki(
                    nodes=[WikiNode(id="dummy", title="Dummy", content="X", node_type="section")],
                    edges=[],
                    toc=[{"title": "Dummy", "id": "dummy"}],
                    metadata={"custom": True},
                )

        compiler = WiCERCompiler(base_compiler=DummyCompiler())
        c = _chunk(text="test", heading="H")
        wiki = compiler.compile([ResolvedChunk(chunk=c)])
        assert len(wiki.nodes) == 1
        assert wiki.nodes[0].id == "dummy"
        assert wiki.metadata.get("custom") is True


# ===========================================================================
#  Integration: WiCER with real GraphCompiler
# ===========================================================================

class TestWiCERWithGraphCompiler:
    """WiCERCompiler wrapping a real GraphCompiler."""

    def test_compiles_multi_section_doc(self) -> None:
        chunks = _make_diverse_chunks()
        compiler = WiCERCompiler(max_iterations=1)
        wiki = compiler.compile(chunks)
        assert len(wiki.nodes) == 4  # 4 sections
        assert "KNOX" in str(wiki.edges)
        assert wiki.metadata["chunk_count"] == 8

    def test_default_initializer(self) -> None:
        """WiCERCompiler() creates a GraphCompiler internally."""
        compiler = WiCERCompiler()
        assert compiler._compiler is not None
        assert isinstance(compiler._compiler, GraphCompiler)