"""What surrounds a figure must reach the caption prompt.

A caption without the surrounding text is a guess: "a diagram" instead
of "a diagram of the two-stage pipeline". These tests pin the three
pieces the resolve stage assembles and the cache key that keeps a
changed caption from being replayed.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from kompend.core.config import EmbeddingConfig, LLMConfig, OutputConfig
from kompend.core.models import Chunk, ChunkNode, ChunkTree, ElementType, MetadataField, ResolvedChunk
from kompend.core.pipeline.config import Compiler, ImageMode, PipelineConfig
from kompend.core.pipeline.orchestrator import build_pipeline
from kompend.core.pipeline.stages.resolve import _collect_image_contexts, resolve_chunks
from kompend.resolution import HeuristicConfig
from kompend.resolution.base import BaseResolver


class _ContextSpyResolver(BaseResolver):
    """Records the figure context the resolve stage handed over."""

    def __init__(self) -> None:
        self.seen: list[dict[str, str]] = []
        self.calls: list[str] = []

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        self.calls.append(chunk.id)
        context = chunk.metadata.get(MetadataField.VLM_CONTEXT)
        if context is not None:
            self.seen.append(context)
        return ResolvedChunk(chunk=chunk, entities=[{"name": "figure"}])


def test_the_context_carries_the_section_the_neighbours_and_the_llm_summary() -> None:
    tree = _tree(
        [
            _chunk("c0", "The pipeline has two stages.", ElementType.TEXT),
            _chunk("c1", "Figure 1", ElementType.IMAGE, heading="Architecture"),
            _chunk("c2", "The second stage runs the assembler.", ElementType.TEXT),
        ],
        contextual="This chapter describes the two-stage pipeline.",
    )

    contexts = _collect_image_contexts([tree], max_chars=1200)

    context = contexts["c1"]
    assert context["section"] == "Architecture"
    assert "The pipeline has two stages." in context["nearby"]
    assert "The second stage runs the assembler." in context["nearby"]
    assert context["document_context"] == "This chapter describes the two-stage pipeline."


def test_the_neighbouring_text_stays_inside_the_budget() -> None:
    tree = _tree(
        [
            _chunk("c0", "x" * 500, ElementType.TEXT),
            _chunk("c1", "Figure 1", ElementType.IMAGE),
            _chunk("c2", "y" * 500, ElementType.TEXT),
        ],
    )

    contexts = _collect_image_contexts([tree], max_chars=100)

    assert len(contexts["c1"]["nearby"]) <= 100


def test_measured_visual_facts_ride_the_context(tmp_path: Path) -> None:
    image = tmp_path / "solid.png"
    Image.new("RGB", (40, 30), "gray").save(image)
    chunk = _chunk("c1", "Figure 1", ElementType.IMAGE)
    chunk.metadata[MetadataField.IMAGE_FILE] = str(image)
    tree = _tree([_chunk("c0", "See the figure.", ElementType.TEXT), chunk])

    context = _collect_image_contexts([tree], max_chars=500, heuristics=_heuristics())["c1"]

    assert "40×30 px" in context["visual_hints"]
    assert "near-uniform" in context["visual_hints"]
    assert context["heuristic_decision"] == "drop"
    assert context["heuristic_confidence"] == "0.97"


def test_the_hints_stay_inside_the_context_budget(tmp_path: Path) -> None:
    image = tmp_path / "solid.png"
    Image.new("RGB", (40, 30), "gray").save(image)
    chunk = _chunk("c1", "Figure 1", ElementType.IMAGE)
    chunk.metadata[MetadataField.IMAGE_FILE] = str(image)
    tree = _tree(
        [
            _chunk("c0", "x" * 500, ElementType.TEXT),
            chunk,
            _chunk("c2", "y" * 500, ElementType.TEXT),
        ]
    )

    context = _collect_image_contexts([tree], max_chars=100, heuristics=_heuristics())["c1"]
    assert len(context["nearby"]) + len(context["visual_hints"]) <= 100

    tight = _collect_image_contexts([tree], max_chars=20, heuristics=_heuristics())["c1"]
    assert len(tight["nearby"]) + len(tight["visual_hints"]) <= 20


def test_hints_are_absent_when_the_filter_is_off(tmp_path: Path) -> None:
    image = tmp_path / "solid.png"
    Image.new("RGB", (40, 30), "gray").save(image)
    chunk = _chunk("c1", "Figure 1", ElementType.IMAGE)
    chunk.metadata[MetadataField.IMAGE_FILE] = str(image)
    tree = _tree([chunk])

    context = _collect_image_contexts([tree], max_chars=500)["c1"]

    assert "visual_hints" not in context
    assert "heuristic_decision" not in context


def test_a_text_chunk_gets_no_figure_context() -> None:
    tree = _tree([_chunk("c0", "just text", ElementType.TEXT)])

    assert _collect_image_contexts([tree], max_chars=1200) == {}


def test_the_resolver_receives_the_context(tmp_path: Path) -> None:
    pipeline = build_pipeline(_make_config(tmp_path))
    resolver = _ContextSpyResolver()
    pipeline.resolver = resolver

    tree = _tree(
        [
            _chunk("c0", "The pipeline has two stages.", ElementType.TEXT),
            _chunk("c1", "Figure 1", ElementType.IMAGE, heading="Architecture"),
        ],
    )

    resolve_chunks(pipeline, [tree])

    assert resolver.seen
    assert resolver.seen[0]["section"] == "Architecture"
    assert "The pipeline has two stages." in resolver.seen[0]["nearby"]


def test_resolve_chunks_measures_hints_when_the_filter_is_on(tmp_path: Path) -> None:
    image = tmp_path / "solid.png"
    Image.new("RGB", (40, 30), "gray").save(image)
    pipeline = build_pipeline(_make_config(tmp_path, heuristics=True))
    resolver = _ContextSpyResolver()
    pipeline.resolver = resolver
    chunk = _chunk("c1", "Figure 1", ElementType.IMAGE)
    chunk.metadata[MetadataField.IMAGE_FILE] = str(image)

    resolve_chunks(pipeline, [_tree([chunk])])

    assert resolver.seen
    assert "visual_hints" in resolver.seen[0]
    assert resolver.seen[0]["heuristic_decision"] == "drop"


def test_a_changed_surrounding_invalidates_the_caption(tmp_path: Path) -> None:
    """The caption is written against the surroundings.

    When the neighbouring text changes, the old caption no longer
    describes the figure, so the resolve cache must not replay it.
    """
    pipeline = build_pipeline(_make_config(tmp_path))
    resolver = _ContextSpyResolver()
    pipeline.resolver = resolver

    first = _tree(
        [
            _chunk("c0", "The pipeline has two stages.", ElementType.TEXT),
            _chunk("c1", "Figure 1", ElementType.IMAGE),
        ],
    )
    resolve_chunks(pipeline, [first])
    assert resolver.calls.count("c1") == 1

    second = _tree(
        [
            _chunk("c0", "The pipeline has three stages now.", ElementType.TEXT),
            _chunk("c1", "Figure 1", ElementType.IMAGE),
        ],
    )
    resolve_chunks(pipeline, [second])

    assert resolver.calls.count("c1") == 2


def _tree(chunks: list[Chunk], *, contextual: str = "") -> ChunkTree:
    root = ChunkNode(id="root", children_ids=[f"n{i}" for i in range(len(chunks))])
    nodes: dict[str, ChunkNode] = {"root": root}
    for index, chunk in enumerate(chunks):
        nodes[f"n{index}"] = ChunkNode(
            id=f"n{index}",
            parent_id="root",
            chunk=chunk,
            contextual_context=contextual,
        )
    return ChunkTree(root_id="root", nodes=nodes)


def _chunk(chunk_id: str, text: str, element_type: ElementType, *, heading: str = "") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        element_types=[element_type],
        source_elements=[],
        heading_context=heading,
    )


def _heuristics() -> HeuristicConfig:
    return HeuristicConfig(
        enabled=True,
        min_aspect_ratio=20.0,
        max_thin_edge=96,
        max_analysis_edge=320,
        max_analysis_pixels=2_000_000,
        uniform_stddev=1.0,
        near_blank_stddev=4.0,
        near_blank_edge_density=0.001,
        transparency_ratio=0.995,
        min_confidence=0.9,
    )


def _make_config(tmp_path: Path, *, heuristics: bool = False) -> PipelineConfig:
    return PipelineConfig(
        source_dir=tmp_path,
        output_dir=tmp_path / "wiki",
        llm=LLMConfig(default_model="ollama/llama3.2", vlm_model=""),
        embedding=EmbeddingConfig(model=""),
        output=OutputConfig(),
        compiler=Compiler.GRAPH,
        image_mode=ImageMode.FILE,
        vlm_heuristic_filter_enabled=heuristics,
    )
