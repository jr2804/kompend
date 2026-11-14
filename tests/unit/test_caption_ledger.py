"""Every caption verdict lands in the stage ledger with its reason.

The resolve stage writes one ``resolve`` row per figure
(``_record_caption_row``). ``kompend explain`` and the doctor's failure
histogram read those rows, so a dropped figure must say which gate
dropped it — the cheap pre-filter, the text relevance gate, or the VLM
itself — and carry the gate's own reason string.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

from kompend.core.config import EmbeddingConfig, LLMConfig, OutputConfig
from kompend.core.models import Chunk, ChunkNode, ChunkTree, ElementType, MetadataField
from kompend.core.pipeline.config import Compiler, ImageMode, PipelineConfig
from kompend.core.pipeline.orchestrator import build_pipeline
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.pipeline.stages.resolve import resolve_chunks
from kompend.core.store import open_store, vault_db_path
from kompend.llm.decision import FigureRelevanceVerdict
from kompend.resolution import HeuristicConfig, VLMEnricher
from kompend.resolution.chain import ResolverChain


class _FakeAgent:
    """Minimal stand-in for ``VlmAgent`` — records calls, returns a caption."""

    def __init__(self, caption: str = "A caption.") -> None:
        self._caption = caption
        self.calls = 0

    def run(self, deps: object) -> str:
        self.calls += 1
        return self._caption


def test_a_heuristic_drop_records_which_gate_and_why(tmp_path: Path) -> None:
    image = _solid(tmp_path / "solid.png")
    chunk = _chunk("c1", "Figure 1", ElementType.IMAGE)
    chunk.metadata["image_file"] = str(image)
    pipeline = build_pipeline(_make_config(tmp_path))
    pipeline.resolver = _enricher(_FakeAgent(), heuristics=_heuristics())

    resolve_chunks(pipeline, [_tree([chunk])])

    rows = _caption_rows(pipeline)
    assert len(rows) == 1
    source, outcome, reason_code, detail = rows[0]
    assert outcome == "skipped"
    assert reason_code == "vlm_filtered"
    assert "near-uniform" in (detail or "")


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


def test_a_relevance_skip_names_the_text_gate(tmp_path: Path) -> None:
    relevance = MagicMock()
    relevance.run.return_value = SimpleNamespace(verdict=FigureRelevanceVerdict.DECORATIVE)
    chunk = _figure_chunk(tmp_path, "c1")
    pipeline = build_pipeline(_make_config(tmp_path))
    pipeline.resolver = _enricher(_FakeAgent(), relevance_agent=relevance)

    resolve_chunks(pipeline, [_tree([chunk])])

    rows = _caption_rows(pipeline)
    assert len(rows) == 1
    _, outcome, reason_code, detail = rows[0]
    assert outcome == "skipped"
    assert reason_code == "vlm_relevance_skipped"
    assert "decorative" in (detail or "")


def test_a_model_skip_records_vlm_skipped(tmp_path: Path) -> None:
    agent = _FakeAgent(caption="SKIP")
    chunk = _figure_chunk(tmp_path, "c1")
    pipeline = build_pipeline(_make_config(tmp_path))
    pipeline.resolver = _enricher(agent)

    resolve_chunks(pipeline, [_tree([chunk])])

    rows = _caption_rows(pipeline)
    assert len(rows) == 1
    _, outcome, reason_code, detail = rows[0]
    assert outcome == "skipped"
    assert reason_code == "vlm_skipped"
    assert detail == "SKIP"
    assert agent.calls == 1


def test_an_enriched_figure_records_ok_without_a_reason(tmp_path: Path) -> None:
    chunk = _figure_chunk(tmp_path, "c1")
    pipeline = build_pipeline(_make_config(tmp_path))
    pipeline.resolver = _enricher(_FakeAgent())

    resolve_chunks(pipeline, [_tree([chunk])])

    rows = _caption_rows(pipeline)
    assert rows == [("c1", "ok", None, None)]


def test_a_figure_without_an_image_path_is_degraded(tmp_path: Path) -> None:
    chunk = _chunk("c1", "Figure 1", ElementType.IMAGE)
    pipeline = build_pipeline(_make_config(tmp_path))
    pipeline.resolver = _enricher(_FakeAgent())

    resolve_chunks(pipeline, [_tree([chunk])])

    rows = _caption_rows(pipeline)
    assert len(rows) == 1
    _, outcome, reason_code, _detail = rows[0]
    assert outcome == "degraded"
    assert reason_code == "no_parser"


def test_a_cache_hit_records_cached_with_the_carried_reason(tmp_path: Path) -> None:
    chunk = _figure_chunk(tmp_path, "c1")
    config = _make_config(tmp_path)
    pipeline = build_pipeline(config)
    pipeline.resolver = _enricher(_FakeAgent())
    resolve_chunks(pipeline, [_tree([chunk])])

    pipeline = build_pipeline(config)
    pipeline.resolver = _enricher(_FakeAgent())
    resolve_chunks(pipeline, [_tree([_figure_chunk(tmp_path, "c1")])])

    rows = _caption_rows(pipeline)
    assert [outcome for _, outcome, _, _ in rows] == ["ok", "cached"]
    assert rows[0][2] is None


def test_two_figures_of_one_source_get_distinct_rows(tmp_path: Path) -> None:
    chunks = [_figure_chunk(tmp_path, "c1"), _figure_chunk(tmp_path, "c2", "Figure 2")]
    pipeline = build_pipeline(_make_config(tmp_path))
    pipeline.resolver = _enricher(_FakeAgent())

    resolve_chunks(pipeline, [_tree(chunks)])

    rows = _caption_rows(pipeline)
    assert [source for source, _, _, _ in rows] == ["c1", "c2"]
    assert all(outcome == "ok" for _, outcome, _, _ in rows)


def _figure_chunk(tmp_path: Path, chunk_id: str, text: str = "Figure 1") -> Chunk:
    """An IMAGE chunk whose image file exists on disk."""
    chunk = _chunk(chunk_id, text, ElementType.IMAGE)
    chunk.metadata[MetadataField.IMAGE_FILE] = str(_solid(tmp_path / f"{chunk_id}.png"))
    return chunk


def _solid(path: Path) -> Path:
    Image.new("RGB", (400, 300), "#808080").save(path)
    return path


def test_text_chunks_leave_no_caption_rows(tmp_path: Path) -> None:
    chunk = _chunk("c1", "Plain text.", ElementType.TEXT)
    pipeline = build_pipeline(_make_config(tmp_path))
    pipeline.resolver = _enricher(_FakeAgent())

    resolve_chunks(pipeline, [_tree([chunk])])

    assert _caption_rows(pipeline) == []


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        source_dir=tmp_path,
        output_dir=tmp_path / "wiki",
        llm=LLMConfig(default_model="ollama/llama3.2", vlm_model=""),
        embedding=EmbeddingConfig(model=""),
        output=OutputConfig(),
        compiler=Compiler.GRAPH,
        image_mode=ImageMode.FILE,
    )


def _chunk(chunk_id: str, text: str, element_type: ElementType) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        element_types=[element_type],
        source_elements=[],
        heading_context="Section",
    )


def _tree(chunks: list[Chunk]) -> ChunkTree:
    root = ChunkNode(id="root", children_ids=[f"n{i}" for i in range(len(chunks))])
    nodes: dict[str, ChunkNode] = {"root": root}
    for index, chunk in enumerate(chunks):
        nodes[f"n{index}"] = ChunkNode(id=f"n{index}", parent_id="root", chunk=chunk)
    return ChunkTree(root_id="root", nodes=nodes)


def _enricher(agent: _FakeAgent, **kwargs: object) -> ResolverChain:
    return ResolverChain([VLMEnricher(agent=agent, min_image_bytes=0, min_image_dimension=0, **kwargs)])  # ty: ignore[invalid-argument-type]


def _caption_rows(pipeline: PipelineRuntime) -> list[tuple[str, str, str | None, str | None]]:
    with open_store(vault_db_path(pipeline.cache_directory())) as conn:
        rows = conn.execute("SELECT source, outcome, reason_code, detail FROM stages WHERE stage = 'resolve' ORDER BY rowid").fetchall()
    return [(str(r[0]), str(r[1]), str(r[2]) if r[2] is not None else None, str(r[3]) if r[3] is not None else None) for r in rows]
