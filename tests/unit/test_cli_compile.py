"""CLI compile command wiring tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kompend.cli._shared import build_cli_overrides as _build_cli_overrides
from kompend.cli.app import app
from kompend.core.config import EmbeddingConfig, LLMConfig, OutputConfig
from kompend.core.models import Chunk, ChunkNode, ChunkTree, ElementType, ResolvedChunk
from kompend.core.pipeline.config import Compiler, ImageMode, PipelineConfig
from kompend.core.pipeline.orchestrator import build_pipeline
from kompend.core.pipeline.stages._resolve import resolve_chunks
from kompend.llm.context_window import ContextSource, ContextWindow
from kompend.resolution.base import BaseResolver
from kompend.resolution.chain import ResolverChain


class _OrderTrackingResolver(BaseResolver):
    """Mock resolver that records the element_types of each resolved chunk."""

    def __init__(self) -> None:
        self.resolved_types: list[ElementType] = []

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        self.resolved_types.append(chunk.element_types[0] if chunk.element_types else ElementType.TEXT)
        return ResolvedChunk(chunk=chunk)


def test_cli_overrides_include_vlm_model() -> None:
    overrides = _build_cli_overrides(vlm_model="openai/gpt-4o")
    assert overrides["compile.llm.vlm_model"] == "openai/gpt-4o"


def test_build_pipeline_wires_the_parsecraft_router(tmp_path: Path) -> None:
    """build_pipeline wires the parsecraft router from the config options.

    There is no backend selector any more: parsecraft is the only parser path.
    """
    config = _make_config(tmp_path, vlm_model="")
    with patch(
        "kompend.core.pipeline.stages._factory.discover_context_window",
        return_value=ContextWindow(max_input=128000, max_output=32768, source=ContextSource.FALLBACK),
    ):
        pipeline = build_pipeline(config)
    assert pipeline.router is not None
    # The default backend is parsecraft, which claims PDFs through its own
    # native/docling/liteparse sub-backends.
    assert pipeline.router.route(Path("a.pdf")) is not None
    assert type(pipeline.router.route(Path("a.pdf"))).__name__ == "ParsecraftParser"


def test_build_pipeline_adds_vlm_enricher_when_configured(tmp_path: Path) -> None:
    """When vlm_model is set, the resolver chain contains a VLMEnricher."""
    config = _make_config(tmp_path, vlm_model="openai/gpt-4o")
    pipeline = build_pipeline(config)
    resolver = pipeline.resolver
    assert isinstance(resolver, ResolverChain)
    resolver_names = {r.__class__.__name__ for r in resolver._resolvers}
    assert "VLMEnricher" in resolver_names


def test_build_pipeline_skips_vlm_without_model(tmp_path: Path) -> None:
    """When vlm_model is unset, the resolver chain has no VLMEnricher."""
    config = _make_config(tmp_path, vlm_model="")
    pipeline = build_pipeline(config)
    resolver = pipeline.resolver
    assert isinstance(resolver, ResolverChain)
    resolver_names = {r.__class__.__name__ for r in resolver._resolvers}
    assert "VLMEnricher" not in resolver_names


def test_iter_sources_scans_source_dir_when_input_files_is_none(tmp_path: Path) -> None:
    """iter_sources discovers all supported files from source_dir when input_files is None."""
    _populate_source_dir(tmp_path, "doc1.pdf", "doc2.md", "doc3.txt", "notes.html")
    config = _make_config(tmp_path, vlm_model="")
    pipeline = build_pipeline(config)
    sources = list(pipeline.iter_sources())
    assert len(sources) == 4
    assert all(s.exists() for s in sources)


def test_iter_sources_scans_source_dir_even_with_input_files_set(tmp_path: Path) -> None:
    """iter_sources ignores input_files and always scans source_dir.

    This is the key behavioral change: input_files no longer scopes
    which files the pipeline processes. All files in source_dir are
    discovered regardless of input_files.
    """
    _populate_source_dir(tmp_path, "doc1.pdf", "doc2.md", "doc3.txt")
    config = _make_config(tmp_path, vlm_model="")
    config.input_files = [tmp_path / "doc1.pdf"]  # only one file in input_files
    pipeline = build_pipeline(config)
    sources = list(pipeline.iter_sources())
    # Should find all 3 files in source_dir, not just the 1 in input_files
    assert len(sources) == 3
    assert all(s.exists() for s in sources)


def test_iter_sources_skips_unsupported_extensions(tmp_path: Path) -> None:
    """Discovery follows the configured backend's declared suffixes.

    parsecraft reads source files as plain text (``.py``), so a code file is
    ingestible; it declares no CSV media type, so ``data.csv`` is skipped.
    """
    _populate_source_dir(tmp_path, "doc1.pdf", "doc2.md", "notes.html", "data.csv", "script.py")
    config = _make_config(tmp_path, vlm_model="")
    pipeline = build_pipeline(config)
    sources = {s.name for s in pipeline.iter_sources()}
    assert sources == {"doc1.pdf", "doc2.md", "notes.html", "script.py"}


def _populate_source_dir(tmp_path: Path, *filenames: str) -> None:
    """Create empty files in tmp_path to simulate a source directory."""
    for name in filenames:
        (tmp_path / name).touch()


def test_resolve_batches_image_chunks_last(tmp_path: Path) -> None:
    """Image chunks are resolved together after all non-image chunks.

    This prevents VLM reload churn: without batching, image chunks
    scattered across 1000+ chunks cause Ollama to load/unload the VLM
    repeatedly. Batching keeps the VLM loaded for one contiguous burst.
    """
    config = _make_config(tmp_path, vlm_model="")
    pipeline = build_pipeline(config)
    pipeline.resolver = _OrderTrackingResolver()
    resolver = pipeline.resolver
    assert isinstance(resolver, _OrderTrackingResolver)
    # Build a tree with interleaved image/text chunks:
    #   text1, image1, text2, image2, text3
    chunks = [
        Chunk(id=f"c{i}", text=f"text{i}", element_types=[t], source_elements=[])
        for i, t in enumerate(
            [ElementType.TEXT, ElementType.IMAGE, ElementType.TEXT, ElementType.IMAGE, ElementType.TEXT],
        )
    ]
    root = ChunkNode(id="root", children_ids=[f"n{i}" for i in range(len(chunks))])
    nodes: dict[str, ChunkNode] = {"root": root}
    for i, c in enumerate(chunks):
        nodes[f"n{i}"] = ChunkNode(id=f"n{i}", parent_id="root", chunk=c)
    tree = ChunkTree(root_id="root", nodes=nodes)

    resolved = resolve_chunks(pipeline, [tree])

    assert len(resolved) == 5
    # Image chunks must all be at the end, contiguous
    types = resolver.resolved_types
    assert types == [ElementType.TEXT, ElementType.TEXT, ElementType.TEXT, ElementType.IMAGE, ElementType.IMAGE]


def _make_config(tmp_path: Path, *, vlm_model: str) -> PipelineConfig:
    return PipelineConfig(
        source_dir=tmp_path,
        output_dir=tmp_path / "wiki",
        llm=LLMConfig(default_model="ollama/llama3.2", vlm_model=vlm_model),
        embedding=EmbeddingConfig(model=""),
        output=OutputConfig(),
        image_mode=ImageMode.FILE,
        compiler=Compiler.GRAPH,
    )


def test_compile_shows_traceback_on_unexpected_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected (non-LLMOutputError) exceptions print a full traceback, not just a message.

    Regression for the 'unhashable type: dict' crash: the generic except
    branch previously printed only `Error: {exc}`, hiding the call site.
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "note.md").write_text("# Test\ncontent")
    kompend_dir = tmp_path / ".kompend"
    kompend_dir.mkdir()
    (kompend_dir / "config.toml").write_text("", encoding="utf-8")

    class _BoomPipeline:
        def run(self, progress_callback: object | None = None) -> None:
            raise RuntimeError("synthetic boom")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KOMP_LLM_DEFAULT_MODEL", "test-model")
    monkeypatch.setattr("kompend.cli.cmds.pipeline.pipeline.check_llm_availability", lambda p: None)
    monkeypatch.setattr("kompend.cli.cmds.pipeline.pipeline.build_pipeline", lambda cfg: _BoomPipeline())

    runner = CliRunner()
    result = runner.invoke(app, ["compile"])

    assert result.exit_code == 1
    assert "Error: synthetic boom" in result.output
    assert "Traceback (most recent call last)" in result.output
