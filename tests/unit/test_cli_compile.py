"""CLI compile command wiring tests."""

from __future__ import annotations

from pathlib import Path

from knox.cli._shared import build_cli_overrides as _build_cli_overrides
from knox.core.config import EmbeddingConfig, LLMConfig, OutputConfig
from knox.core.pipeline import build_pipeline
from knox.core.pipeline_defs import Compiler, ImageMode, PipelineConfig


def test_cli_overrides_include_vlm_model() -> None:
    overrides = _build_cli_overrides(vlm_model="openai/gpt-4o")
    assert overrides["compile.llm.vlm_model"] == "openai/gpt-4o"


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


def test_build_pipeline_adds_vlm_enricher_when_configured(tmp_path: Path) -> None:
    """When vlm_model is set, the resolver chain contains a VLMEnricher."""
    config = _make_config(tmp_path, vlm_model="openai/gpt-4o")
    pipeline = build_pipeline(config)
    assert pipeline.resolver is not None
    resolver_names = {r.__class__.__name__ for r in pipeline.resolver._resolvers}
    assert "VLMEnricher" in resolver_names


def test_build_pipeline_skips_vlm_without_model(tmp_path: Path) -> None:
    """When vlm_model is unset, the resolver chain has no VLMEnricher."""
    config = _make_config(tmp_path, vlm_model="")
    pipeline = build_pipeline(config)
    assert pipeline.resolver is not None
    resolver_names = {r.__class__.__name__ for r in pipeline.resolver._resolvers}
    assert "VLMEnricher" not in resolver_names
