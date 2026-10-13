"""Tests for generation checkpoint persistence (save/load round-trip)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kompend.compilation._checkpoints import (
    delete_gen_checkpoint,
    load_gen_checkpoint,
    save_gen_checkpoint,
)
from kompend.core.config._configs import EmbeddingConfig, LLMConfig
from kompend.core.models import WikiNode
from kompend.core.pipeline.config import PipelineConfig


def test_round_trip_preserves_caller_progress(tmp_pipeline_config: PipelineConfig) -> None:
    deps = {"model": "ollama/test", "prompt_sha": "abc"}
    nodes = [WikiNode(id="node-0", title="Topic 0", content="content", node_type="concept")]
    slug_to_idx = {"node-0": 0}
    completed: set[int] = set()
    caller_progress = {"seg_0": ["Topic 0"], "seg_1": ["Topic 1", "Topic 2"]}

    save_gen_checkpoint(
        tmp_pipeline_config,
        deps,
        nodes,
        slug_to_idx,
        completed,
        caller_progress=caller_progress,
    )

    loaded = load_gen_checkpoint(tmp_pipeline_config, deps)
    assert loaded is not None
    ld_nodes, ld_slug_to_idx, ld_completed, ld_caller_progress = loaded
    assert [n.id for n in ld_nodes] == ["node-0"]
    assert ld_slug_to_idx == {"node-0": 0}
    assert ld_completed == set()
    assert ld_caller_progress == caller_progress


def test_load_returns_none_when_no_cache(tmp_pipeline_config: PipelineConfig) -> None:
    assert load_gen_checkpoint(tmp_pipeline_config, {"model": "x"}) is None


def test_load_returns_none_on_deps_mismatch(tmp_pipeline_config: PipelineConfig) -> None:
    deps = {"model": "a"}
    save_gen_checkpoint(tmp_pipeline_config, deps, [], {}, set(), caller_progress={"seg_0": ["x"]})
    assert load_gen_checkpoint(tmp_pipeline_config, {"model": "b"}) is None


def test_delete_removes_checkpoint(tmp_pipeline_config: PipelineConfig) -> None:
    deps = {"model": "a"}
    save_gen_checkpoint(tmp_pipeline_config, deps, [], {}, set(), caller_progress={"seg_0": ["x"]})
    assert load_gen_checkpoint(tmp_pipeline_config, deps) is not None
    delete_gen_checkpoint(tmp_pipeline_config, deps)
    assert load_gen_checkpoint(tmp_pipeline_config, deps) is None


def test_save_is_idempotent_on_no_cache(tmp_pipeline_config: PipelineConfig) -> None:
    """``no_cache=True`` short-circuits save/load/delete (parity with existing semantics)."""
    tmp_pipeline_config.no_cache = True
    deps = {"model": "a"}
    save_gen_checkpoint(tmp_pipeline_config, deps, [], {}, set(), caller_progress={"seg_0": ["x"]})
    assert load_gen_checkpoint(tmp_pipeline_config, deps) is None  # no_cache prevents reading too
    delete_gen_checkpoint(tmp_pipeline_config, deps)  # also a no-op


@pytest.fixture
def tmp_pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Minimal PipelineConfig rooted at *tmp_path* with deterministic cache_dir."""
    base = tmp_path
    sources = base / "sources"
    output = base / "wiki"
    sources.mkdir()
    output.mkdir()
    return PipelineConfig(
        source_dir=sources,
        output_dir=output,
        llm=LLMConfig(),
        embedding=EmbeddingConfig(),
        base_dir=base,
    )
