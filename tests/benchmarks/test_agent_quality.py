"""Agent pipeline benchmark.

Measures latency and throughput of the PydanticAI agent layer without
contacting a real LLM. The underlying ``run_sync`` is mocked so the
benchmark exercises the full kompend wiring path:

  Model string → build_model → KompendModel → Agent construction →
  Template render → run(deps) → result validation

Reports:

  - Agent construction latency (model + agent build)
  - Template render latency (system + user prompt)
  - Full run latency (construction + render + mock run_sync)
  - Throughput (runs/sec)

Skips if ``pydantic_ai`` cannot be imported.

This benchmark was originally scoped as "PydanticAI vs raw litellm" but
litellm was removed entirely (commit c8055c0). The comparison is now
"per-agent overhead" — establishing a baseline so future optimisations
can be measured.
"""

from __future__ import annotations

import importlib
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kompend.agents.deps import PipelineAgentDeps, QueryDeps, ToolGenerationDeps
from kompend.compilation.schema import ExtractionOutput

# Agents to benchmark — each is a KompendAgentBase subclass
_AGENT_SPECS: list[tuple[str, str, str, str, bool]] = [
    # (label, module_name, class_name, template_dir, is_query_agent)
    ("extraction", "kompend.agents.extraction", "ExtractionAgent", "compilation", False),
    ("generation", "kompend.agents.generation", "AgenticGenerationAgent", "compilation", False),
    ("query", "kompend.agents.query", "QueryAgent", "query", True),
]

pytestmark = pytest.mark.benchmark


class TestAgentConstructionBenchmark:
    """Benchmark: agent construction latency (model string → KompendAgentBase instance)."""

    @pytest.mark.parametrize(("label", "module_name", "cls_name", "_dir", "_is_query"), _AGENT_SPECS)
    def test_construction_latency(self, label: str, module_name: str, cls_name: str, _dir: str, _is_query: bool) -> None:
        agent_cls = _import_agent(module_name, cls_name)
        _agent, elapsed = _benchmark_construction(agent_cls)
        print(f"\n  [{label}] construction: {elapsed * 1000:.1f} ms")
        # Sanity: construction should be under 2 s even with model resolution
        assert elapsed < 2.0


class TestTemplateRenderBenchmark:
    """Benchmark: system-prompt template render latency."""

    @pytest.mark.parametrize(("label", "module_name", "cls_name", "_dir", "is_query"), _AGENT_SPECS)
    def test_render_latency(self, label: str, module_name: str, cls_name: str, _dir: str, is_query: bool) -> None:
        agent_cls = _import_agent(module_name, cls_name)
        agent, _ = _benchmark_construction(agent_cls)
        deps = QueryDeps(question="test", context="ctx", source_id="test") if is_query else PipelineAgentDeps(text="test chunk", source_id="test")
        elapsed = _benchmark_template_render(agent, deps)
        print(f"\n  [{label}] system-prompt render: {elapsed * 1000:.1f} ms")
        # Sanity: template render should be under 500 ms
        assert elapsed < 0.5


class TestFullRunBenchmark:
    """Benchmark: full agent.run() with mocked LLM call."""

    @pytest.mark.parametrize(("label", "module_name", "cls_name", "_dir", "is_query"), _AGENT_SPECS)
    def test_run_latency(self, label: str, module_name: str, cls_name: str, _dir: str, is_query: bool) -> None:
        agent_cls = _import_agent(module_name, cls_name)

        if label == "extraction":
            deps = PipelineAgentDeps(text="test chunk", source_id="test")
            result_obj = ExtractionOutput(
                summary="Test",
                entities=[],
                concepts=[],
                tags=[],
                findings=[],
            )
        elif label == "generation":
            deps = ToolGenerationDeps(
                source_content="test content",
                source_id="test",
            )
            # AgenticGenerationAgent returns GenerationOutput but we mock run_sync
            result_obj = MagicMock()
        elif label == "query":
            deps = QueryDeps(
                question="test question",
                context="test context",
                source_id="test",
            )
            result_obj = "test answer"
        else:
            pytest.skip(f"No deps configured for {label}")

        elapsed = _benchmark_full_run(agent_cls, result_obj, deps)
        throughput = 1.0 / elapsed if elapsed > 0 else float("inf")
        print(f"\n  [{label}] full run (mocked): {elapsed * 1000:.1f} ms | throughput: {throughput:.0f} runs/sec")
        # Sanity: full run with mocked LLM should be under 1 s
        assert elapsed < 1.0


def _import_agent(module_name: str, class_name: str) -> type:
    """Import an agent class lazily."""
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _benchmark_construction(agent_cls: type, model_string: str = "ollama/qwen3") -> tuple[Any, float]:
    """Measure agent construction time (excludes lazy pydantic_ai.Agent build)."""
    start = time.perf_counter()
    agent = agent_cls(
        model_string=model_string,
        timeout=30.0,
        max_tokens=2048,
        retries=1,
        reasoning_effort="none",
    )
    elapsed = time.perf_counter() - start
    return agent, elapsed


def _benchmark_template_render(agent: Any, deps: Any) -> float:
    """Measure system-prompt template render time."""
    start = time.perf_counter()
    agent.render_system_prompt(deps)
    elapsed = time.perf_counter() - start
    return elapsed


def _benchmark_full_run(agent_cls: type, result_obj: Any, deps: Any) -> float:
    """Measure full run() with mocked run_sync — exercises the entire wiring path."""
    agent = agent_cls(
        model_string="ollama/qwen3",
        timeout=30.0,
        max_tokens=2048,
        retries=1,
        reasoning_effort="none",
    )

    # Mock the lazy pydantic_ai Agent's run_sync
    mock_result = MagicMock()
    mock_result.data = result_obj
    mock_usage = MagicMock()
    mock_result.usage.return_value = mock_usage
    mock_result.all_messages.return_value = []

    with patch.object(agent, "_build_agent") as mock_build:
        mock_pa = MagicMock()
        mock_pa.run_sync.return_value = mock_result
        mock_build.return_value = mock_pa

        start = time.perf_counter()
        agent.run(deps)
        elapsed = time.perf_counter() - start

    return elapsed


@pytest.fixture(scope="module", autouse=True)
def _check_deps() -> None:
    """Skip all benchmarks if pydantic_ai is unavailable."""
    pytest.importorskip("pydantic_ai")
