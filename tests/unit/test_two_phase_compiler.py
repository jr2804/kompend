"""Compile accounting - the phase lines read the shared UsageTracker.

knox-zu3w: the compiler kept a private list for the call and token totals,
but nothing ever appended to it, so every phase line and the compile summary
printed zeros while the run had made thousands of calls. The numbers live in
the shared :class:`~kompend.llm.usage.UsageTracker` that every agent of the
run records into; these tests stub the two phases with known call counts and
assert the lines carry them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from kompend.compilation.two_phase import ExtractionSet, TwoPhaseCompiler
from kompend.core.config import EmbeddingConfig, LLMConfig
from kompend.core.models import Chunk, CompiledWiki, ElementType, ExtractionStats, ResolvedChunk, WikiField, WikiNode
from kompend.core.models.enums import NodeType
from kompend.core.models.metadata import meta_dict
from kompend.core.pipeline.config import PipelineConfig
from kompend.llm.context_window import ContextSource, ContextWindow
from tests.conftest import make_tmp_vault

_EXTRACT_CALLS = 2
_GENERATE_CALLS = 3
_EXTRACT_PROMPT = 150
_EXTRACT_COMPLETION = 60
_GENERATE_PROMPT = 200
_GENERATE_COMPLETION = 100


def test_phase_lines_carry_the_calls_the_run_made(caplog: pytest.LogCaptureFixture) -> None:
    """Every phase line reports the calls its own phase made."""
    vault = make_tmp_vault()
    compiler = _compiler(vault)

    with caplog.at_level(logging.INFO):
        _run_compile(compiler, vault)

    records = caplog.records
    extract_line = _log_line(records, "TwoPhase: extract phase done")
    assert f"{_EXTRACT_CALLS} LLM call(s)" in extract_line
    assert f"{_EXTRACT_CALLS * _EXTRACT_PROMPT} prompt + {_EXTRACT_CALLS * _EXTRACT_COMPLETION} completion tokens" in extract_line

    generate_line = _log_line(records, "TwoPhase: generate phase done")
    assert f"{_GENERATE_CALLS} LLM call(s)" in generate_line
    assert f"{_GENERATE_CALLS * _GENERATE_PROMPT} prompt + {_GENERATE_CALLS * _GENERATE_COMPLETION} completion tokens" in generate_line

    done_line = _log_line(records, "TwoPhase: compile done")
    assert f"{_EXTRACT_CALLS + _GENERATE_CALLS} LLM call(s)" in done_line
    assert (
        f"{_EXTRACT_CALLS * _EXTRACT_PROMPT + _GENERATE_CALLS * _GENERATE_PROMPT} prompt + "
        f"{_EXTRACT_CALLS * _EXTRACT_COMPLETION + _GENERATE_CALLS * _GENERATE_COMPLETION} completion tokens" in done_line
    )


def _log_line(caplog_records: list[logging.LogRecord], prefix: str) -> str:
    for record in caplog_records:
        message = record.getMessage()
        if message.startswith(prefix):
            return message
    raise AssertionError(f"no log line starting with {prefix!r}")


def test_compile_summary_agrees_with_the_usage_snapshot() -> None:
    """The summary and the usage snapshot read the same tracker - one account."""
    vault = make_tmp_vault()
    compiler = _compiler(vault)

    wiki = _run_compile(compiler, vault)

    assert wiki.metadata["llm_calls"] == _EXTRACT_CALLS + _GENERATE_CALLS
    assert wiki.metadata["llm_prompt_tokens"] == _EXTRACT_CALLS * _EXTRACT_PROMPT + _GENERATE_CALLS * _GENERATE_PROMPT
    assert wiki.metadata["llm_completion_tokens"] == _EXTRACT_CALLS * _EXTRACT_COMPLETION + _GENERATE_CALLS * _GENERATE_COMPLETION
    snapshot = meta_dict(wiki.metadata.get(str(WikiField.LLM_USAGE)))
    assert snapshot["call_count"] == wiki.metadata["llm_calls"]
    assert snapshot["prompt_tokens"] == wiki.metadata["llm_prompt_tokens"]
    assert snapshot["completion_tokens"] == wiki.metadata["llm_completion_tokens"]


def _compiler(vault: Path) -> TwoPhaseCompiler:
    config = PipelineConfig(
        source_dir=vault / "sources",
        output_dir=vault / "wiki",
        llm=LLMConfig(default_model="ollama/test", vlm_model="vlm"),
        embedding=EmbeddingConfig(model="model"),
    )
    config.no_cache = True
    return TwoPhaseCompiler(config=config, context_window=_make_cw())


def _make_cw() -> ContextWindow:
    return ContextWindow(max_input=8192, max_output=4096, source=ContextSource.FALLBACK)


def _run_compile(compiler: TwoPhaseCompiler, vault: Path) -> CompiledWiki:
    with (
        patch.object(compiler, "extract", side_effect=_stub_extract(compiler)),
        patch.object(compiler, "generate", side_effect=_stub_generate(compiler)),
    ):
        return compiler.compile(_make_chunks(vault))


def _make_chunks(vault: Path) -> list[ResolvedChunk]:
    return [
        ResolvedChunk(
            chunk=Chunk(
                id="chunk_001",
                text="Some content about kompend.",
                element_types=[ElementType.TEXT],
                source_elements=[0],
                heading_context="Section 1",
                source_file=vault / "sources" / "doc.md",
            ),
        ),
    ]


def _stub_extract(compiler: TwoPhaseCompiler) -> Callable[..., ExtractionSet]:
    """Extraction that records its calls the way the agents do."""

    def _extract(chunks: list[ResolvedChunk], *, trees: object = None) -> ExtractionSet:
        for _ in range(_EXTRACT_CALLS):
            compiler._usage_tracker.record(
                prompt_tokens=_EXTRACT_PROMPT,
                completion_tokens=_EXTRACT_COMPLETION,
                model="test-model",
            )
        compiler._extraction_stats.append(ExtractionStats(source="doc.md", duration_s=1.0))
        return ExtractionSet()

    return _extract


def _stub_generate(compiler: TwoPhaseCompiler) -> Callable[..., CompiledWiki]:
    """Generation that records its calls the way the agents do."""

    def _generate(
        extractions: ExtractionSet,
        chunks: list[ResolvedChunk],
        *,
        trees: object = None,
        pinned_facts: list[str] | None = None,
    ) -> CompiledWiki:
        for _ in range(_GENERATE_CALLS):
            compiler._usage_tracker.record(
                prompt_tokens=_GENERATE_PROMPT,
                completion_tokens=_GENERATE_COMPLETION,
                model="test-model",
            )
        return CompiledWiki(nodes=[WikiNode(id="concept-1", title="Concept", content="Body.", node_type=NodeType.CONCEPT)], edges=[], toc=[])

    return _generate
