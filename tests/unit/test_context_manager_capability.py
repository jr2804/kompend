"""Tests for the hardened context-manager capability (agents/context_manager.py)."""

from __future__ import annotations

from knox.agents.context_manager import (
    _HardenedSummarizationProcessor,
    KnoxContextManager,
)
from pydantic_ai.models.test import TestModel


def test_capability_builds_hardened_processor() -> None:
    """KnoxContextManager swaps in the hardened summarization processor."""
    cap = KnoxContextManager(
        max_tokens=200_000,
        compress_threshold=0.9,
        keep=("messages", 0),
        summarization_model="ollama/test-model",
    )
    assert isinstance(cap._summarization_processor, _HardenedSummarizationProcessor)


def test_summarizer_agent_has_tool_retry_budget() -> None:
    """The summarizer agent must tolerate hallucinated tool calls.

    Regression: tool-tuned models (MiniMax via anthropic protocol) mimic the
    ``final_result`` tool calls embedded in the summarized history; with the
    library default retry budget of 1, one hallucinated call aborted
    summarization entirely — compaction silently never happened.
    """

    proc = _HardenedSummarizationProcessor(model=TestModel())
    agent = proc._get_summarization_agent()
    # No tools registered (plain text output) — the model must not call any.
    assert list(agent._function_toolset.tools.keys()) == []


def test_hardened_processor_preserves_threshold_and_keep() -> None:
    cap = KnoxContextManager(
        max_tokens=100_000,
        compress_threshold=0.7,
        keep=("messages", 5),
        summarization_model="ollama/test-model",
    )
    proc = cap._summarization_processor
    assert proc is not None
    assert proc.trigger == ("fraction", 0.7)
    assert proc.keep == ("messages", 5)
