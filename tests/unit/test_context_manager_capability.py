"""Tests for kompend's compaction wiring on pydantic_ai_harness.compaction."""

from __future__ import annotations

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.compaction import SlidingWindowCompaction, SummarizingCompaction

from kompend.agents.base import _CONTEXT_PRESETS, _SLIDING_KEEP_MESSAGES, _SLIDING_MAX_MESSAGES
from kompend.agents.context_manager import _extractive_summary, build_summarizing_compaction


class TestBuildSummarizingCompaction:
    def test_maps_the_kompend_knobs_onto_the_harness(self) -> None:
        capability = build_summarizing_compaction(
            context_window=200_000,
            compress_threshold=0.9,
            keep_messages=5,
            model="ollama/test-model",
        )

        assert isinstance(capability, SummarizingCompaction)
        assert capability.max_fraction == 0.9
        assert capability.keep_messages == 5
        assert capability.context_window == 200_000

    def test_uses_kompend_subclass_with_the_extractive_fallback(self) -> None:
        capability = build_summarizing_compaction(context_window=None, compress_threshold=0.7, keep_messages=0)

        assert type(capability).__mro__[1] is SummarizingCompaction

    def test_presets_are_ordered_by_aggressiveness(self) -> None:
        default_threshold, _ = _CONTEXT_PRESETS["default"]
        aggressive_threshold, aggressive_keep = _CONTEXT_PRESETS["aggressive"]

        assert aggressive_threshold < default_threshold
        assert aggressive_keep > 0


class TestSlidingWindowDefaults:
    def test_keeps_fewer_messages_than_it_triggers_on(self) -> None:
        assert _SLIDING_KEEP_MESSAGES < _SLIDING_MAX_MESSAGES
        capability = SlidingWindowCompaction(max_messages=_SLIDING_MAX_MESSAGES, keep_messages=_SLIDING_KEEP_MESSAGES)

        assert capability.keep_messages == _SLIDING_KEEP_MESSAGES


class TestExtractiveFallback:
    def test_renders_the_conversation_text(self) -> None:
        messages = [
            ModelRequest(parts=[UserPromptPart(content="first question")]),
            ModelResponse(parts=[TextPart(content="first answer")]),
        ]

        summary = _extractive_summary(messages)

        assert "first question" in summary
        assert "first answer" in summary

    def test_empty_history_has_a_placeholder(self) -> None:
        assert _extractive_summary([]) == "No previous conversation history."
