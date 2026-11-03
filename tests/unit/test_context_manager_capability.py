"""Tests for kompend's compaction wiring on pydantic_ai_harness.compaction."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import FallbackCompaction, SlidingWindowCompaction, SummarizingCompaction

from kompend.agents.base import _CONTEXT_PRESETS, _SLIDING_KEEP_MESSAGES, _SLIDING_MAX_MESSAGES
from kompend.agents.context_manager import build_fallback_compaction, build_summarizing_compaction
from kompend.core.config import ContextManagerPreset


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

    def test_uses_the_kompend_logging_subclass(self) -> None:
        capability = build_summarizing_compaction(context_window=None, compress_threshold=0.7, keep_messages=0)

        assert type(capability).__mro__[1] is SummarizingCompaction

    def test_presets_are_ordered_by_aggressiveness(self) -> None:
        default_threshold, _ = _CONTEXT_PRESETS[ContextManagerPreset.DEFAULT]
        aggressive_threshold, aggressive_keep = _CONTEXT_PRESETS[ContextManagerPreset.AGGRESSIVE]

        assert aggressive_threshold < default_threshold
        assert aggressive_keep > 0


class TestSlidingWindowDefaults:
    def test_keeps_fewer_messages_than_it_triggers_on(self) -> None:
        assert _SLIDING_KEEP_MESSAGES < _SLIDING_MAX_MESSAGES
        capability = SlidingWindowCompaction(max_messages=_SLIDING_MAX_MESSAGES, keep_messages=_SLIDING_KEEP_MESSAGES)

        assert capability.keep_messages == _SLIDING_KEEP_MESSAGES


class TestFallbackCompaction:
    """A summarizer that cannot answer trims history instead of killing the run.

    Regression: pydantic/pydantic-ai#12307 — the harness's summarizer agent
    runs with ``ToolManager``'s default retry budget of 1, so one unknown-tool
    call (easy: it reads the run's own tool calls in the transcript) raises
    ``UnexpectedModelBehavior`` and aborts the parent agent.
    """

    def test_builds_a_summarizer_then_a_sliding_window(self) -> None:
        capability = build_fallback_compaction(
            context_window=200_000,
            compress_threshold=0.9,
            keep_messages=0,
            model="ollama/test-model",
        )

        assert isinstance(capability, FallbackCompaction)
        summarizer, fallback = capability.fallback_chain
        assert isinstance(summarizer, SummarizingCompaction)
        assert summarizer.max_fraction == 0.9
        assert summarizer.context_window == 200_000
        assert isinstance(fallback, SlidingWindowCompaction)
        # The fallback trims on the same trigger, keeping half of it.
        assert fallback.max_fraction == 0.9
        assert fallback.keep_tokens == 90_000

    def test_only_model_behaviour_falls_through(self) -> None:
        """Provider and programming errors must still burn."""
        capability = build_fallback_compaction(context_window=8192, compress_threshold=0.9, keep_messages=0)

        assert capability.fallback_on == (UnexpectedModelBehavior,)

    async def test_a_dead_summarizer_falls_back_to_the_sliding_window(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        capability = build_fallback_compaction(
            context_window=8192,
            compress_threshold=0.8,
            keep_messages=1,
            model="ollama/test-model",
        )
        summarizer = capability.fallback_chain[0]

        async def _broken(*args: object, **kwargs: object) -> str:
            raise UnexpectedModelBehavior("Tool 'lookup' exceeded max retries count of 1")

        monkeypatch.setattr(summarizer, "_summarize", _broken)
        messages = self._history(turns=120)

        with caplog.at_level("WARNING", logger="kompend.agents.context_manager"):
            compacted = await capability.compact(messages, _ctx())

        assert len(compacted) < len(messages)
        assert "Summarizer failed" in caplog.text

    async def test_a_broken_provider_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capability = build_fallback_compaction(
            context_window=8192,
            compress_threshold=0.8,
            keep_messages=1,
            model="ollama/test-model",
        )
        summarizer = capability.fallback_chain[0]

        async def _broken(*args: object, **kwargs: object) -> str:
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(summarizer, "_summarize", _broken)

        with pytest.raises(RuntimeError, match="provider exploded"):
            await capability.compact(self._history(turns=120), _ctx())

    @staticmethod
    def _history(turns: int) -> list[ModelMessage]:
        messages: list[ModelMessage] = []
        for index in range(turns):
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"question {index} " + "padding " * 40)]))
            messages.append(ModelResponse(parts=[TextPart(content=f"answer {index} " + "padding " * 40)]))
        return messages


class TestCompactionFailurePropagates:
    """A summarizer model that cannot answer must burn, not degrade silently.

    Policy (.agents/CODE.md, "LLM calls"): no fallback summary, no
    hand-crafted stand-in — the exception reaches the caller.
    """

    async def test_summarizer_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capability = build_summarizing_compaction(
            context_window=8192,
            compress_threshold=0.8,
            keep_messages=2,
            model="ollama/test-model",
        )

        async def _broken(*args: object, **kwargs: object) -> str:
            raise RuntimeError("Tool 'final_result' exceeded max retries count of 1")

        monkeypatch.setattr(capability, "_summarize", _broken)

        with pytest.raises(RuntimeError, match="exceeded max retries"):
            await capability._summarize([], ctx=cast(RunContext[object], None))  # noqa: SLF001


class TestWrittenSummaryCompat:
    """pydantic_ai_harness 0.62 returns a WrittenSummary record from ``_summarize``.

    kompend only logs the summary length; the record must pass through
    unchanged (regression: ``len(WrittenSummary)`` crashed every compaction).
    """

    async def test_summary_record_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Record:
            text = "the summary"
            created_at = datetime(2026, 10, 27, 16, 33, 20)

        capability = build_summarizing_compaction(
            context_window=8192,
            compress_threshold=0.8,
            keep_messages=2,
            model="ollama/test-model",
        )
        record = _Record()
        monkeypatch.setattr(SummarizingCompaction, "_summarize", self._returning(record))

        result = await capability._summarize([], ctx=cast(RunContext[object], None))  # noqa: SLF001

        assert result is record

    async def test_plain_string_summary_still_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capability = build_summarizing_compaction(
            context_window=8192,
            compress_threshold=0.8,
            keep_messages=2,
            model="ollama/test-model",
        )
        monkeypatch.setattr(SummarizingCompaction, "_summarize", self._returning("plain summary"))

        result = await capability._summarize([], ctx=cast(RunContext[object], None))  # noqa: SLF001

        assert result == "plain summary"

    @staticmethod
    def _returning(value: object) -> object:
        async def _stub(*args: object, **kwargs: object) -> object:
            return value

        return _stub


def _ctx() -> RunContext[object]:
    """A bare run context; the compaction capabilities only read ``.model``/``.tracer``."""
    return RunContext(deps=object(), model=TestModel(), usage=RunUsage())
