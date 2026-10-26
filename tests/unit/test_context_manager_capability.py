"""Tests for kompend's compaction wiring on pydantic_ai_harness.compaction."""

from __future__ import annotations

import pytest
from pydantic_ai_harness.compaction import SlidingWindowCompaction, SummarizingCompaction

from kompend.agents.base import _CONTEXT_PRESETS, _SLIDING_KEEP_MESSAGES, _SLIDING_MAX_MESSAGES
from kompend.agents.context_manager import build_summarizing_compaction


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
        default_threshold, _ = _CONTEXT_PRESETS["default"]
        aggressive_threshold, aggressive_keep = _CONTEXT_PRESETS["aggressive"]

        assert aggressive_threshold < default_threshold
        assert aggressive_keep > 0


class TestSlidingWindowDefaults:
    def test_keeps_fewer_messages_than_it_triggers_on(self) -> None:
        assert _SLIDING_KEEP_MESSAGES < _SLIDING_MAX_MESSAGES
        capability = SlidingWindowCompaction(max_messages=_SLIDING_MAX_MESSAGES, keep_messages=_SLIDING_KEEP_MESSAGES)

        assert capability.keep_messages == _SLIDING_KEEP_MESSAGES


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
            await capability._summarize([], ctx=None)  # type: ignore[arg-type]  # noqa: SLF001
