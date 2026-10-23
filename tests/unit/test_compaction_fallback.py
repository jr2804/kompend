"""Compaction must survive a summarizer model that cannot answer."""

from __future__ import annotations

from pydantic_ai.messages import ModelRequest, UserPromptPart

from kompend.agents.context_manager import _FALLBACK_SUMMARY_CHARS, _extractive_summary


class TestExtractiveSummary:
    def test_keeps_the_rendered_history(self) -> None:
        summary = _extractive_summary(_history(2))

        assert "turn 0" in summary
        assert "turn 1" in summary

    def test_truncates_a_long_history_to_the_tail(self) -> None:
        summary = _extractive_summary(_history(400))

        assert summary.startswith("… ")
        assert len(summary) <= _FALLBACK_SUMMARY_CHARS + 2
        assert "turn 399" in summary

    def test_empty_history_has_a_placeholder(self) -> None:
        assert _extractive_summary([]) == "No previous conversation history."


def _history(count: int, size: int = 50) -> list[ModelRequest]:
    return [ModelRequest(parts=[UserPromptPart(content=f"turn {index} " + "x" * size)]) for index in range(count)]
