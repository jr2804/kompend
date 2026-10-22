"""Hardened context-manager capability for kompend agents.

Wraps :class:`~pydantic_ai_summarization.ContextManagerCapability` with a
summarizer agent that survives tool-hallucinating models.

The library's internal summarizer agent is built with no tools and a retry
budget of 1. Tool-tuned models (e.g. MiniMax via the anthropic protocol) see
``final_result`` tool calls in the embedded conversation history and mimic
the pattern — the hallucinated call burns the retry budget on the first
attempt, summarization aborts, and compaction silently never happens (the
processor keeps the original history, so context grows until the segment
fails). The hardened agent raises the retry budget so the model can
self-correct after the "Unknown tool name" feedback, and states explicitly
that no tools exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai_summarization import ContextManagerCapability, SummarizationProcessor

__all__ = ["KompendContextManager"]

#: Fallback tool-call retry budget for the summarizer agent when no config
#: value is available. One hallucinated tool call must not abort
#: summarization — the ModelRetry feedback ("Unknown tool name") is usually
#: enough for the model to answer in plain text.
_DEFAULT_TOOL_RETRIES = 3


@dataclass
class _HardenedSummarizationProcessor(SummarizationProcessor):
    """SummarizationProcessor whose agent tolerates hallucinated tool calls."""

    tool_retries: int = _DEFAULT_TOOL_RETRIES

    def _get_summarization_agent(self) -> Agent[None, str]:
        if self._summarization_agent is None:
            self._summarization_agent = Agent(
                self.model,
                instructions=(
                    "You are a context summarization assistant. "
                    "Extract the most important information from conversations. "
                    "Respond with plain text only — you have no tools and must "
                    "never attempt to call any."
                ),
                retries=self.tool_retries,
            )
        return self._summarization_agent


@dataclass
class KompendContextManager(ContextManagerCapability):
    """ContextManagerCapability whose summarizer survives tool-hallucinating models."""

    tool_retries: int = _DEFAULT_TOOL_RETRIES

    def __post_init__(self) -> None:
        super().__post_init__()
        self._summarization_processor = _HardenedSummarizationProcessor(
            trigger=("fraction", self.compress_threshold),
            keep=self.keep,
            model=self.summarization_model,
            token_counter=self.token_counter,
            summary_prompt=self.summary_prompt,
            max_input_tokens=self._resolved_max_tokens,
            tool_retries=self.tool_retries,
        )
