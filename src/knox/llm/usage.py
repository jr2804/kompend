"""LLM usage tracking — accumulates token counts across LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageTracker:
    """Accumulates LLM usage statistics across complete() calls.

    Each :class:`LLMClient` holds one instance. After pipeline runs,
    the caller reads ``client.usage`` to populate the wiki log.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    total_ttft_s: float = 0.0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)
    # Latest context-window utilization reported by ContextManagerCapability.
    context_tokens: int = 0
    context_budget: int = 0

    def record(self, *, prompt_tokens: int, completion_tokens: int, model: str, ttft_s: float = 0.0) -> None:
        """Record one successful completion."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.call_count += 1
        self.total_ttft_s += ttft_s
        entry = self.per_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
        entry["prompt"] += prompt_tokens
        entry["completion"] += completion_tokens
        entry["calls"] += 1

    def record_context_usage(self, current: int, max_tokens: int) -> None:
        """Record the latest context-window utilization.

        Called by ``ContextManagerCapability.on_usage_update`` on each token
        update. Keeps the most-recent snapshot (the capability fires repeatedly
        as the conversation grows).
        """
        self.context_tokens = current
        self.context_budget = max_tokens

    @property
    def avg_ttft(self) -> float:
        """Average time-to-first-token across all recorded calls."""
        return self.total_ttft_s / self.call_count if self.call_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki.metadata storage."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "call_count": self.call_count,
            "avg_ttft": round(self.avg_ttft, 2),
            "per_model": dict(self.per_model),
            "context_tokens": self.context_tokens,
            "context_budget": self.context_budget,
        }
