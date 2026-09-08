"""LLM usage tracking — accumulates token counts across LLM calls.

Single source of truth for both token accounting and throughput (tok/s):
all rate arithmetic lives here, and every consumer (compile progress,
log.md lines, doctor probes) reads from :class:`UsageTracker` /
:func:`tokens_per_second`. Token counts are the *real* values reported
by the provider in the response usage — never character-based estimates.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


def tokens_per_second(tokens: int, elapsed_s: float) -> float:
    """Throughput for one completed interval (0.0 when nothing/zero time)."""
    if tokens <= 0 or elapsed_s <= 0.0:
        return 0.0
    return tokens / elapsed_s


@dataclass
class UsageTracker:
    """Accumulates LLM usage statistics across complete() calls.

    Each :class:`LLMClient` holds one instance. After pipeline runs,
    the caller reads ``client.usage`` to populate the wiki log.

    Also keeps a timestamped event log of completion tokens (stamped at
    response-completion time) so throughput can be reported as an
    overall average and as a rolling average across the last N output
    tokens (:attr:`throughput_window_tokens`).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    total_ttft_s: float = 0.0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)
    # Latest context-window utilization reported by ContextManagerCapability.
    context_tokens: int = 0
    context_budget: int = 0
    #: Rolling-window size for :meth:`recent_rate`, in output tokens.
    throughput_window_tokens: int = 1000

    #: (monotonic timestamp, completion tokens) per finished call, oldest first.
    _events: deque[tuple[float, int]] = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, prompt_tokens: int, completion_tokens: int, model: str, ttft_s: float = 0.0) -> None:
        """Record one successful completion. Thread-safe."""
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.call_count += 1
            self.total_ttft_s += ttft_s
            entry = self.per_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
            entry["prompt"] += prompt_tokens
            entry["completion"] += completion_tokens
            entry["calls"] += 1
            if completion_tokens > 0:
                self._events.append((time.monotonic(), completion_tokens))

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

    def overall_rate(self) -> float:
        """Output tokens/second across all recorded calls (0.0 if none)."""
        with self._lock:
            if not self._events:
                return 0.0
            first, last = self._events[0][0], self._events[-1][0]
            tokens = sum(t for _, t in self._events)
        return tokens_per_second(tokens, last - first)

    def recent_rate(self, window_tokens: int | None = None) -> float:
        """Rolling output tokens/second across the last *window_tokens*.

        Walks events newest→oldest until the window's token budget is
        consumed (the crossing call counts fully). The interval starts at
        the completion of the event *before* the window's oldest member —
        the best available proxy for when that call started — and ends at
        the newest completion. Returns 0.0 when the window spans fewer
        than two completions (no measurable interval).
        """
        window = self.throughput_window_tokens if window_tokens is None else window_tokens
        with self._lock:
            if len(self._events) < 2:
                return 0.0
            events = list(self._events)
        accumulated = 0
        start_idx = 0  # oldest event index inside the window
        for i in range(len(events) - 1, -1, -1):
            accumulated += events[i][1]
            start_idx = i
            if accumulated >= window:
                break
        # Interval start: completion of the event preceding the window
        # (proxy for the window's first call having started after it).
        span_start = events[start_idx - 1][0] if start_idx > 0 else events[0][0]
        span = events[-1][0] - span_start
        return tokens_per_second(accumulated, span)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki.metadata storage."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "call_count": self.call_count,
            "avg_ttft": round(self.avg_ttft, 2),
            "overall_tok_s": round(self.overall_rate(), 2),
            "recent_tok_s": round(self.recent_rate(), 2),
            "per_model": dict(self.per_model),
            "context_tokens": self.context_tokens,
            "context_budget": self.context_budget,
        }
