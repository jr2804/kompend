"""Abstract base for KNOX chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final

from knox.core.models import ChunkTree, ParsedDocument
from knox.llm.client import ContextWindow

# Absolute token reserve carved out of the context window before computing the
# effective chunk size.  Replaces an earlier `reserve_ratio` (20%) that wasted
# ~25K tokens on a 128K-context model and ~200K on a 1M-context model — even
# though real LLM-call overhead (system prompt + output budget) is bounded.
# Sized to cover ``gen_output_max`` (16K) is excessive for a chunker that only
# needs headroom for prompt/output framing; 8K is a balanced default that
# zeroes out on tiny 8K-ctx models (and floors to ``chunk_size_min``) while
# leaving the bulk of large windows available for chunk content.
_DEFAULT_RESERVE_TOKENS: Final[int] = 8192


class BaseChunker(ABC):
    """Chunker interface for KNOX Tier 2.

    Implementations split a ParsedDocument into a :class:`ChunkTree`,
    preserving structure (tables, figures, code blocks remain atomic).
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        reserve_tokens: int | None = None,
        context_window: ContextWindow | None = None,
        chunk_size_min: int = 2048,
        chunk_size_max: int = 131072,
        **kwargs: Any,
    ) -> None:
        self.max_tokens = max_tokens
        self._chunk_size_min = chunk_size_min
        self._chunk_size_max = chunk_size_max
        if context_window is not None:
            # Production path: real context window discovered from the LLM
            # client.  Apply the default absolute reserve unless the caller
            # overrides it (e.g. for benchmarking with a tight budget).
            effective_reserve = reserve_tokens if reserve_tokens is not None else _DEFAULT_RESERVE_TOKENS
            self.reserve_tokens = effective_reserve
            raw = max(0, context_window.max_input - effective_reserve)
            self._effective_max = max(self._chunk_size_min, min(raw, self._chunk_size_max))
        elif max_tokens is not None:
            # Test-only fallback: caller provides an explicit token budget.
            # Default reserve is 0 here so small synthetic ``max_tokens``
            # values (e.g. ``max_tokens=100``) are not clobbered by an
            # 8K production reserve.
            effective_reserve = reserve_tokens if reserve_tokens is not None else 0
            self.reserve_tokens = effective_reserve
            self._effective_max = max(0, max_tokens - effective_reserve)
        else:
            msg = "Either context_window or max_tokens must be provided"
            raise ValueError(msg)

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> ChunkTree:
        """Split a parsed document into a hierarchical chunk tree."""
        ...
