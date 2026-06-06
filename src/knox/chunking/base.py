"""Abstract base for KNOX chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from knox.core.models import Chunk, ParsedDocument


class BaseChunker(ABC):
    """Chunker interface for KNOX Tier 2.

    Implementations split a ParsedDocument into a list of Chunks,
    preserving structure (tables, figures, code blocks remain atomic).
    """

    def __init__(self, max_tokens: int = 4096, reserve_ratio: float = 0.2, **kwargs: Any) -> None:
        self.max_tokens = max_tokens
        self.reserve_ratio = reserve_ratio
        self._effective_max = int(max_tokens * (1 - reserve_ratio))

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into semantic chunks."""
        ...
