"""Abstract base for KNOX chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from knox.core.models import Chunk, ParsedDocument


class BaseChunker(ABC):
    """Chunker interface for KNOX Tier 2."""

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into semantic chunks."""
        ...
