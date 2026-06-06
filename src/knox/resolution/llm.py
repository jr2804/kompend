"""LLM-based entity/relation resolver."""

from __future__ import annotations

from knox.core.models import Chunk, ResolvedChunk
from knox.resolution.base import BaseResolver


class LLMResolver(BaseResolver):
    """Stub: Uses LLM to extract entities and resolve references."""

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        return ResolvedChunk(
            chunk=chunk,
            entities=[],
            relations=[],
            resolved_references=[],
        )
