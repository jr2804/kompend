"""Abstract base for KNOX resolvers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from knox.core.models import Chunk, ResolvedChunk


class BaseResolver(ABC):
    """Resolver interface for KNOX Tier 3."""

    @abstractmethod
    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        """Resolve entities, relations, and cross-references in a chunk."""
        ...
