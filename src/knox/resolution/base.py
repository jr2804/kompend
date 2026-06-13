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

    @staticmethod
    def enrich(resolved: ResolvedChunk) -> ResolvedChunk:
        """Enrich an already-resolved chunk.

        The default implementation returns the chunk unchanged. Resolvers
        that transform an existing :class:`ResolvedChunk` (e.g. image caption
        enrichment) should override this method so a :class:`ResolverChain`
        can accumulate their output instead of discarding earlier results.
        """
        return resolved
