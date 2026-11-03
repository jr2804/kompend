"""Abstract base for kompend resolvers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kompend.agents.failure import UnitFailureLog
from kompend.core.models import Chunk, ResolvedChunk


class BaseResolver(ABC):
    """Resolver interface for kompend Tier 3."""

    #: The failures of the units this resolver processed, when it calls a
    #: model. ``None`` for resolvers that do not. The resolve stage checks
    #: it when the batch is done, so a batch that was mostly failures stops
    #: instead of being published as a smaller but complete result.
    @property
    def failures(self) -> UnitFailureLog | None:
        """The failure log of this resolver, when it keeps one."""
        return None

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
