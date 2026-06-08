"""Resolver chain that applies multiple resolvers sequentially.

Each resolver in the chain receives the output of the previous one.
If a resolver returns ``None`` or raises, the chain short-circuits and
returns the last successful result.
"""

from __future__ import annotations

import logging

from knox.core.models import Chunk, ResolvedChunk
from knox.resolution.base import BaseResolver

logger = logging.getLogger(__name__)


class ResolverChain(BaseResolver):
    """Apply a sequence of :class:`BaseResolver` instances to each chunk.

    The first resolver converts a :class:`Chunk` into a
    :class:`ResolvedChunk`. Subsequent resolvers receive and return
    :class:`ResolvedChunk` instances, enriching them progressively.
    """

    def __init__(self, resolvers: list[BaseResolver]) -> None:
        if not resolvers:
            msg = "ResolverChain requires at least one resolver"
            raise ValueError(msg)
        self._resolvers = resolvers

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        """Run all resolvers in sequence, returning the final result."""
        result: ResolvedChunk = self._resolvers[0].resolve(chunk)
        for resolver in self._resolvers[1:]:
            try:
                enriched = resolver.resolve(chunk)
                if enriched is not None:
                    result = enriched
            except Exception:
                logger.warning("Resolver %s failed, keeping previous result", resolver.__class__.__name__)
        return result
