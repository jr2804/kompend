"""Content-level duplicate detection for ingested sources."""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "base": ["ContentMatcher", "DedupVerdict"],
        "embedding_matcher": ["EmbeddingMatcher"],
        "minhash_matcher": ["MinhashMatcher"],
        "orchestrator": ["DedupOrchestrator"],
        "store": ["DedupStore"],
    },
)
