"""Content-level duplicate detection for ingested sources."""

from __future__ import annotations

from knox.preprocessing.dedup.base import ContentMatcher, DedupVerdict
from knox.preprocessing.dedup.embedding_matcher import EmbeddingMatcher
from knox.preprocessing.dedup.minhash_matcher import MinhashMatcher
from knox.preprocessing.dedup.orchestrator import DedupOrchestrator
from knox.preprocessing.dedup.store import DedupStore

__all__ = [
    "ContentMatcher",
    "DedupOrchestrator",
    "DedupStore",
    "DedupVerdict",
    "EmbeddingMatcher",
    "MinhashMatcher",
]
