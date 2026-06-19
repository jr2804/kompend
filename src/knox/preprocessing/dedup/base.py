"""Abstract protocol for content-level similarity matchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any


class DedupVerdict(StrEnum):
    """Verdict for a single document after duplicate detection."""

    IDENTICAL = "identical"
    SIMILAR = "similar"
    DIFFERENT = "different"


class ContentMatcher(ABC):
    """Abstract base for a similarity matcher (embedding, MinHash, …).

    Each matcher produces a score in [0, 1] for a pair of documents,
    and maintains its own persistent fingerprints (vectors, signatures).
    """

    name: str = "base"

    @abstractmethod
    def fingerprint(self, text: str) -> Any:
        """Compute a fingerprint for a document's text content.

        The fingerprint type is matcher-specific — an embedding vector
        (``list[float]``), a MinHash signature, etc.
        """
        ...

    @abstractmethod
    def score(self, fp_a: Any, fp_b: Any) -> float:
        """Return a similarity score in [0, 1] between two fingerprints.

        1.0 = identical, 0.0 = completely different.
        """
        ...

    @abstractmethod
    def serialize(self, fingerprint: Any) -> list:
        """Convert a fingerprint to a JSON-serialisable list."""
        ...

    @abstractmethod
    def deserialize(self, data: list) -> Any:
        """Restore a fingerprint from a serialised list."""
        ...


def classify_score(
    score: float,
    threshold_identical: float = 0.95,
    threshold_similar: float = 0.80,
) -> DedupVerdict:
    """Map a combined similarity score to a verdict."""
    if score >= threshold_identical:
        return DedupVerdict.IDENTICAL
    if score >= threshold_similar:
        return DedupVerdict.SIMILAR
    return DedupVerdict.DIFFERENT
