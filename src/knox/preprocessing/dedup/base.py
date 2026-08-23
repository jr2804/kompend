"""Abstract protocol for content-level similarity matchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar, TypeVar, cast

from knox.types import JSONValue

FpT = TypeVar("FpT")


class DedupVerdict(StrEnum):
    """Verdict for a single document after duplicate detection."""

    IDENTICAL = "identical"
    SIMILAR = "similar"
    DIFFERENT = "different"


class ContentMatcher[FpT](ABC):
    """Abstract base for a similarity matcher (embedding, MinHash, …).

    Each matcher produces a score in [0, 1] for a pair of documents,
    and maintains its own persistent fingerprints (vectors, signatures).
    """

    name: str = "base"

    #: Runtime type of this matcher's fingerprints. Used by :meth:`score_for`
    #: to validate dynamically paired fingerprints before scoring.
    fp_type: ClassVar[type] = object

    def score_for(self, fp_a: object, fp_b: object) -> float:
        """Score fingerprints paired to this matcher by name at runtime.

        Fingerprints reach the orchestrator through cache serialization,
        so their static type is lost in transit; validate against
        :attr:`fp_type` instead of trusting the round-trip.
        """
        if not (isinstance(fp_a, self.fp_type) and isinstance(fp_b, self.fp_type)):
            raise TypeError(
                f"{self.name} matcher received fingerprint of unexpected type: {type(fp_a).__name__} / {type(fp_b).__name__}, expected {self.fp_type.__name__}"
            )
        return self.score(cast("FpT", fp_a), cast("FpT", fp_b))

    @abstractmethod
    def fingerprint(self, text: str) -> FpT:
        """Compute a fingerprint for a document's text content.

        The fingerprint type is matcher-specific — an embedding vector
        (``list[float]``), a MinHash signature, etc.
        """
        ...

    @abstractmethod
    def score(self, fp_a: FpT, fp_b: FpT) -> float:
        """Return a similarity score in [0, 1] between two fingerprints.

        1.0 = identical, 0.0 = completely different.
        """
        ...

    def serialize_checked(self, fingerprint: object) -> list[JSONValue]:
        """Serialize a dynamically paired fingerprint (see :meth:`score_for`)."""
        if not isinstance(fingerprint, self.fp_type):
            raise TypeError(f"{self.name} matcher received fingerprint of unexpected type: {type(fingerprint).__name__}, expected {self.fp_type.__name__}")
        return list(self.serialize(cast("FpT", fingerprint)))

    @abstractmethod
    def serialize(self, fingerprint: FpT) -> Sequence[JSONValue]:
        """Convert a fingerprint to a JSON-serialisable list."""
        ...

    @abstractmethod
    def deserialize(self, data: list[JSONValue]) -> FpT:
        """Restore a fingerprint from a serialised list."""
        ...


def require_json_number(value: JSONValue) -> float:
    """Coerce a decoded JSON scalar to float, rejecting non-numerics."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric fingerprint value, got {type(value).__name__}")
    return float(value)


def require_json_int(value: JSONValue) -> int:
    """Coerce a decoded JSON scalar to int, rejecting non-integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer fingerprint value, got {type(value).__name__}")
    return value


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
