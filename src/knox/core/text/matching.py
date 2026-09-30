"""Single source of truth for string similarity and semantic matching.

Every place in the codebase that compares "did this text mean that text"
routes through this module — concept merge, wikilink alias resolution,
lint remediation matching, and agentic generation coverage credit.

Two tiers:

- **T2 — string similarity** (:func:`similarity`, :func:`best_match`):
  rapidfuzz token-set ratio. Deterministic, fast, no model needed.
- **T3 — semantic matching** (:class:`SemanticMatcher`): embedding
  cosine via the zvec embedder. Used as a *conditional* fallback when
  string similarity is below threshold and an embedding model is
  configured. Skipped silently when no embedder is available — T2
  alone always bounds the caller's work.

High-volume *content* dedup (whole-document fingerprints) is a different
problem and stays with :class:`knox.preprocessing.dedup.minhash_matcher.MinhashMatcher`
(datasketch MinHash+LSH). Reference/topic embedding dedup reuses
:class:`SemanticMatcher` so there is exactly one embedding-comparison path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rapidfuzz import fuzz

from knox.logging import get_logger

#: Default acceptance threshold for :func:`similarity` (rapidfuzz scale 0–100).
#: 85 keeps "Artificial Mouth" vs "Artificial Ear" (≈83) out while clear
#: paraphrases (≥90) pass. Residual borderline cases (76–84) are the T3
#: semantic tier's job.
DEFAULT_SIMILARITY_THRESHOLD = 85.0

_NON_ALNUM_RE = __import__("re").compile(r"[^a-z0-9]+")

logger = get_logger(__name__)


@dataclass
class SemanticMatcher:
    """Embedding-based matcher over a fixed candidate set (T3).

    Candidates are embedded once at construction; queries are embedded
    per call and cosine-compared. Construction fails softly: pass
    ``embedder=None`` (or let the factory catch provider errors) and
    :attr:`available` is ``False`` — every query then returns ``None``
    and callers fall through to their T2 result.
    """

    embedder: Embedder | None = None
    #: candidate key → embedded text
    candidates: dict[str, str] = field(default_factory=dict)
    #: cosine acceptance threshold
    threshold: float = 0.55
    _vectors: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.embedder is None or not self.candidates:
            return
        for key, text in self.candidates.items():
            try:
                self._vectors[key] = self.embedder.embed(text)
            except Exception as exc:  # ponytail: provider errors vary widely
                logger.debug("SemanticMatcher: embed failed for %r: %s", key, exc)
                self.embedder = None
                self._vectors.clear()
                return

    @property
    def available(self) -> bool:
        """True when the candidate vectors were built successfully."""
        return bool(self._vectors)

    def match(self, query: str) -> tuple[str, float] | None:
        """Best candidate for *query* by cosine, or ``None``."""
        if not self.available or not query or self.embedder is None:
            return None
        try:
            q = self.embedder.embed(query)
        except Exception as exc:
            logger.debug("SemanticMatcher: query embed failed: %s", exc)
            return None
        best_key: str | None = None
        best_score = self.threshold
        for key, vec in self._vectors.items():
            score = cosine(q, vec)
            if score >= best_score:
                best_score = score
                best_key = key
        if best_key is None:
            return None
        return best_key, best_score


class Embedder(Protocol):
    """Minimal embedder surface — satisfied by zvec's ``build_embedder``."""

    def embed(self, text: str) -> list[float]: ...


def best_match(
    target: str,
    candidates: dict[str, str] | set[str] | list[str],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[str, float] | None:
    """Best candidate for *target* with :func:`similarity` ≥ *threshold*.

    ``candidates`` maps keys to comparable text (or, for set/list input,
    the text *is* the key). Returns ``(key, score)`` or ``None`` when no
    candidate clears the threshold. Exact case-insensitive equality wins
    outright — cheap short-circuit before fuzzy scanning.
    """
    if not target:
        return None
    items: list[tuple[str, str]]
    items = list(candidates.items()) if isinstance(candidates, dict) else [(c, c) for c in candidates]
    if not items:
        return None

    target_lower = target.strip().lower()
    for key, text in items:
        if text.strip().lower() == target_lower:
            return key, 100.0

    best_key: str | None = None
    best_score = threshold
    for key, text in items:
        score = similarity(target, text)
        if score >= best_score:
            best_score = score
            best_key = key
    if best_key is None:
        return None
    return best_key, best_score


def similarity(a: str, b: str) -> float:
    """Normalized similarity between two strings (0–100).

    rapidfuzz token ratios on punctuation-normalized text. Robust against
    token order, punctuation variants (``P.10/G.100`` ≡ ``P 10 G 100``),
    and extra tokens ("... (MOS)" vs "...") — the cases plain Jaccard and
    edit distance handle poorly.
    """
    if not a or not b:
        return 0.0
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return float(max(fuzz.token_set_ratio(na, nb), fuzz.token_ratio(na, nb)))


def _norm(s: str) -> str:
    """Lowercase and collapse punctuation/spacing (``P.10/G.100`` → ``p 10 g 100``)."""
    return _NON_ALNUM_RE.sub(" ", s.lower()).strip()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns ``0.0`` when either vector has zero norm (e.g. an all-zero
    embedding).

    Raises:
        ValueError: when either vector is empty or the lengths differ.
            Embeddings produced by one model always share a dimension, so a
            mismatch is a bug — ``zip``'s silent truncation would otherwise
            return a plausible-looking but wrong score.
    """
    if not a or len(a) != len(b):
        raise ValueError(f"cosine requires two equal-length non-empty vectors (got {len(a)} and {len(b)})")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
