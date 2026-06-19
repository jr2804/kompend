"""MinHash-based content similarity matcher.

Uses datasketch to generate MinHash signatures and estimate Jaccard
similarity. Supports LSH indexing for sub-linear candidate lookup on
large corpora.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from datasketch import MinHash, MinHashLSH

from knox.preprocessing.dedup.base import ContentMatcher

logger = logging.getLogger(__name__)

# Default number of permutation hash functions for MinHash.
# 128 gives a good balance of accuracy vs. storage for corpus sizes
# up to ~100k documents.
_DEFAULT_NUM_PERM = 128

# Default shingle size (word-level n-grams).
_DEFAULT_NGRAM = 5


class MinhashMatcher(ContentMatcher):
    """Matcher that uses datasketch MinHash signatures + LSH indexing.

    Documents are tokenised into word-level n-grams and hashed into a
    fixed-size MinHash signature. Similarity is estimated via the Jaccard
    coefficient approximated by the signature overlap.
    """

    name = "minhash"

    def __init__(
        self,
        num_perm: int = _DEFAULT_NUM_PERM,
        ngram: int = _DEFAULT_NGRAM,
        *,
        lsh_threshold: float = 0.8,
    ) -> None:
        self._num_perm = num_perm
        self._ngram = ngram
        self._lsh_threshold = lsh_threshold
        self._lsh: MinHashLSH | None = None
        self._slugs_in_lsh: set[str] = set()

    # ── LSH management ───────────────────────────────────────────────────

    @property
    def lsh_threshold(self) -> float:
        return self._lsh_threshold

    def init_lsh(self, storage_config: dict[str, Any] | None = None) -> None:
        """Initialise (or re-initialise) the LSH index.

        Call once before feeding fingerprints into the index. When
        *storage_config* is provided, uses a persistent LSH (e.g.
        ``redis`` or a file-backed store); otherwise in-memory.
        """
        params: dict[str, Any] = {
            "threshold": self._lsh_threshold,
            "num_perm": self._num_perm,
        }
        if storage_config:
            params.update(storage_config)
        self._lsh = MinHashLSH(**params)
        self._slugs_in_lsh = set()

    def lsh_insert(self, slug: str, minhash: MinHash) -> None:
        """Insert a MinHash signature into the LSH index."""
        if self._lsh is not None:
            self._lsh.insert(slug, minhash)
            self._slugs_in_lsh.add(slug)

    def lsh_query(self, minhash: MinHash) -> list[str]:
        """Return candidate slugs whose signatures are near *minhash*.

        Returns an empty list if no LSH index is active.
        """
        if self._lsh is None:
            return []
        return self._lsh.query(minhash)

    # ── ContentMatcher implementation ────────────────────────────────────

    def fingerprint(self, text: str) -> MinHash:
        """Build a MinHash signature from *text*.

        Tokenises by lowercasing, then extracting word-level n-grams.
        """
        m = MinHash(num_perm=self._num_perm)
        if not text or not text.strip():
            return m

        tokens = self._tokenise(text)
        for ngram in self._ngrams(tokens, self._ngram):
            m.update(ngram.encode("utf-8"))
        return m

    def score(self, fp_a: MinHash, fp_b: MinHash) -> float:  # noqa: PLR6301 — overrides ABC
        """Estimate Jaccard similarity between two MinHash signatures."""
        return float(fp_a.jaccard(fp_b))

    def serialize(self, fingerprint: MinHash) -> list[int]:  # noqa: PLR6301 — overrides ABC
        """Export a MinHash signature as a list of hash values."""
        return [int(h) for h in fingerprint.hashvalues]

    def deserialize(self, data: list) -> MinHash:
        """Restore a MinHash signature from a serialised list."""
        m = MinHash(num_perm=self._num_perm)
        m.hashvalues = [self._to_uint64(v) for v in data]
        return m

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Tokenise *text* into lowercased word tokens.

        Keeps only alphanumeric sequences (or words with hyphens/apostrophes
        as a single token). Strips short tokens (< 2 chars).
        """
        tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'\-]*[a-zA-Z0-9]|[a-zA-Z0-9]", text.lower())
        return [t for t in tokens if len(t) >= 2]

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[str]:
        """Generate word-level n-grams from a token list."""
        if len(tokens) < n:
            return [" ".join(tokens)]
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def _to_uint64(value: Any) -> int:
        """Ensure an integer fits in uint64 (datasketch uses uint64)."""
        v = int(value)
        # Mask to 64 bits if negative
        if v < 0:
            v = v & 0xFFFFFFFFFFFFFFFF
        return v

    def __repr__(self) -> str:
        return f"MinhashMatcher(num_perm={self._num_perm}, ngram={self._ngram})"
