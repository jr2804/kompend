"""Orchestrator — composes multiple matchers, produces verdicts, manages store.

This is the main entry point for content-level duplicate detection. It:

1. Loads the persistent fingerprint store (with version-stamp validation).
2. For each incoming document, runs all configured matchers against the
   full corpus (existing + other incoming docs in the same batch).
3. Combines scores (arithmetic mean across matchers), produces a verdict.
4. Updates closest-match metadata for both new and existing documents.
5. Saves the updated store.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasketch import MinHash

from kompend.logging import get_logger
from kompend.preprocessing.dedup.base import ContentMatcher, DedupVerdict, classify_score
from kompend.preprocessing.dedup.embedding_matcher import EmbeddingMatcher
from kompend.preprocessing.dedup.minhash_matcher import MinhashMatcher
from kompend.preprocessing.dedup.store import DedupStore, StoreEntry
from kompend.rag.zvec_store import make_provider
from kompend.types import JSONDict

ProgressFn = Callable[[int, int], None]  # (completed, total)

logger = get_logger(__name__)


# Matcher-specific fingerprint values crossing the orchestrator's dynamic
# name-based pairing (see ContentMatcher.score_for for the runtime contract).
Fingerprint = list[float] | MinHash
AnyMatcher = ContentMatcher[list[float]] | ContentMatcher[MinHash]


@dataclass
class DedupResult:
    """Outcome of duplicate detection for a single document."""

    verdict: DedupVerdict
    closest_match: str | None
    closest_score: float
    closest_method: str
    embedding_score: float | None = None
    minhash_score: float | None = None


class DedupOrchestrator:
    """Orchestrates content-level duplicate detection.

    Usage::

        orch = DedupOrchestrator(base_dir, config)
        results = orch.process_batch([("doc-1", "text of doc 1"), ("doc-2", "text of doc 2")])
        for slug, result in results.items():
            print(slug, result.verdict)
    """

    def __init__(
        self,
        base_dir: Path,
        config: JSONDict | None = None,
    ) -> None:
        self._base_dir = base_dir
        self._config = config or {}
        self._store = self._build_store()
        self._matchers, self._matcher_by_name = self._build_matchers()

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("dedup.enabled", False))

    @property
    def store(self) -> DedupStore:
        return self._store

    def fingerprint(self, slug: str, text: str) -> dict[str, Fingerprint]:
        """Compute all configured fingerprints for a single document.

        Call this **during extraction** (per-document). Embedding network
        calls happen here — if the embedding model is misconfigured or the
        provider is unreachable, the exception propagates immediately so
        the caller can abort with a clear per-document error.

        Args:
            slug: Document identifier (typically the file stem).
            text: Parsed document text.

        Returns:
            Dict mapping matcher name → fingerprint. Empty when dedup is
            disabled.
        """
        if not self.enabled:
            return {}
        fps: dict[str, Fingerprint] = {}
        for matcher in self._matchers:
            fps[matcher.name] = matcher.fingerprint(text)
        return fps

    def compare(  # noqa: PLR0915 — multi-phase comparison flow
        self,
        fingerprints: dict[str, dict[str, Fingerprint]],
        progress_fn: ProgressFn | None = None,
    ) -> dict[str, DedupResult]:
        """Compare pre-computed fingerprints and produce verdicts.

        Call this **at end of ingest**. Pure CPU — no network calls. Each
        document is compared against the persistent corpus and against
        other documents in the same batch.

        Args:
            fingerprints: Dict mapping slug → ``{matcher_name: fingerprint}``.
                Produced by calling :meth:`fingerprint` per document during
                extraction.
            progress_fn: Optional ``(completed, total)`` callback.

        Returns:
            A dict mapping each slug to its :class:`DedupResult`.
        """
        if not self.enabled:
            logger.debug("Dedup disabled — skipping")
            return {slug: DedupResult(DedupVerdict.DIFFERENT, None, 0.0, "") for slug in fingerprints}

        self._store.load()

        if not self._store.is_fresh():
            logger.info("Dedup store stamp mismatch — rebuilding fingerprints from cache")
            self._store = self._build_store(recreate=True)
            # Cross-batch dedup history is lost on stamp mismatch (store schema or
            # config changed). Only the current batch's docs are compared against
            # each other. Re-fingerprinting cached parse outputs from .kompend/cache/
            # would restore cross-batch comparison, but that requires loading each
            # cached doc's text — deferred until dedup store invalidation becomes
            # frequent enough to matter (currently rare: only on _STORE_VERSION bump).

        results: dict[str, DedupResult] = {}
        new_slugs = list(fingerprints.keys())

        # Phase 2 — compare each new doc against corpus + other new docs
        corpus_slugs = list(self._store.slugs)
        n = len(new_slugs)
        m = len(corpus_slugs)
        total_comparisons = n * (n - 1) // 2 + n * m

        for slug in new_slugs:
            best_score = -1.0
            best_match: str | None = None
            best_method = ""
            scores: dict[str, float] = {}

            fps = fingerprints[slug]

            for candidate_slug in list(self._store.slugs) + new_slugs:
                if candidate_slug == slug:
                    continue

                matcher_scores: dict[str, float] = {}
                for matcher in self._matchers:
                    fp_a = fps.get(matcher.name)
                    fp_b = self._get_fingerprint(candidate_slug, matcher.name, fingerprints, matcher)
                    if fp_a is not None and fp_b is not None:
                        matcher_scores[matcher.name] = matcher.score_for(fp_a, fp_b)

                combined = sum(matcher_scores.values()) / len(matcher_scores) if matcher_scores else 0.0
                if combined > best_score:
                    best_score = combined
                    best_match = candidate_slug
                    best_method = max(matcher_scores, key=lambda m: matcher_scores[m], default="")
                    scores = matcher_scores

            # Update progress: (i+1) * M + (i+1) * i / 2 comparisons done so far
            i = new_slugs.index(slug)
            done = (i + 1) * m + (i + 1) * i // 2
            if progress_fn is not None:
                progress_fn(min(done, total_comparisons), total_comparisons)

            verdict = classify_score(
                best_score,
                threshold_identical=_cfg_float(self._config, "dedup.threshold_identical", 0.95),
                threshold_similar=_cfg_float(self._config, "dedup.threshold_similar", 0.80),
            )

            result = DedupResult(
                verdict=verdict,
                closest_match=best_match,
                closest_score=best_score,
                closest_method=best_method,
                embedding_score=scores.get("embedding"),
                minhash_score=scores.get("minhash"),
            )
            results[slug] = result

            # Store this doc's fingerprints + verdict
            entry: StoreEntry = {
                "verdict": result.verdict.value,
                "closest_match": best_match,
                "closest_score": best_score,
                "closest_method": best_method,
                "embedding_score": scores.get("embedding"),
                "minhash_score": scores.get("minhash"),
                "embedding": (
                    self._matcher_by_name["embedding"].serialize_checked(fps.get("embedding"))
                    if "embedding" in self._matcher_by_name and "embedding" in fps
                    else None
                ),
                "minhash": (
                    self._matcher_by_name["minhash"].serialize_checked(fps.get("minhash")) if "minhash" in self._matcher_by_name and "minhash" in fps else None
                ),
            }
            self._store[slug] = entry

            # Phase 3 — propagate closest-match updates to existing docs
            if best_match is not None and best_score >= 0:
                self._store.update_closest(
                    best_match,
                    closest_match=slug,
                    closest_score=best_score,
                    closest_method=best_method,
                    embedding_score=scores.get("embedding"),
                    minhash_score=scores.get("minhash"),
                )

        self._store.save()
        logger.info("Dedup compared %d documents", len(fingerprints))
        return results

    def process_batch(
        self,
        documents: list[tuple[str, str]],
        progress_fn: ProgressFn | None = None,
    ) -> dict[str, DedupResult]:
        """Convenience: fingerprint all docs, then compare.

        For the preferred incremental pattern, call :meth:`fingerprint`
        per document during extraction, then :meth:`compare` at end of
        ingest. This makes embedding errors attributable to the specific
        document that caused them.
        """
        fingerprints: dict[str, dict[str, Any]] = {}
        for slug, text in documents:
            fingerprints[slug] = self.fingerprint(slug, text)
        return self.compare(fingerprints, progress_fn=progress_fn)

    # ── Internals ────────────────────────────────────────────────────────

    def _build_store(self, recreate: bool = False) -> DedupStore:
        # The legacy path argument is now only used as a handle to the vault;
        # the dedup rows live in the single store keyed by vault root.
        return DedupStore(self._base_dir, config=self._config)

    def _build_matchers(self) -> tuple[list[AnyMatcher], dict[str, AnyMatcher]]:
        matchers_cfg = self._config.get("dedup.matchers", "embedding,minhash")
        if not isinstance(matchers_cfg, str):
            raise TypeError(f"Config key 'dedup.matchers' must be a string, got {type(matchers_cfg).__name__}")
        names = [m.strip() for m in matchers_cfg.split(",") if m.strip()]
        matchers: list[AnyMatcher] = []
        by_name: dict[str, AnyMatcher] = {}

        # Resolve the effective embedding model.
        # Empty string "" and None are both treated as "not set" (falsy),
        # so `or` chains the fallback: dedup.model → compile.embedding.model → built-in.
        raw_model = self._config.get("dedup.model", "") or self._config.get("compile.embedding.model", "")
        model = raw_model if isinstance(raw_model, str) else ""
        lsh_threshold = _cfg_float(self._config, "dedup.minhash_lsh_threshold", 0.8)

        for name in names:
            if name == "embedding":
                if not model:
                    logger.warning("dedup embedding matcher requested but no embedding model configured — skipping")
                    continue
                # Reuse the canonical EmbeddingProvider factory — it requires
                # an explicit provider/model prefix (no local fallback). The
                # dimension is detected from the model at runtime (provider.dim),
                # matching the vectors the provider returns.
                provider = make_provider(model)
                m = EmbeddingMatcher(model_name=model, dim=provider.dim, provider=provider)
                matchers.append(m)
                by_name[name] = m
            elif name == "minhash":
                m = MinhashMatcher(lsh_threshold=lsh_threshold)
                matchers.append(m)
                by_name[name] = m
            else:
                logger.warning("Unknown dedup matcher: %s", name)

        if not matchers:
            logger.warning("No dedup matchers configured — dedup will be a no-op")

        return matchers, by_name

    def _get_fingerprint(
        self,
        slug: str,
        matcher_name: str,
        slug_to_fp: dict[str, dict[str, Fingerprint]],
        matcher: AnyMatcher,
    ) -> Fingerprint | None:
        """Get a fingerprint for *slug* — from batch cache or persistent store."""
        # First check the current batch (incoming docs)
        incoming = slug_to_fp.get(slug, {})
        fp = incoming.get(matcher_name)
        if fp is not None:
            return fp

        # Then check the persistent store
        stored = self._store.get(slug)
        if stored is not None:
            raw = stored.get(matcher_name)
            if raw is not None:
                try:
                    return matcher.deserialize(raw)
                except Exception:
                    logger.exception("Failed to deserialize %s fingerprint for %s", matcher_name, slug)
        return None


def _cfg_float(config: JSONDict, key: str, default: float) -> float:
    """Read a numeric config value, accepting int/float/numeric string."""
    raw = config.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise TypeError(f"Config key {key!r} must be numeric, got {type(raw).__name__}")
    return float(raw)
