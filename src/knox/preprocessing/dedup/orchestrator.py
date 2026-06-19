"""Orchestrator — composes multiple matchers, produces verdicts, manages store.

This is the main entry point for content-level duplicate detection. It:

1. Loads the persistent fingerprint store (with version-stamp validation).
2. For each incoming document, runs all configured matchers against the
   full corpus (existing + other incoming docs in the same batch).
3. Combines scores (max across matchers), produces a verdict.
4. Updates closest-match metadata for both new and existing documents.
5. Saves the updated store.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knox.core.paths import VaultPaths
from knox.preprocessing.dedup.base import ContentMatcher, DedupVerdict, classify_score
from knox.preprocessing.dedup.embedding_matcher import EmbeddingMatcher
from knox.preprocessing.dedup.minhash_matcher import MinhashMatcher
from knox.preprocessing.dedup.store import DedupStore

ProgressFn = Callable[[int, int], None]  # (completed, total)

logger = logging.getLogger(__name__)


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
        results = orch.process_batch(
            [("doc-1", "text of doc 1"), ("doc-2", "text of doc 2")]
        )
        for slug, result in results.items():
            print(slug, result.verdict)
    """

    def __init__(
        self,
        base_dir: Path,
        config: dict[str, Any] | None = None,
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

    def process_batch(  # noqa: PLR0915 — multi-phase batch processing
        self,
        documents: list[tuple[str, str]],
        progress_fn: ProgressFn | None = None,
    ) -> dict[str, DedupResult]:
        """Run dedup on a batch of ``(slug, text)`` pairs.

        Compares each document against all existing fingerprints *and*
        against other incoming documents in the same batch.

        Args:
            documents: List of ``(slug, text)`` pairs for the current ingest batch.
            progress_fn: Optional ``(completed, total)`` callback invoked after
                each document is compared against the full corpus. Total is
                ``N * (N - 1) / 2 + N * M`` where N = batch size, M = existing corpus.

        Returns:
            A dict mapping each slug to its :class:`DedupResult`.
        """
        if not self.enabled:
            logger.debug("Dedup disabled — skipping")
            return {slug: DedupResult(DedupVerdict.DIFFERENT, None, 0.0, "") for slug, _ in documents}

        self._store.load()

        if not self._store.is_fresh():
            logger.info("Dedup store stamp mismatch — rebuilding fingerprints from cache")
            self._store = self._build_store(recreate=True)
            # TODO: re-fingerprint cached parsed docs from .knox/cache/

        results: dict[str, DedupResult] = {}
        new_slugs: list[str] = []

        # Phase 1 — fingerprint all incoming docs
        slug_to_fp: dict[str, dict[str, Any]] = {}
        for slug, text in documents:
            fps: dict[str, Any] = {}
            for matcher in self._matchers:
                try:
                    fps[matcher.name] = matcher.fingerprint(text)
                except Exception:
                    logger.exception("Fingerprint failed for %s via %s", slug, matcher.name)
                    fps[matcher.name] = None
            slug_to_fp[slug] = fps
            new_slugs.append(slug)

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

            fps = slug_to_fp[slug]

            for candidate_slug in list(self._store.slugs) + new_slugs:
                if candidate_slug == slug:
                    continue

                matcher_scores: dict[str, float] = {}
                for matcher in self._matchers:
                    fp_a = fps.get(matcher.name)
                    fp_b = self._get_fingerprint(candidate_slug, matcher.name, slug_to_fp, matcher)
                    if fp_a is not None and fp_b is not None:
                        try:
                            matcher_scores[matcher.name] = matcher.score(fp_a, fp_b)
                        except Exception:
                            logger.exception("Score failed for %s vs %s via %s", slug, candidate_slug, matcher.name)
                            matcher_scores[matcher.name] = 0.0

                combined = max(matcher_scores.values()) if matcher_scores else 0.0
                if combined > best_score:
                    best_score = combined
                    best_match = candidate_slug
                    best_method = max(matcher_scores, key=matcher_scores.get) if matcher_scores else ""
                    scores = matcher_scores

            # Update progress: (i+1) * M + (i+1) * i / 2 comparisons done so far
            i = new_slugs.index(slug)
            done = (i + 1) * m + (i + 1) * i // 2
            if progress_fn is not None:
                progress_fn(min(done, total_comparisons), total_comparisons)

            verdict = classify_score(
                best_score,
                threshold_identical=float(self._config.get("dedup.threshold_identical", 0.95)),
                threshold_similar=float(self._config.get("dedup.threshold_similar", 0.80)),
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
            entry: dict[str, Any] = {
                "verdict": result.verdict.value,
                "closest_match": best_match,
                "closest_score": best_score,
                "closest_method": best_method,
                "embedding_score": scores.get("embedding"),
                "minhash_score": scores.get("minhash"),
                "embedding": (
                    self._matcher_by_name["embedding"].serialize(fps.get("embedding"))
                    if fps.get("embedding") is not None and "embedding" in self._matcher_by_name
                    else None
                ),
                "minhash": (
                    self._matcher_by_name["minhash"].serialize(fps.get("minhash"))
                    if fps.get("minhash") is not None and "minhash" in self._matcher_by_name
                    else None
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
        logger.info("Dedup processed %d documents", len(documents))
        return results

    # ── Internals ────────────────────────────────────────────────────────

    def _build_store(self, recreate: bool = False) -> DedupStore:
        paths = VaultPaths(self._base_dir)
        store_path = paths.dedup_dir / "fingerprints.json"
        return DedupStore(store_path, config=self._config)

    def _build_matchers(self) -> tuple[list[ContentMatcher], dict[str, ContentMatcher]]:
        matchers_str = self._config.get("dedup.matchers", "embedding,minhash")
        names = [m.strip() for m in matchers_str.split(",") if m.strip()]
        matchers: list[ContentMatcher] = []
        by_name: dict[str, ContentMatcher] = {}

        # Resolve the effective embedding model
        model = self._config.get("dedup.model", "") or self._config.get("compile.embedding.model", "") or "all-MiniLM-L6-v2"
        dim = int(self._config.get("compile.embedding.dim", 384))
        lsh_threshold = float(self._config.get("dedup.minhash_lsh_threshold", 0.8))

        for name in names:
            if name == "embedding":
                m = EmbeddingMatcher(model_name=model, dim=dim)
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
        slug_to_fp: dict[str, dict[str, Any]],
        matcher: ContentMatcher,
    ) -> Any:
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

    @staticmethod
    def _serialize_fp(fps: dict[str, Any], name: str) -> list | None:
        fp = fps.get(name)
        if fp is not None:
            return fp  # embedding vectors are already lists; MinHash handled by caller
        return None
