"""Semantic source-coverage verification for gap analysis.

For each "Missing topic" recommendation, checks whether the **sources**
already discuss the topic — using the embedding model to compare the topic
against source-chunk texts from the parse cache (``.knox/cache/chunk/``).

A topic that is semantically covered by a source is NOT a "add a source"
gap — it is a generation/linking gap: the material exists but no wiki page
was produced for it, and the right fix is ``knox lint remediate`` (which
generates the page from the cached extraction).

Chunk embeddings are cached in ``.knox/cache/gap_chunk_vectors.json`` keyed
by content hash, so repeated lint runs only embed new/changed chunks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from knox.core.paths import safe_stem
from knox.logging import get_logger
from knox.rag.zvec_store import make_provider

#: Cosine similarity above which a source chunk is considered to "cover" a
#: missing topic. Short topic names vs. long chunks rarely score extremely
#: high; 0.55 is conservative enough to avoid false "covered" verdicts while
#: catching genuine terminological matches.
SEMANTIC_COVER_THRESHOLD = 0.55

_GAP_VECTOR_CACHE = "gap_chunk_vectors.json"

logger = get_logger(__name__)


@dataclass
class TopicCoverage:
    """Result of the semantic source check for one topic."""

    topic: str
    covered: bool
    best_score: float
    best_source: str
    """Source file stem of the best-matching chunk ('' when uncovered)."""


def verify_topics(
    topics: list[str],
    base_dir: Path,
    embed_model: str,
    *,
    threshold: float = SEMANTIC_COVER_THRESHOLD,
) -> dict[str, TopicCoverage]:
    """Check each *topic* for semantic coverage in the source chunks.

    Returns a mapping topic → :class:`TopicCoverage`. Topics whose best
    chunk similarity reaches *threshold* are marked covered. Any failure
    (no chunks, embedding provider error) degrades to "not covered" for
    all topics — the check is strictly additive signal.
    """
    vp_paths = base_dir / ".knox"
    chunk_cache_dir = vp_paths / "cache" / "chunk"
    vector_cache_path = vp_paths / "cache" / _GAP_VECTOR_CACHE

    chunks = _load_leaf_chunks(chunk_cache_dir)
    results = {t: TopicCoverage(topic=t, covered=False, best_score=0.0, best_source="") for t in topics}
    if not chunks:
        logger.info("Semantic gap check: no source chunks found in %s", chunk_cache_dir)
        return results

    cache = _load_vector_cache(vector_cache_path)
    try:
        provider = make_provider(embed_model)
    except Exception:
        logger.warning("Semantic gap check: embedding provider unavailable for %s", embed_model, exc_info=True)
        return results

    # Embed chunks (cache misses only).
    vectors: list[list[float] | None] = []
    dirty = False
    try:
        for _stem, text in chunks:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            vec = cache.get(key)
            if vec is None:
                vec = provider.embed(text)
                cache[key] = vec
                dirty = True
            vectors.append(vec)
        topic_vecs = [provider.embed(t) for t in topics]
    except Exception:
        logger.warning("Semantic gap check: embedding failed — skipping coverage verification", exc_info=True)
        return results
    finally:
        if dirty:
            _save_vector_cache(vector_cache_path, cache)

    for topic, tvec in zip(topics, topic_vecs, strict=False):
        best_score = 0.0
        best_source = ""
        for (stem, _text), cvec in zip(chunks, vectors, strict=False):
            if cvec is None:
                continue
            score = _cosine(tvec, cvec)
            if score > best_score:
                best_score = score
                best_source = stem
        results[topic] = TopicCoverage(
            topic=topic,
            covered=best_score >= threshold,
            best_score=round(best_score, 4),
            best_source=best_source,
        )
    return results


def _load_leaf_chunks(chunk_cache_dir: Path) -> list[tuple[str, str]]:
    """Load ``(source_stem, text)`` for leaf chunks from the parse cache."""
    chunks: list[tuple[str, str]] = []
    if not chunk_cache_dir.is_dir():
        return chunks
    for cache_file in sorted(chunk_cache_dir.glob("*.json")):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        value = data.get("value") if isinstance(data, dict) else None
        nodes = value.get("nodes") if isinstance(value, dict) else None
        if not isinstance(nodes, dict):
            continue
        source_file = str(value.get("source_file", "")) or cache_file.stem
        stem = safe_stem(source_file) if source_file else cache_file.stem
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            # Chunk text lives on node["chunk"] (null for internal tree nodes,
            # whose content duplicates the leaves).
            chunk = node.get("chunk")
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text", "")).strip()
            if text:
                chunks.append((stem, text))
    return chunks


def _load_vector_cache(cache_path: Path) -> dict[str, list[float]]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Gap vector cache corrupt — rebuilding: %s (%s)", cache_path, exc)
    return {}


def _save_vector_cache(cache_path: Path, cache: dict[str, list[float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
