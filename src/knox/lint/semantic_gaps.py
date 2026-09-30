"""Semantic source-coverage verification for gap analysis.

For each "Missing topic" recommendation, checks whether the **sources**
already discuss the topic — using the embedding model to compare the topic
against source-chunk texts from the ``chunk`` layer of the single
``.knox/knox.db`` cache store.

A topic that is semantically covered by a source is NOT a "add a source"
gap — it is a generation/linking gap: the material exists but no wiki page
was produced for it, and the right fix is ``knox lint remediate`` (which
generates the page from the cached extraction).

Chunk embeddings are cached in ``gap_chunk_vectors.json`` keyed by content
hash **and** the embedding model name (key is a flat dict under that path),
so repeated lint runs only embed new/changed chunks with the same model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from knox.core.cache import CacheStore
from knox.core.models import ChunkField, ChunkNodeField, ChunkTreeField, MetadataField
from knox.core.paths import safe_stem
from knox.core.text.embedding_text import embed_text
from knox.core.text.matching import cosine
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
    vector_cache_path = vp_paths / "cache" / _GAP_VECTOR_CACHE

    chunks = _load_leaf_chunks(base_dir)
    results = {t: TopicCoverage(topic=t, covered=False, best_score=0.0, best_source="") for t in topics}
    if not chunks:
        logger.warning(
            "Semantic gap check: no source chunks in the vault store chunk layer; run 'knox ingest' first for semantic coverage to apply",
        )
        return results

    cache = _load_vector_cache(vector_cache_path, embed_model)
    try:
        provider = make_provider(embed_model)
    except Exception:
        logger.warning("Semantic gap check: embedding provider unavailable for %s", embed_model, exc_info=True)
        return results

    # Embed chunks (cache misses only).
    vectors: list[list[float] | None] = []
    dirty = False
    try:
        for _stem, text, context in chunks:
            embed_input = embed_text(text, context)
            key = hashlib.sha256(embed_input.encode("utf-8")).hexdigest()[:16]
            vec = cache.get(key)
            if vec is None:
                vec = provider.embed(embed_input)
                cache[key] = vec
                dirty = True
            vectors.append(vec)
        topic_vecs = [provider.embed(t) for t in topics]
    except Exception:
        logger.warning("Semantic gap check: embedding failed — skipping coverage verification", exc_info=True)
        return results
    finally:
        if dirty:
            _save_vector_cache(vector_cache_path, cache, embed_model)

    for topic, tvec in zip(topics, topic_vecs, strict=False):
        best_score = 0.0
        best_source = ""
        for (stem, _text, _context), cvec in zip(chunks, vectors, strict=False):
            if cvec is None:
                continue
            score = cosine(tvec, cvec)
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


def _load_leaf_chunks(base_dir: Path) -> list[tuple[str, str, str]]:
    """Load ``(source_stem, text, contextual_context)`` for every leaf chunk.

    Reads the ``chunk`` layer from the single ``.knox/knox.db`` store (the
    former ``cache/chunk/<stem>-<hash8>.json`` files are gone). A missing store
    or unreadable rows yield fewer chunks, never an error.
    """
    chunks: list[tuple[str, str, str]] = []
    for value in CacheStore(base_dir).load_layer("chunk"):
        if not isinstance(value, dict):
            continue
        nodes = value.get(ChunkTreeField.NODES)
        if not isinstance(nodes, dict):
            continue
        source_file = str(value.get(ChunkTreeField.SOURCE_FILE) or "")
        stem = safe_stem(source_file) if source_file else ""
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            # Chunk text lives on node[CHUNK] (null for internal tree nodes,
            # whose content duplicates the leaves).
            chunk = node.get(ChunkNodeField.CHUNK)
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get(ChunkField.TEXT, "")).strip()
            if text:
                context = str(node.get(ChunkNodeField.CONTEXTUAL_CONTEXT, ""))
                chunks.append((stem, text, context))
    return chunks


def _load_vector_cache(cache_path: Path, embed_model: str) -> dict[str, list[float]]:
    """Load chunk vectors cached for *embed_model*.

    A cache written for a different embedding model — or a legacy model-less
    ``{key: vector}`` file — is treated as a full miss: vectors from different
    models are not comparable.
    """
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Gap vector cache unreadable — rebuilding: %s (%s)", cache_path, exc)
        return {}
    if not isinstance(raw, dict) or raw.get(MetadataField.MODEL) != embed_model:
        return {}
    vectors = raw.get("vectors")
    if not isinstance(vectors, dict):
        return {}
    return {str(key): [float(x) for x in vec] for key, vec in vectors.items() if isinstance(vec, list)}


def _save_vector_cache(cache_path: Path, vectors: dict[str, list[float]], embed_model: str) -> None:
    dim = len(next(iter(vectors.values()))) if vectors else 0
    payload = {MetadataField.MODEL: embed_model, "dim": dim, "vectors": vectors}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
