"""The read path — vector and lexical SQL, the hit shape, and the fusion.

The queries are hand-written SQL: ``vec0`` (``MATCH`` + ``k``) and FTS5
(``MATCH`` + ``bm25()``) are virtual-table grammars SQLAlchemy Core cannot
express, so the query layer is SQL text by contract. The filters are rendered
by :mod:`kompend.rag.sqlite_vec.filters`.

Scores are higher-is-better: the vector path returns ``1 - distance`` (the
cosine similarity), the lexical path ``-bm25()``.
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from typing import Any, Final

import sqlite_vec

from kompend.rag.sqlite_vec.filters import filter_clause
from kompend.rag.sqlite_vec.schema import VEC_FTS, VEC_ITEMS
from kompend.rag.vector_store import VectorField

#: Reciprocal-rank-fusion constant: a document's contribution is
#: ``weight / (RRF_K + rank)``. 60 is the value the previous store's
#: reranker used.
RRF_K: Final[int] = 60

#: Candidate pool multiplier when filters are applied: the filters run over
#: the vector candidates, so a filtered query needs more than ``top_k``
#: candidates to still return ``top_k`` rows.
FILTER_OVERSAMPLE: Final[int] = 4

#: Lower bound for the candidate pool of a filtered query.
FILTER_POOL_MIN: Final[int] = 50

#: The word characters of a query — everything else is an FTS5 operator.
TERM_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


class Hit:
    """One search result, in the shape the retrievers consume."""

    __slots__ = ("content_preview", "id", "node_type", "score", "source_file", "title")

    def __init__(
        self,
        *,
        id: str,
        score: float,
        title: str = "",
        node_type: str = "",
        source_file: str = "",
        content_preview: str = "",
    ) -> None:
        self.id = id
        self.score = score
        self.title = title
        self.node_type = node_type
        self.source_file = source_file
        self.content_preview = content_preview


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    """Search-result dict — the contract the retrievers read."""
    return {
        "id": hit.id,
        "score": hit.score,
        VectorField.NODE_ID: hit.id,
        VectorField.TITLE: hit.title,
        VectorField.NODE_TYPE: hit.node_type,
        VectorField.SOURCE_FILE: hit.source_file,
        VectorField.CONTENT_PREVIEW: hit.content_preview,
    }


def vector_hits(
    conn: sqlite3.Connection,
    embedding: list[float],
    top_k: int,
    *,
    tags_all: Sequence[str] | None = None,
    tags_dnf: Sequence[Sequence[str]] | None = None,
    attributes_all: Sequence[str] | None = None,
    attribute_keys: Sequence[str] | None = None,
) -> list[Hit]:
    """Run the vector query and return the hits in rank order."""
    clause, params = filter_clause(
        "v",
        tags_all=tags_all,
        tags_dnf=tags_dnf,
        attributes_all=attributes_all,
        attribute_keys=attribute_keys,
    )
    pool = candidate_pool(top_k, filtered=bool(clause))
    sql = f"""
        SELECT v.id, v.title, v.node_type, v.source_file, v.content_preview, v.distance
        FROM {VEC_ITEMS} v
        WHERE v.embedding MATCH ? AND v.k = ?
        {clause}
        ORDER BY v.distance
    """
    rows = conn.execute(sql, (sqlite_vec.serialize_float32(embedding), pool, *params)).fetchall()
    return [
        Hit(
            id=str(row[0]),
            score=score_from_distance(row[5]),
            title=row[1] or "",
            node_type=row[2] or "",
            source_file=row[3] or "",
            content_preview=row[4] or "",
        )
        for row in rows
    ]


def fts_ranked(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
    *,
    column: str = "title",
    tags_all: Sequence[str] | None = None,
    tags_dnf: Sequence[Sequence[str]] | None = None,
    attributes_all: Sequence[str] | None = None,
    attribute_keys: Sequence[str] | None = None,
) -> list[Hit]:
    """Run one lexical query and return the hits in rank order.

    The query matches a single indexed column — ``title`` or
    ``content_preview``. FTS5's ``MATCH`` operator needs the *real* table
    name on its left for a cross-column match, so a column-scoped match
    (``f.title MATCH ?``) is the form that works with an alias.
    """
    match = fts_match_expression(query)
    if match is None:
        return []
    clause, params = filter_clause(
        "f",
        tags_all=tags_all,
        tags_dnf=tags_dnf,
        attributes_all=attributes_all,
        attribute_keys=attribute_keys,
    )
    sql = f"""
        SELECT f.id, f.title, f.content_preview, f.node_type, f.source_file, bm25({VEC_FTS}) AS rank
        FROM {VEC_FTS} f
        WHERE f.{column} MATCH ? {clause}
        ORDER BY rank
        LIMIT ?
    """
    rows = conn.execute(sql, (match, *params, top_k)).fetchall()
    return [
        Hit(
            id=str(row[0]),
            score=-float(row[5]),
            title=row[1] or "",
            node_type=row[3] or "",
            source_file=row[4] or "",
            content_preview=row[2] or "",
        )
        for row in rows
    ]


def score_from_distance(distance: float | None) -> float:
    """Convert a cosine distance to a similarity score (higher = better).

    The embedding field is indexed with ``distance_metric=cosine``, whose
    distance is in ``[0, 2]``. Retrievers rank descending by ``score``, so
    expose ``1 - distance``: for L2-normalised embeddings this equals the
    cosine similarity.
    """
    if distance is None:
        return 0.0
    return 1.0 - float(distance)


def candidate_pool(top_k: int, *, filtered: bool) -> int:
    """How many vector candidates to pull before the filters are applied."""
    if not filtered:
        return top_k
    return max(FILTER_POOL_MIN, top_k * FILTER_OVERSAMPLE)


def fts_match_expression(query: str) -> str | None:
    """Turn a user query into an FTS5 match expression, or ``None``.

    FTS5's query language is not a plain-text search: ``-`` is the NOT
    operator, ``:`` introduces a column filter, and ``"`` quotes a phrase. A
    user query like ``context-window`` therefore fails to parse ("no such
    column: window"). Splitting the query into its word characters and
    quoting each term keeps every term literal and every operator out of the
    expression.
    """
    terms = TERM_RE.findall(query)
    if not terms:
        return None
    return " ".join(f'"{term}"' for term in terms)


def fuse(
    ranked_lists: Sequence[tuple[list[Hit], float]],
    top_k: int,
) -> list[Hit]:
    """Weighted reciprocal rank fusion over the ranked lists.

    A document's score is the sum of ``weight / (RRF_K + rank)`` over the
    lists it appears in; rank is 1-based. Documents missing from a list
    contribute nothing from it.
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits, weight in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (RRF_K + rank)
            current = best.get(hit.id)
            if current is None or hit.score > current.score:
                best[hit.id] = hit
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[Hit] = []
    for node_id, score in ordered[:top_k]:
        hit = best[node_id]
        fused.append(
            Hit(
                id=hit.id,
                score=score,
                title=hit.title,
                node_type=hit.node_type,
                source_file=hit.source_file,
                content_preview=hit.content_preview,
            )
        )
    return fused
