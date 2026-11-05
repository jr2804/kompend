"""Tests for the sqlite-vec lexical (FTS) and hybrid retrieval paths."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pytest

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import NodeType
from kompend.rag.sqlite_vec_store import SqliteVecWikiStore
from kompend.rag.vector_store import EmbeddingProvider, VectorField
from tests.conftest import as_float


class _StubProvider(EmbeddingProvider):
    """Bag-of-words embedding over a fixed vocabulary — no LLM, stable ranking.

    The buckets come from a digest, not ``hash()``: Python randomises string
    hashing per process, so a ``hash()``-based provider ranks differently from
    run to run.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        raw = [0.0] * self._dim
        for token in _tokens(text):
            raw[_bucket(token, self._dim)] += 1.0
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]


def _tokens(text: str) -> list[str]:
    """The word tokens of *text* — case-folded, punctuation split off."""
    return [token for token in re.split(r"[^\w]+", text.lower()) if token]


def _bucket(token: str, dim: int) -> int:
    """The stable bucket a token falls into."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


class TestFtsSchema:
    def test_lexical_index_covers_title_and_preview(self, tmp_path: Path) -> None:
        """Both lexical fields are indexed, so a page matches on either."""
        store = _store(tmp_path)
        columns = {row[1] for row in store._require_conn().execute("PRAGMA table_info(vec_fts)").fetchall()}
        assert {VectorField.TITLE.value, VectorField.CONTENT_PREVIEW.value} <= columns
        store.close()

    def test_attr_keys_field_exists(self) -> None:
        assert VectorField.ATTR_KEYS.value == "attr_keys"


class TestFtsSearch:
    def test_identifier_query_hits(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        hits = store.fts_search("ITU-T P.863", top_k=5)
        assert hits
        assert hits[0]["id"] == "n1"
        # higher = better
        scores = [as_float(hit["score"]) for hit in hits]
        assert scores == sorted(scores, reverse=True)
        store.close()

    def test_matches_the_preview_too(self, tmp_path: Path) -> None:
        """The preview is the second lexical field — a page matches on it."""
        store = _store(tmp_path)
        hits = store.fts_search("codecs", top_k=5)
        assert [hit["id"] for hit in hits] == ["n3"]
        store.close()

    def test_same_dict_shape_as_search(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        hit = store.fts_search("listening quality", top_k=1)[0]
        assert {"id", "score", VectorField.TITLE.value, VectorField.NODE_ID.value} <= set(hit)
        store.close()

    def test_no_lexical_match_returns_empty(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.fts_search("zzz-nonexistent-token", top_k=5) == []
        store.close()

    def test_operators_in_the_query_do_not_break_the_search(self, tmp_path: Path) -> None:
        """FTS5's own syntax must not leak into a user query.

        ``-`` is FTS5's NOT operator and ``:`` introduces a column filter, so
        a query like ``context-window`` fails to parse when handed to FTS5
        verbatim.
        """
        store = _store(tmp_path)
        assert store.fts_search("ITU-T P.863", top_k=5)
        assert store.fts_search('unclosed "quote', top_k=5) == []
        store.close()

    def test_closed_store_returns_empty(self) -> None:
        store = SqliteVecWikiStore()
        assert store.fts_search("anything") == []


class TestHybridSearch:
    def test_returns_hits_best_first(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        hits = store.hybrid_search("ITU-T P.863 listening", top_k=3)
        assert hits
        assert hits[0]["id"] == "n1"
        scores = [as_float(hit["score"]) for hit in hits]
        assert scores == sorted(scores, reverse=True)
        store.close()

    def test_same_dict_shape_as_search(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        hit = store.hybrid_search("quality", top_k=1)[0]
        assert {"id", "score", VectorField.TITLE.value} <= set(hit)
        store.close()

    def test_weights_are_accepted(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        lean_lexical = store.hybrid_search("ITU-T P.863", top_k=3, vector_weight=0.2, fts_weight=1.0)
        assert lean_lexical
        assert lean_lexical[0]["id"] == "n1"
        store.close()


class TestPushdownFilters:
    def test_tags_dnf_or_of_ands(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        # (a AND b) OR c  ->  n1 (a,b) and n3 (c); n2 (a only) excluded
        ids = {hit["id"] for hit in store.fts_search("quality", top_k=10, tags_dnf=[["a", "b"], ["c"]])}
        assert ids == {"n1", "n3"}
        store.close()

    def test_attribute_keys_presence(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        ids = {hit["id"] for hit in store.fts_search("quality listening", top_k=10, attribute_keys=["version"])}
        assert ids == {"n1"}
        store.close()

    def test_dnf_and_presence_combine(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        ids = {hit["id"] for hit in store.fts_search("quality", top_k=10, tags_dnf=[["a"], ["c"]], attribute_keys=["lang"])}
        assert ids == {"n2"}
        store.close()


class TestFtsDegradation:
    """A store without the lexical index answers lexically with nothing."""

    def test_fts_search_returns_empty_without_fts(self, tmp_path: Path) -> None:
        store = self._legacy_store(tmp_path)
        assert store.fts_search("ITU-T P.863") == []
        store.close()

    def test_hybrid_degrades_to_vector_path(self, tmp_path: Path) -> None:
        store = self._legacy_store(tmp_path)
        hits = store.hybrid_search("ITU-T P.863", top_k=3)
        assert hits  # the vector fallback served the hits
        assert {hit["id"] for hit in hits} == {"n1", "n2", "n3"}
        store.close()

    def test_fts_unavailable_logs_once(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        store = self._legacy_store(tmp_path)
        caplog.set_level(logging.WARNING)
        store.fts_search("a")
        store.fts_search("b")
        store.hybrid_search("c")
        warnings = [record for record in caplog.records if "Lexical search failed" in record.message]
        assert len(warnings) == 1
        store.close()

    def _legacy_store(self, tmp_path: Path) -> SqliteVecWikiStore:
        """A store whose lexical index is missing, opened for querying.

        The vector table is intact, so the vector path still answers; the
        lexical table is gone, which is what the degradation covers.
        """
        db = tmp_path / "vectors.db"
        store = SqliteVecWikiStore(db_file=db, embedding_provider=_StubProvider())
        assert store.build_index(_wiki()) == 3
        store._require_conn().execute("DROP TABLE vec_fts")
        store.close()
        return SqliteVecWikiStore.open(str(db), embedding_provider=_StubProvider())


def _store(tmp_path: Path, name: str = "vectors.db") -> SqliteVecWikiStore:
    store = SqliteVecWikiStore(db_file=tmp_path / name, embedding_provider=_StubProvider())
    assert store.build_index(_wiki()) == 3
    return store


def _wiki() -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(
                id="n1",
                title="ITU-T P.863",
                content="perceptual objective listening quality assessment",
                node_type=NodeType.CONCEPT,
                metadata={"tags": ["a", "b"], "attributes": {"version": ["2024"]}},
            ),
            WikiNode(
                id="n2",
                title="MOS-LQO",
                content="mean opinion score listening quality objective",
                node_type=NodeType.CONCEPT,
                metadata={"tags": ["a"], "attributes": {"lang": ["en"]}},
            ),
            WikiNode(
                id="n3",
                title="Codec overview",
                content="overview of wideband codecs and their quality",
                node_type=NodeType.CONCEPT,
                metadata={"tags": ["c"], "attributes": {}},
            ),
        ],
        edges=[],
        toc=[],
    )
