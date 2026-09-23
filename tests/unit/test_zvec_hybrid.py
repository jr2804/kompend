"""Tests for zvec lexical (FTS) and hybrid retrieval paths."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import zvec as _zvec

from knox.core.models import CompiledWiki, WikiNode
from knox.rag.zvec_store import EmbeddingProvider, ZvecField, ZvecWikiStore


class _StubProvider(EmbeddingProvider):
    """Deterministic keyword-ish embedding — no LLM, stable ranking."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        raw = [0.0] * self._dim
        for token in text.lower().split():
            raw[hash(token) % self._dim] += 1.0
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]


class TestFtsSchema:
    def test_title_and_preview_have_fts_index(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store._collection is not None
        kinds = {f.name: f.index_param.type.name for f in store._collection.schema.fields if f.index_param is not None}
        assert kinds[ZvecField.TITLE.value] == "FTS"
        assert kinds[ZvecField.CONTENT_PREVIEW.value] == "FTS"
        store.close()

    def test_attr_keys_field_exists(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store._collection is not None
        assert ZvecField.ATTR_KEYS.value in {f.name for f in store._collection.schema.fields}
        store.close()


class TestFtsSearch:
    def test_identifier_query_hits(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        hits = store.fts_search("ITU-T P.863", top_k=5)
        assert hits
        assert hits[0]["id"] == "n1"
        # higher = better
        scores = [float(h["score"]) for h in hits]
        assert scores == sorted(scores, reverse=True)
        store.close()

    def test_same_dict_shape_as_search(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        hit = store.fts_search("listening quality", top_k=1)[0]
        assert {"id", "score", ZvecField.TITLE.value, ZvecField.NODE_ID.value} <= set(hit)
        store.close()

    def test_no_lexical_match_returns_empty(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.fts_search("zzz-nonexistent-token", top_k=5) == []
        store.close()

    def test_closed_store_returns_empty(self) -> None:
        store = ZvecWikiStore()
        assert store.fts_search("anything") == []


class TestHybridSearch:
    def test_returns_hits_best_first(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        hits = store.hybrid_search("ITU-T P.863 listening", top_k=3)
        assert hits
        assert hits[0]["id"] == "n1"
        scores = [float(h["score"]) for h in hits]
        assert scores == sorted(scores, reverse=True)
        store.close()

    def test_same_dict_shape_as_search(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        hit = store.hybrid_search("quality", top_k=1)[0]
        assert {"id", "score", ZvecField.TITLE.value} <= set(hit)
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
        ids = {h["id"] for h in store.fts_search("quality codec listening", top_k=10, tags_dnf=[["a", "b"], ["c"]])}
        assert ids == {"n1", "n3"}
        store.close()

    def test_attribute_keys_presence(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        ids = {h["id"] for h in store.fts_search("quality listening", top_k=10, attribute_keys=["version"])}
        assert ids == {"n1"}
        store.close()

    def test_dnf_and_presence_combine(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        ids = {h["id"] for h in store.fts_search("quality", top_k=10, tags_dnf=[["a"], ["c"]], attribute_keys=["lang"])}
        assert ids == {"n2"}
        store.close()


class TestFtsDegradation:
    def test_fts_search_returns_empty_without_fts(self, tmp_path: Path) -> None:
        store = self._legacy_store(tmp_path)
        assert store.fts_search("ITU-T P.863") == []
        store.close()

    def test_hybrid_degrades_to_vector_path(self, tmp_path: Path) -> None:
        store = self._legacy_store(tmp_path)
        hits = store.hybrid_search("ITU-T P.863", top_k=3)
        assert [h["id"] for h in hits] == ["n1"]  # vector fallback served the hit
        store.close()

    def test_fts_unavailable_logs_once(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        store = self._legacy_store(tmp_path)
        caplog.set_level(logging.WARNING)
        store.fts_search("a")
        store.fts_search("b")
        store.hybrid_search("c")
        warnings = [r for r in caplog.records if "FTS unavailable" in r.message]
        assert len(warnings) == 1
        store.close()

    def _legacy_store(self, tmp_path: Path) -> ZvecWikiStore:
        """A collection with NO FTS index, opened for querying."""
        db = tmp_path / "vectors"
        legacy_schema = _zvec.CollectionSchema(
            name="wiki",
            fields=[
                _zvec.FieldSchema("node_id", _zvec.DataType.STRING),
                _zvec.FieldSchema("title", _zvec.DataType.STRING),
                _zvec.FieldSchema("node_type", _zvec.DataType.STRING),
                _zvec.FieldSchema("source_file", _zvec.DataType.STRING),
                _zvec.FieldSchema("content_preview", _zvec.DataType.STRING),
                _zvec.FieldSchema("tags", _zvec.DataType.ARRAY_STRING),
                _zvec.FieldSchema("attributes", _zvec.DataType.ARRAY_STRING),
                _zvec.FieldSchema("attr_keys", _zvec.DataType.ARRAY_STRING),
            ],
            vectors=_zvec.VectorSchema("embedding", _zvec.DataType.VECTOR_FP32, 8, index_param=_zvec.HnswIndexParam(metric_type=_zvec.MetricType.COSINE)),
        )
        legacy = _zvec.create_and_open(path=str(db), schema=legacy_schema)
        legacy.upsert(
            [
                _zvec.Doc(
                    id="n1",
                    vectors={"embedding": [1.0] + [0.0] * 7},
                    fields={
                        "node_id": "n1",
                        "title": "ITU-T P.863",
                        "node_type": "concept",
                        "source_file": "",
                        "content_preview": "quality",
                        "tags": [],
                        "attributes": [],
                        "attr_keys": [],
                    },
                ),
            ],
        )
        legacy.close()
        return ZvecWikiStore.open(str(db), embedding_provider=_StubProvider())


class TestRebuildOnFtsSchema:
    def test_index_kind_change_triggers_rebuild(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Adding an FTS index leaves name+data_type unchanged; the fingerprint
        must include the index kind so the vault is rebuilt.
        """
        db = tmp_path / "vectors"
        legacy_schema = _zvec.CollectionSchema(
            name="wiki",
            fields=[
                _zvec.FieldSchema("node_id", _zvec.DataType.STRING),
                _zvec.FieldSchema("title", _zvec.DataType.STRING),  # no FTS
                _zvec.FieldSchema("node_type", _zvec.DataType.STRING),
                _zvec.FieldSchema("source_file", _zvec.DataType.STRING),
                _zvec.FieldSchema("content_preview", _zvec.DataType.STRING),  # no FTS
                _zvec.FieldSchema("tags", _zvec.DataType.ARRAY_STRING, index_param=_zvec.InvertIndexParam()),
                _zvec.FieldSchema("attributes", _zvec.DataType.ARRAY_STRING, index_param=_zvec.InvertIndexParam()),
                _zvec.FieldSchema("attr_keys", _zvec.DataType.ARRAY_STRING, index_param=_zvec.InvertIndexParam()),
            ],
            vectors=_zvec.VectorSchema("embedding", _zvec.DataType.VECTOR_FP32, 8),
        )
        legacy = _zvec.create_and_open(path=str(db), schema=legacy_schema)
        legacy.close()

        caplog.set_level(logging.WARNING)
        store = ZvecWikiStore(db_file=db, embedding_provider=_StubProvider())
        assert store.build_index(_wiki()) == 3
        assert store._collection is not None
        kinds = {f.name: f.index_param.type.name for f in store._collection.schema.fields if f.index_param is not None}
        assert kinds[ZvecField.TITLE.value] == "FTS"
        assert any("schema mismatch" in rec.message for rec in caplog.records)
        store.close()


def _store(tmp_path: Path, name: str = "vectors") -> ZvecWikiStore:
    store = ZvecWikiStore(db_file=tmp_path / name, embedding_provider=_StubProvider())
    assert store.build_index(_wiki()) == 3
    return store


def _wiki() -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(
                id="n1",
                title="ITU-T P.863",
                content="perceptual objective listening quality assessment",
                node_type="concept",
                metadata={"tags": ["a", "b"], "attributes": {"version": ["2024"]}},
            ),
            WikiNode(
                id="n2",
                title="MOS-LQO",
                content="mean opinion score listening quality objective",
                node_type="concept",
                metadata={"tags": ["a"], "attributes": {"lang": ["en"]}},
            ),
            WikiNode(
                id="n3",
                title="Codec overview",
                content="overview of wideband codecs and their quality",
                node_type="concept",
                metadata={"tags": ["c"], "attributes": {}},
            ),
        ],
        edges=[],
        toc=[],
    )
