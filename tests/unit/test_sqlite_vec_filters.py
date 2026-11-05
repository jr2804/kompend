"""Tests for the sqlite-vec store's tag/attribute schema and SQL filtering.

Verifies that tags and attributes are stored in indexed side tables and that
``search()`` pushes the filters down into SQL — and that the SQL filters pick
exactly the nodes the canonical Python predicates in
``kompend.search.filters`` pick.
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import NodeType
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.sqlite_vec_store import SqliteVecWikiStore
from kompend.rag.vector_store import VectorField
from kompend.search.filters import matches_attr_filter, matches_tag_filter


class _StubProvider(EmbeddingProvider):
    """Constant L2-normalised embedding — ranking ties are irrelevant here."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.5] * self._dim


class TestVectorFieldTags:
    """VectorField.TAGS schema constant."""

    def test_tags_field_exists(self) -> None:
        """TAGS is a valid member of VectorField."""
        assert hasattr(VectorField, "TAGS")
        assert VectorField.TAGS.value == "tags"

    def test_tags_field_is_string(self) -> None:
        """TAGS value is a plain string."""
        assert isinstance(VectorField.TAGS.value, str)


class TestFilterSchema:
    """Tags/attributes live in side tables the filters can index."""

    def test_tags_are_stored_in_the_tag_table(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        rows = store._require_conn().execute("SELECT item_id, tag FROM vec_item_tags ORDER BY item_id, tag").fetchall()
        assert [(row[0], row[1]) for row in rows] == [("n1", "x"), ("n1", "y"), ("n2", "y"), ("n2", "z")]
        store.close()

    def test_attribute_tokens_and_keys_are_stored_separately(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        conn = store._require_conn()
        tokens = conn.execute("SELECT item_id, token FROM vec_item_attributes ORDER BY item_id").fetchall()
        keys = conn.execute("SELECT item_id, key FROM vec_item_attr_keys ORDER BY item_id").fetchall()
        assert [(row[0], row[1]) for row in tokens] == [("n1", "lang=python"), ("n2", "lang=rust")]
        assert [(row[0], row[1]) for row in keys] == [("n1", "lang"), ("n2", "lang")]
        store.close()


class TestFilterPushdown:
    """The filters narrow the result set inside SQL."""

    def test_no_filter_returns_all(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10)}
        assert ids == {"n1", "n2", "n3"}
        store.close()

    def test_tags_all_filter(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10, tags_all=["y"])}
        assert ids == {"n1", "n2"}
        store.close()

    def test_tags_dnf_filter(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        # (x AND y) OR z -> n1 has x+y, n2 has z
        ids = {hit["id"] for hit in store.search("q", top_k=10, tags_dnf=[["x", "y"], ["z"]])}
        assert ids == {"n1", "n2"}
        store.close()

    def test_attributes_all_filter(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10, attributes_all=["lang=rust"])}
        assert ids == {"n2"}
        store.close()

    def test_attribute_keys_filter_is_key_presence(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10, attribute_keys=["lang"])}
        assert ids == {"n1", "n2"}
        store.close()

    def test_filters_combine_with_and(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10, tags_all=["y"], attributes_all=["lang=rust"])}
        assert ids == {"n2"}
        store.close()

    def test_filter_match_returns_empty_for_absent_tag(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        assert store.search("q", top_k=10, tags_all=["nonexistent"]) == []
        store.close()


class TestFilterParityWithCanonicalPredicates:
    """The SQL filters pick exactly what the Python predicates pick.

    ``search/filters.py`` holds the canonical semantics — tags are an OR of
    ANDs, attributes an AND of (key, value) checks. The store evaluates the
    same rules in SQL; this is the test that keeps the two in step.
    """

    def test_tag_filter_parity(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        groups = [["x"], ["y", "z"]]
        expected = {node_id for node_id, tags in _tags_by_id().items() if matches_tag_filter(set(tags), groups)}
        ids = {hit["id"] for hit in store.search("q", top_k=10, tags_dnf=groups)}
        assert ids == expected
        store.close()

    def test_attribute_filter_parity(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        checks = [("lang", "rust"), ("lang", None)]
        expected = {node_id for node_id, attrs in _attrs_by_id().items() if matches_attr_filter(attrs, checks)}
        ids = {hit["id"] for hit in store.search("q", top_k=10, attribute_keys=["lang"], attributes_all=["lang=rust"])}
        assert ids == expected
        store.close()


def _built_store(tmp_path: Path) -> SqliteVecWikiStore:
    store = SqliteVecWikiStore(db_file=tmp_path / "vectors.db", embedding_provider=_StubProvider())
    assert store.build_index(_tagged_wiki()) == 3
    return store


def _tagged_wiki() -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(id="n1", title="a", content="alpha", node_type=NodeType.CONCEPT, metadata={"tags": ["x", "y"], "attributes": {"lang": ["python"]}}),
            WikiNode(id="n2", title="b", content="beta", node_type=NodeType.CONCEPT, metadata={"tags": ["y", "z"], "attributes": {"lang": ["rust"]}}),
            WikiNode(id="n3", title="c", content="gamma", node_type=NodeType.CONCEPT, metadata={"tags": []}),
        ],
        edges=[],
        toc=[],
    )


def _tags_by_id() -> dict[str, list[str]]:
    return {
        "n1": ["x", "y"],
        "n2": ["y", "z"],
        "n3": [],
    }


def _attrs_by_id() -> dict[str, dict[str, list[str]]]:
    return {
        "n1": {"lang": ["python"]},
        "n2": {"lang": ["rust"]},
        "n3": {},
    }
