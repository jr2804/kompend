"""Tests for zvec store tag/attribute schema and server-side filtering.

Verifies that tags/attributes are stored as ``ARRAY_STRING`` and that
``search()`` pushes ``CONTAIN_ANY``/``CONTAIN_ALL`` filters into zvec.
"""

from __future__ import annotations

from pathlib import Path

import zvec as _zvec

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import NodeType
from kompend.rag.zvec_store import EmbeddingProvider, ZvecField, ZvecWikiStore


class _StubProvider(EmbeddingProvider):
    """Constant L2-normalised embedding — ranking ties are irrelevant here."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.5] * self._dim


class TestZvecFieldTags:
    """ZvecField.TAGS schema constant."""

    def test_tags_field_exists(self) -> None:
        """TAGS is a valid member of ZvecField."""
        assert hasattr(ZvecField, "TAGS")
        assert ZvecField.TAGS.value == "tags"

    def test_tags_field_is_string(self) -> None:
        """TAGS value is a plain string."""
        assert isinstance(ZvecField.TAGS.value, str)


class TestArrayTagSchema:
    """Tags/attributes are array fields, filterable server-side."""

    def test_tags_and_attributes_are_array_string(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        assert store._collection is not None
        field_types = {f.name: f.data_type for f in store._collection.schema.fields}
        assert field_types[ZvecField.TAGS.value] == _zvec.DataType.ARRAY_STRING
        assert field_types[ZvecField.ATTRIBUTES.value] == _zvec.DataType.ARRAY_STRING
        store.close()

    def test_tags_stored_and_returned_as_list(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        by_id = {hit["id"]: hit for hit in store.search("alpha", top_k=10)}
        assert by_id["n1"][ZvecField.TAGS.value] == ["x", "y"]
        assert by_id["n3"][ZvecField.TAGS.value] == []
        store.close()

    def test_no_filter_returns_all(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10)}
        assert ids == {"n1", "n2", "n3"}
        store.close()

    def test_tags_any_filter(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10, tags_any=["x"])}
        assert ids == {"n1"}
        store.close()

    def test_tags_all_filter(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10, tags_all=["y"])}
        assert ids == {"n1", "n2"}
        store.close()

    def test_attributes_any_filter(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10, attributes_any=["lang=rust"])}
        assert ids == {"n2"}
        store.close()

    def test_filters_combine_with_and(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        ids = {hit["id"] for hit in store.search("q", top_k=10, tags_all=["y"], attributes_any=["lang=rust"])}
        assert ids == {"n2"}
        store.close()

    def test_filter_match_returns_empty_for_absent_tag(self, tmp_path: Path) -> None:
        store = _built_store(tmp_path)
        assert store.search("q", top_k=10, tags_any=["nonexistent"]) == []
        store.close()


def _built_store(tmp_path: Path) -> ZvecWikiStore:
    store = ZvecWikiStore(db_file=tmp_path / "vectors", embedding_provider=_StubProvider())
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
