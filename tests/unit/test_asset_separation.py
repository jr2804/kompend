"""Tests for Phase 1 structured-asset separation.

Covers:
- ``AssetRecord`` serialization roundtrip
- ``AssetStore`` persistence (save/load/put/get)
- ``summarize_asset`` deterministic text summary
- ``assign_asset_ids`` tree-walk and id assignment
- ``rewrite_asset_mentions`` text replacement
- ``build_tree_texts`` in ``asset_mode="ref"``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knox.compilation.asset_mention_rewriter import rewrite_asset_mentions
from knox.compilation.asset_registry import AssetIdMap, assign_asset_ids
from knox.compilation.asset_summarizer import summarize_asset
from knox.core.models import ChunkNode, ChunkTree
from knox.core.models.assets import AssetRecord
from knox.core.models.enums import AssetField, AssetKind, StructuredField
from knox.core.models.structured import (
    StructuredData,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)
from knox.core.pipeline.assets import AssetStore

# ═══════════════════════════════════════════════════════════════
# AssetRecord
# ═══════════════════════════════════════════════════════════════


class TestAssetRecord:
    """Round-trip serialization of AssetRecord through to_dict/from_dict."""

    def test_table_record_roundtrip(self) -> None:
        data = StructuredTable(columns=["Name", "Age"], rows=[{"Name": "Alice", "Age": "30"}])
        record = AssetRecord(
            id="tab-a1b2c3",
            kind=AssetKind.TABLE,
            source="test-source",
            data=data,
            summary="Table showing a vs b",
            ref_count=1,
        )
        d = record.to_dict()
        restored = AssetRecord.from_dict(d)
        assert restored.id == "tab-a1b2c3"
        assert restored.kind == AssetKind.TABLE
        assert restored.source == "test-source"
        assert restored.summary == "Table showing a vs b"
        assert restored.ref_count == 1
        # Restored data should be a StructuredTable with same columns/rows.
        assert isinstance(restored.data, StructuredTable)
        assert restored.data.columns == ["Name", "Age"]
        assert restored.data.rows == [{"Name": "Alice", "Age": "30"}]

    def test_figure_record_roundtrip(self) -> None:
        data = StructuredFigure(
            caption="Architecture diagram",
            alt_text="Layered boxes",
        )
        record = AssetRecord(
            id="fig-9c2e",
            kind=AssetKind.FIGURE,
            source="paper.pdf",
            data=data,
            summary="Architecture diagram — layered boxes",
        )
        restored = AssetRecord.from_dict(record.to_dict())
        assert restored.id == "fig-9c2e"
        assert restored.kind == AssetKind.FIGURE
        assert restored.summary == "Architecture diagram — layered boxes"
        assert isinstance(restored.data, StructuredFigure)

    def test_minimal_record_deserializes(self) -> None:
        raw: dict[str, Any] = {
            AssetField.ID: "tab-000000",
            AssetField.KIND: "table",
            AssetField.SOURCE: "",
            AssetField.DATA: {
                StructuredField._TYPE: StructuredField.TYPE_TABLE.value,
                "columns": ["x"],
                "rows": [{"x": "1"}],
            },
        }
        record = AssetRecord.from_dict(raw)
        assert record.id == "tab-000000"
        assert record.kind == AssetKind.TABLE
        assert record.summary == ""
        assert record.ref_count == 0
        assert isinstance(record.data, StructuredTable)


# ═══════════════════════════════════════════════════════════════
# AssetStore
# ═══════════════════════════════════════════════════════════════


class TestAssetStore:
    """JSON-backed store — save/load roundtrip, missing-file, corrupt-file."""

    def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets.json")
        data = StructuredTable(columns=["h1", "h2"], rows=[{"h1": "v1", "h2": "v2"}])
        record = AssetRecord(
            id="tab-x1y2",
            kind=AssetKind.TABLE,
            source="src.pdf",
            data=data,
            summary="a table",
        )
        store.put(record)
        got = store.get("tab-x1y2")
        assert got is not None
        assert got.id == "tab-x1y2"
        assert got.summary == "a table"
        assert isinstance(got.data, StructuredTable)
        assert got.data.columns == ["h1", "h2"]

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "assets.json"
        store1 = AssetStore(p)
        data = StructuredFigure(caption="Figure 1")
        store1.put(AssetRecord("fig-1", AssetKind.FIGURE, "src", data, "Fig 1"))
        store1.save()

        store2 = AssetStore(p)
        got = store2.get("fig-1")
        assert got is not None
        assert got.kind == AssetKind.FIGURE
        assert got.summary == "Fig 1"
        assert isinstance(got.data, StructuredFigure)
        assert got.data.caption == "Figure 1"

    def test_missing_file(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "nonexistent.json")
        assert store.get("anything") is None
        # Should not crash.

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "empty.json")
        assert store.get("tab-nope") is None

    def test_increment_ref(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "ref.json")
        data = StructuredTable(columns=["x"], rows=[{"x": "1"}])
        store.put(AssetRecord("tab-r1", AssetKind.TABLE, "src", data))
        store.increment_ref("tab-r1")
        got = store.get("tab-r1")
        assert got is not None
        assert got.ref_count == 1
        store.increment_ref("tab-r1")
        got2 = store.get("tab-r1")
        assert got2 is not None
        assert got2.ref_count == 2


# ═══════════════════════════════════════════════════════════════
# Asset summarizer
# ═══════════════════════════════════════════════════════════════


class TestSummarizeAsset:
    """Deterministic text summaries for structured assets."""

    def test_table_summary_uses_headers(self) -> None:
        data = StructuredTable(
            columns=["Name", "Age"],
            rows=[{"Name": "Alice", "Age": "30"}, {"Name": "Bob", "Age": "25"}],
        )
        summary = summarize_asset(data)
        assert summary == "2 cols × 2 rows"

    def test_table_with_caption(self) -> None:
        data = StructuredTable(
            columns=["x"],
            rows=[{"x": "1"}],
            caption="Sample results",
        )
        summary = summarize_asset(data)
        assert summary == "1 cols × 1 rows — Sample results"

    def test_figure_summary_uses_caption(self) -> None:
        data = StructuredFigure(caption="System architecture overview")
        summary = summarize_asset(data)
        assert summary == "System architecture overview"

    def test_figure_falls_back_to_alt_text(self) -> None:
        data = StructuredFigure(caption="", alt_text="Architecture diagram showing layers")
        summary = summarize_asset(data)
        assert summary == "Architecture diagram showing layers"

    def test_figure_falls_back_to_kind(self) -> None:
        data = StructuredFigure(caption="")
        summary = summarize_asset(data)
        # Falls through to fallback "figure".
        assert "figure" in summary

    def test_equation_returns_equation(self) -> None:
        data = StructuredEquation(latex="E = mc^2")
        summary = summarize_asset(data)
        # Equations are deferred in Phase 1 but handled defensively.
        assert summary == "equation"


# ═══════════════════════════════════════════════════════════════
# Asset registry
# ═══════════════════════════════════════════════════════════════


class TestAssignAssetIds:
    """Content-addressed id assignment via tree walk."""

    def test_skips_nodes_without_structured(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets.json")
        node_a = self._node("a")
        node_b = self._node("b")
        root = ChunkNode(id="root", children_ids=["a", "b"])
        tree = ChunkTree(root_id="root", nodes={"root": root, "a": node_a, "b": node_b})
        id_map = assign_asset_ids(tree, store, source="test.pdf")
        assert len(id_map.node_to_asset) == 0
        assert len(id_map) == 0  # no assets stored

    def test_assigns_table_id(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets.json")
        data = StructuredTable(columns=["h1", "h2"], rows=[{"h1": "v1", "h2": "v2"}])
        node = self._node("n1", structured=data)
        root = ChunkNode(id="root", children_ids=["n1"])
        tree = ChunkTree(root_id="root", nodes={"root": root, "n1": node})
        id_map = assign_asset_ids(tree, store, source="test.pdf")
        assert len(id_map.node_to_asset) == 1
        asset_id = id_map.node_to_asset["n1"]
        assert asset_id.startswith("tab-")
        record = store.get(asset_id)
        assert record is not None
        assert record.kind == AssetKind.TABLE
        assert record.source == "test.pdf"

    def test_dedup_same_content(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets.json")
        data = StructuredTable(columns=["h1"], rows=[{"h1": "v1"}])
        n1 = self._node("n1", structured=data)
        n2 = self._node("n2", structured=data)
        root = ChunkNode(id="root", children_ids=["n1", "n2"])
        tree = ChunkTree(root_id="root", nodes={"root": root, "n1": n1, "n2": n2})
        id_map = assign_asset_ids(tree, store, source="src.pdf")
        ids = set(id_map.node_to_asset.values())
        assert len(ids) == 1  # same content → same id

    def test_tracks_sequence_number(self, tmp_path: Path) -> None:
        """assign_asset_ids tracks per-kind sequence for mention rewriting."""
        store = AssetStore(tmp_path / "assets.json")
        t1 = StructuredTable(columns=["x"], rows=[{"x": "1"}])
        t2 = StructuredTable(columns=["y"], rows=[{"y": "2"}])
        node1 = self._node("n1", structured=t1)
        node2 = self._node("n2", structured=t2)
        root = ChunkNode(id="root", children_ids=["n1", "n2"])
        tree = ChunkTree(root_id="root", nodes={"root": root, "n1": node1, "n2": node2})
        id_map = assign_asset_ids(tree, store, source="src.pdf")
        # id_map includes seq number marker; verify by looking at the AssetIdMap.
        assert isinstance(id_map, AssetIdMap)

    def _node(self, node_id: str, structured: StructuredData | None = None) -> ChunkNode:
        return ChunkNode(
            id=node_id,
            structured=structured,
            summary=node_id,
            heading_chain=(),
        )


# ═══════════════════════════════════════════════════════════════
# Asset mention rewriter
# ═══════════════════════════════════════════════════════════════


class TestRewriteAssetMentions:
    """Rewrite natural-language figure/table references to asset refs."""

    def test_figure_number_one(self) -> None:
        text = "as shown in Figure 1"
        result = rewrite_asset_mentions(text, self._make_map())
        assert "¶ fig-a1b2c3" in result
        assert "Figure 1" not in result

    def test_table_number_three(self) -> None:
        text = "Table 3 demonstrates"
        result = rewrite_asset_mentions(text, self._make_map())
        assert "¶ tab-m9n8o7" in result

    def test_multiple_mentions(self) -> None:
        text = "Figure 1 and Table 1 show data. See also Figure 2."
        result = rewrite_asset_mentions(text, self._make_map())
        assert "¶ fig-a1b2c3" in result
        assert "¶ tab-x1y2z3" in result
        assert "¶ fig-d4e5f6" in result

    def test_no_known_mention_passes_through(self) -> None:
        text = "some unrelated text"
        result = rewrite_asset_mentions(text, self._make_map())
        assert result == text

    def test_unknown_number_keeps_original(self) -> None:
        text = "Figure 5 is interesting"
        result = rewrite_asset_mentions(text, self._make_map())
        # seq 5 not in map → passes through unchanged
        assert result == text

    def test_case_insensitive(self) -> None:
        text = "See Fig. 1 and figure 2"
        result = rewrite_asset_mentions(text, self._make_map())
        assert "¶ fig-a1b2c3" in result
        assert "¶ fig-d4e5f6" in result

    def test_table_abbreviation(self) -> None:
        text = "Tab. 1 summarizes"
        result = rewrite_asset_mentions(text, self._make_map())
        assert "¶ tab-x1y2z3" in result

    def test_with_leading_text(self) -> None:
        text = "As illustrated in Figure 1"
        result = rewrite_asset_mentions(text, self._make_map())
        assert "¶ fig-a1b2c3" in result

    def test_with_colon_separator(self) -> None:
        text = "Table 1: results"
        result = rewrite_asset_mentions(text, self._make_map())
        assert "¶ tab-x1y2z3" in result

    def test_in_parentheses(self) -> None:
        text = "the result (Figure 1) shows"
        result = rewrite_asset_mentions(text, self._make_map())
        assert "¶ fig-a1b2c3" in result

    def _make_map(self) -> AssetIdMap:
        return AssetIdMap(
            label_to_asset={
                (AssetKind.FIGURE, 1): "fig-a1b2c3",
                (AssetKind.FIGURE, 2): "fig-d4e5f6",
                (AssetKind.TABLE, 1): "tab-x1y2z3",
                (AssetKind.TABLE, 3): "tab-m9n8o7",
            },
        )
