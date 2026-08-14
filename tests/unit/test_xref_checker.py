"""Tests for xref_checker.py — programmatic cross-reference consistency checks."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import (
    Chunk,
    ChunkNode,
    ChunkTree,
    ElementType,
    StructuredTable,
)
from knox.core.resolution.entity_registry import EntityRegistry
from knox.core.resolution.xref_checker import (
    CrossReferenceChecker,
    _extract_heading_number,
    _extract_number,
)


class TestHelpers:
    def test_extract_heading_number(self) -> None:
        assert _extract_heading_number("3.2 Methodology") == "3.2"
        assert _extract_heading_number("1 Introduction") == "1"
        assert _extract_heading_number("  4.1.2  Sub-section") == "4.1.2"
        assert _extract_heading_number("Introduction") == ""
        assert _extract_heading_number("") == ""

    def test_extract_number(self) -> None:
        assert _extract_number("Figure 5: Overview") == "5"
        assert _extract_number("Table 3a: Results") == ""  # 3a is not a standalone number


class TestCrossReferenceChecker:
    def test_check_definitions_delegates_to_registry(self) -> None:
        reg = EntityRegistry()
        reg.register("SNR", definition="Signal to Noise Ratio")
        reg.register("SNR", definition="Signal-to-Noise power Ratio")
        checker = CrossReferenceChecker(reg)
        issues = checker.check_definitions()
        assert len(issues) >= 1
        assert issues[0].issue_type == "conflict"

    def test_check_references_resolved(self) -> None:
        """Valid references to existing sections/tables produce no issues."""
        reg = EntityRegistry()
        nodes = [
            ChunkNode(
                id="h1",
                parent_id="root",
                depth=1,
                heading_chain=("3.1 Signal Processing",),
            ),
            _text_node(
                "t1",
                "h1",
                "As described in Section 3.1, the method works.",
                heading=("3.1 Signal Processing",),
            ),
        ]
        nodes[0].children_ids = ["t1"]
        tree = _leaf_tree(nodes)

        checker = CrossReferenceChecker(reg)
        issues = checker.check_references([tree])
        assert len(issues) == 0

    def test_check_references_unresolved(self) -> None:
        """References to non-existent sections are flagged."""
        reg = EntityRegistry()
        tree = _leaf_tree(
            [
                _text_node("t1", "root", "See Section 7.4 for details."),
            ]
        )
        checker = CrossReferenceChecker(reg)
        issues = checker.check_references([tree])
        assert len(issues) >= 1
        assert issues[0].issue_type == "unresolved"
        assert "section 7.4" in issues[0].description

    def test_check_references_multiple_ref_types(self) -> None:
        """References to tables, figures, and equations without targets flagged."""
        reg = EntityRegistry()
        tree = _leaf_tree(
            [
                _text_node(
                    "t1",
                    "root",
                    "See Table 2 and Figure 3 and Equation 1.",
                ),
            ]
        )
        checker = CrossReferenceChecker(reg)
        issues = checker.check_references([tree])
        assert len(issues) == 3

    def test_check_references_with_targets_resolved(self) -> None:
        """When table/figure/equation nodes exist, references resolve."""
        reg = EntityRegistry()
        nodes = [
            _table_node(
                "tbl1",
                "root",
                StructuredTable(columns=["A"], rows=[{"A": "1"}], caption="Table 2: Setup"),
            ),
            _text_node("t1", "root", "The results are shown in Table 2."),
        ]
        tree = _leaf_tree(nodes)
        checker = CrossReferenceChecker(reg)
        issues = checker.check_references([tree])
        assert len(issues) == 0

    def test_check_table_consistency_no_issues(self) -> None:
        """Identical values across tables produce no issues."""
        reg = EntityRegistry()
        t1 = StructuredTable(
            columns=["Name", "Value"],
            rows=[{"Name": "SNR", "Value": "20"}],
        )
        t2 = StructuredTable(
            columns=["Name", "Value"],
            rows=[{"Name": "SNR", "Value": "20"}],
        )
        nodes = [
            _table_node("a", "root", t1),
            _table_node("b", "root", t2),
        ]
        tree = _leaf_tree(nodes)
        checker = CrossReferenceChecker(reg)
        issues = checker.check_table_consistency([tree])
        assert len(issues) == 0

    def test_check_table_consistency_contradiction(self) -> None:
        """Contradicting values for same entity across tables flagged."""
        reg = EntityRegistry()
        t1 = StructuredTable(
            columns=["Name", "Value"],
            rows=[{"Name": "SNR", "Value": "20"}],
        )
        t2 = StructuredTable(
            columns=["Name", "Value"],
            rows=[{"Name": "SNR", "Value": "25"}],
        )
        nodes = [
            _table_node("a", "root", t1),
            _table_node("b", "root", t2),
        ]
        tree = _leaf_tree(nodes)
        checker = CrossReferenceChecker(reg)
        issues = checker.check_table_consistency([tree])
        assert len(issues) >= 1
        assert issues[0].issue_type == "contradiction"
        assert "SNR" in issues[0].entity

    def test_check_table_consistency_different_columns_ignored(self) -> None:
        """Tables with no shared columns are skipped."""
        reg = EntityRegistry()
        t1 = StructuredTable(columns=["A"], rows=[{"A": "1"}])
        t2 = StructuredTable(columns=["B"], rows=[{"B": "2"}])
        nodes = [
            _table_node("a", "root", t1),
            _table_node("b", "root", t2),
        ]
        tree = _leaf_tree(nodes)
        checker = CrossReferenceChecker(reg)
        issues = checker.check_table_consistency([tree])
        assert len(issues) == 0

    def test_check_table_consistency_single_table(self) -> None:
        """One table alone produces no issues."""
        reg = EntityRegistry()
        t1 = StructuredTable(
            columns=["Name", "Value"],
            rows=[{"Name": "SNR", "Value": "20"}],
        )
        tree = _leaf_tree([_table_node("a", "root", t1)])
        checker = CrossReferenceChecker(reg)
        issues = checker.check_table_consistency([tree])
        assert len(issues) == 0

    def test_check_all_integration(self) -> None:
        """Full check_all combines all checks."""
        reg = EntityRegistry()
        reg.register("SNR", definition="Signal to Noise Ratio")
        reg.register("SNR", definition="Signal-to-Noise power Ratio")

        tree = _leaf_tree(
            [
                _text_node("t1", "root", "See Section 99.9 for details."),
            ]
        )

        checker = CrossReferenceChecker(reg)
        result = checker.check_all([tree])
        assert result.entity_count >= 1
        assert result.resolved_count >= 0
        assert len(result.issues) >= 2


def _text_node(nid: str, parent: str, text: str, heading: tuple[str, ...] = ()) -> ChunkNode:
    return ChunkNode(
        id=nid,
        parent_id=parent,
        depth=len(heading) + 1,
        heading_chain=heading,
        chunk=_make_chunk(nid, text),
        token_count=max(1, len(text) // 4),
    )


def _make_chunk(cid: str, text: str, source: str = "doc.md") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[],
        source_file=Path(source),
        token_count=max(1, len(text) // 4),
    )


def _table_node(nid: str, parent: str, table: StructuredTable, heading: tuple[str, ...] = ()) -> ChunkNode:
    return ChunkNode(
        id=nid,
        parent_id=parent,
        depth=len(heading) + 1,
        heading_chain=heading,
        structured=table,
        token_count=1,
    )


def _leaf_tree(nodes: list[ChunkNode], source: str = "doc.md") -> ChunkTree:
    root = ChunkNode(id="root", depth=0, children_ids=[n.id for n in nodes])
    nodes.insert(0, root)
    nodes_by_id = {n.id: n for n in nodes}
    return ChunkTree(
        root_id="root",
        nodes=nodes_by_id,
        source_file=Path(source),
    )
