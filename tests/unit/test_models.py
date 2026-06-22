"""Tests for KNOX core data models."""

import json
from pathlib import Path

import pytest

from knox.core.models import (
    CHUNK_TREE_CACHE_VERSION,
    Chunk,
    ChunkNode,
    ChunkStrategy,
    ChunkTree,
    CompiledWiki,
    CrossReferenceIssue,
    CrossReferenceResult,
    DocumentElement,
    ElementType,
    EquationVariable,
    ParsedDocument,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
    WikiNode,
    _chunk_from_dict,
    _chunk_to_dict,
    _node_from_dict,
    _node_to_dict,
    _structured_from_dict,
    _structured_to_dict,
    tree_from_dict,
    tree_to_dict,
)


def test_element_type_enum() -> None:
    assert ElementType.TEXT.name == "TEXT"
    assert ElementType.TABLE.name == "TABLE"
    assert ElementType.IMAGE.name == "IMAGE"
    # StrEnum: values are lowercase strings
    assert ElementType.TEXT.value == "text"
    assert str(ElementType.TABLE) == "table"


def test_document_element_creation() -> None:
    elem = DocumentElement(
        type=ElementType.HEADING,
        content="# Introduction",
        page=1,
        reading_order=0,
        heading_level=1,
        metadata={"style": "Title"},
    )
    assert elem.type == ElementType.HEADING
    assert elem.heading_level == 1
    assert elem.metadata["style"] == "Title"


def test_parsed_document_structure() -> None:
    elems = [
        DocumentElement(
            type=ElementType.HEADING,
            content="# Title",
            page=1,
            reading_order=0,
            heading_level=1,
        ),
        DocumentElement(
            type=ElementType.TEXT,
            content="Body text here.",
            page=1,
            reading_order=1,
        ),
    ]
    doc = ParsedDocument(
        source_path=Path("test.pdf"),
        elements=elems,
        title="Test Doc",
        author="Test Author",
        page_count=2,
    )
    assert len(doc.elements) == 2
    assert doc.title == "Test Doc"
    assert doc.author == "Test Author"
    assert doc.elements[0].type == ElementType.HEADING


def test_chunk_creation() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="Sample text",
        element_types=[ElementType.TEXT],
        source_elements=[0, 1],
        heading_context="Section 1",
    )
    assert chunk.id == "chunk-1"
    assert chunk.heading_context == "Section 1"
    assert chunk.token_count == 0
    assert chunk.reading_order is None


def test_wiki_node_creation() -> None:
    node = WikiNode(
        id="node-1",
        title="Introduction",
        content="The intro...",
        node_type="section",
    )
    assert node.node_type == "section"
    assert node.title == "Introduction"


def test_compiled_wiki_creation() -> None:
    node = WikiNode(
        id="node-1",
        title="Intro",
        content="...",
        node_type="section",
    )
    wiki = CompiledWiki(
        nodes=[node],
        edges=[],
        toc=[],
    )
    assert len(wiki.nodes) == 1
    assert len(wiki.edges) == 0
    assert wiki.quality_score == 0.0
    assert wiki.iteration == 0


def test_element_type_json_roundtrip() -> None:
    """StrEnum values survive JSON serialization."""
    chunk = Chunk(
        id="c1",
        text="hello",
        element_types=[ElementType.TEXT, ElementType.TABLE],
        source_elements=[0],
        token_count=1,
        reading_order=(0, 5),
    )
    serialized = json.dumps({"types": chunk.element_types}, default=str)
    assert '"text"' in serialized
    assert '"table"' in serialized


# ── ChunkStrategy ────────────────────────────────────────────────────────


def test_chunk_strategy_values() -> None:
    assert ChunkStrategy.SINGLE_PASS.value == "single_pass"
    assert ChunkStrategy.TWO_LEVEL.value == "two_level"
    assert ChunkStrategy.MULTI_RESOLUTION.value == "multi_resolution"
    assert str(ChunkStrategy.SINGLE_PASS) == "single_pass"


# ── StructuredData types ─────────────────────────────────────────────────


def test_structured_table_creation() -> None:
    table = StructuredTable(
        columns=["Name", "Value"],
        rows=[{"Name": "SNR", "Value": "30 dB"}],
        caption="Test results",
    )
    assert table.columns == ["Name", "Value"]
    assert table.rows[0]["Value"] == "30 dB"
    assert table.caption == "Test results"


def test_structured_equation_creation() -> None:
    eq = StructuredEquation(
        latex=r"E = mc^2",
        variables=[
            EquationVariable(name="E", definition="Energy", units="J"),
            EquationVariable(name="m", definition="Mass", units="kg"),
        ],
    )
    assert eq.latex == r"E = mc^2"
    assert len(eq.variables) == 2
    assert eq.variables[0].units == "J"


def test_structured_figure_creation() -> None:
    fig = StructuredFigure(
        caption="Frequency response",
        alt_text="Graph showing frequency response",
        image_ref="images/fig1.png",
    )
    assert fig.caption == "Frequency response"
    assert fig.image_ref == "images/fig1.png"


# ── ChunkNode ────────────────────────────────────────────────────────────


def test_chunk_node_leaf_prose() -> None:
    chunk = Chunk(
        id="c1",
        text="Sample",
        element_types=[ElementType.TEXT],
        source_elements=[0],
    )
    node = ChunkNode(id="n1", chunk=chunk, depth=2)
    assert node.chunk is not None
    assert node.structured is None
    assert node.summary == ""


def test_chunk_node_leaf_structured() -> None:
    table = StructuredTable(columns=["A"], rows=[{"A": "1"}])
    node = ChunkNode(id="n2", structured=table, depth=2)
    assert node.chunk is None
    assert node.structured is not None
    assert isinstance(node.structured, StructuredTable)


def test_chunk_node_internal() -> None:
    node = ChunkNode(id="section1", summary="Section summary", depth=1)
    assert node.chunk is None
    assert node.structured is None
    assert node.summary == "Section summary"


# ── ChunkTree ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_tree() -> tuple[ChunkTree, dict[str, ChunkNode]]:
    nodes = {
        "root": ChunkNode(id="root", children_ids=["section-a", "section-b"], depth=0, summary="Root"),
        "section-a": ChunkNode(
            id="section-a",
            parent_id="root",
            children_ids=["leaf-1", "leaf-2"],
            depth=1,
            heading_path=("Chapter 1", "Section A"),
            summary="Section A summary",
            token_count=10,
        ),
        "section-b": ChunkNode(
            id="section-b",
            parent_id="root",
            children_ids=["leaf-3"],
            depth=1,
            heading_path=("Chapter 1", "Section B"),
            summary="Section B summary",
            token_count=5,
        ),
        "leaf-1": ChunkNode(
            id="leaf-1",
            parent_id="section-a",
            depth=2,
            heading_path=("Chapter 1", "Section A"),
            chunk=Chunk(id="c1", text="text 1", element_types=[ElementType.TEXT], source_elements=[0], token_count=3),
            token_count=3,
        ),
        "leaf-2": ChunkNode(
            id="leaf-2",
            parent_id="section-a",
            depth=2,
            heading_path=("Chapter 1", "Section A"),
            structured=StructuredTable(columns=["X"], rows=[{"X": "1"}]),
            token_count=7,
        ),
        "leaf-3": ChunkNode(
            id="leaf-3",
            parent_id="section-b",
            depth=2,
            heading_path=("Chapter 1", "Section B"),
            chunk=Chunk(id="c3", text="text 3", element_types=[ElementType.TEXT], source_elements=[2], token_count=5),
            token_count=5,
        ),
    }
    tree = ChunkTree(
        root_id="root",
        nodes=nodes,
        strategy=ChunkStrategy.MULTI_RESOLUTION,
    )
    return tree, nodes


def test_tree_leaves(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    leaf_ids = {n.id for n in tree.leaves()}
    assert leaf_ids == {"leaf-1", "leaf-2", "leaf-3"}


def test_tree_walk_depth_first(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    walk_ids = [n.id for n in tree.walk()]
    # Root first, then section-a subtree, then section-b subtree
    assert walk_ids[0] == "root"
    assert walk_ids[1] == "section-a"
    assert walk_ids[2] == "leaf-1"
    assert walk_ids[3] == "leaf-2"
    assert walk_ids[4] == "section-b"
    assert walk_ids[5] == "leaf-3"


def test_tree_get_path(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    path = tree.get_path("leaf-2")
    assert [n.id for n in path] == ["root", "section-a", "leaf-2"]


def test_tree_get_path_root(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    path = tree.get_path("root")
    assert len(path) == 1
    assert path[0].id == "root"


def test_tree_siblings(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    sibs = tree.siblings("leaf-1")
    assert {s.id for s in sibs} == {"leaf-2"}


def test_tree_siblings_no_parent(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    assert tree.siblings("root") == []


def test_tree_subtree_tokens(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    # root: 0 + section-a(10) + leaf-1(3) + leaf-2(7) + section-b(5) + leaf-3(5) = 30
    assert tree.subtree_tokens("root") == 30
    # section-a: 10 + leaf-1(3) + leaf-2(7) = 20
    assert tree.subtree_tokens("section-a") == 20
    # leaf-3: 5 (no children)
    assert tree.subtree_tokens("leaf-3") == 5


def test_tree_subtree_tokens_missing(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    assert tree.subtree_tokens("nonexistent") == 0


# ── Serialization ────────────────────────────────────────────────────────


def test_structured_table_roundtrip() -> None:
    original = StructuredTable(columns=["A", "B"], rows=[{"A": "1", "B": "2"}], caption="Test")
    restored = _structured_from_dict(_structured_to_dict(original))
    assert restored == original


def test_structured_equation_roundtrip() -> None:
    original = StructuredEquation(
        latex="x = y + z",
        variables=[EquationVariable(name="x", definition="position", units="m")],
        caption="Position equation",
    )
    restored = _structured_from_dict(_structured_to_dict(original))
    assert restored == original


def test_structured_figure_roundtrip() -> None:
    original = StructuredFigure(caption="Fig 1", alt_text="Alt", image_ref="img.png")
    restored = _structured_from_dict(_structured_to_dict(original))
    assert restored == original


def test_chunk_roundtrip() -> None:
    original = Chunk(
        id="c1",
        text="hello",
        element_types=[ElementType.TEXT, ElementType.TABLE],
        source_elements=[0, 1],
        source_file=Path("test.pdf"),
        heading_context="Section 1",
        token_count=5,
        reading_order=(0, 10),
    )
    restored = _chunk_from_dict(_chunk_to_dict(original))
    assert restored.id == original.id
    assert restored.text == original.text
    assert restored.element_types == original.element_types
    assert restored.source_elements == original.source_elements
    assert restored.heading_context == original.heading_context
    assert restored.token_count == original.token_count
    assert restored.reading_order == (0, 10)


def test_node_roundtrip_prose(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    _, nodes = sample_tree
    node = nodes["leaf-1"]
    restored = _node_from_dict(_node_to_dict(node))
    assert restored.id == node.id
    assert restored.depth == node.depth
    assert restored.chunk is not None
    assert restored.chunk.id == "c1"
    assert restored.structured is None


def test_node_roundtrip_structured(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    _, nodes = sample_tree
    node = nodes["leaf-2"]
    restored = _node_from_dict(_node_to_dict(node))
    assert restored.structured is not None
    assert isinstance(restored.structured, StructuredTable)
    assert restored.structured.columns == ["X"]


def test_node_roundtrip_internal(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    _, nodes = sample_tree
    node = nodes["section-a"]
    restored = _node_from_dict(_node_to_dict(node))
    assert restored.summary == "Section A summary"
    assert restored.children_ids == ["leaf-1", "leaf-2"]
    assert restored.chunk is None
    assert restored.structured is None


def test_tree_full_roundtrip(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    d = tree_to_dict(tree)
    # Must be JSON-serializable
    json_str = json.dumps(d)
    restored = tree_from_dict(json.loads(json_str))
    assert restored.root_id == tree.root_id
    assert restored.strategy == tree.strategy
    assert set(restored.nodes.keys()) == set(tree.nodes.keys())
    # Verify leaf-2 structured data survived
    leaf2 = restored.nodes["leaf-2"]
    assert leaf2.structured is not None
    assert isinstance(leaf2.structured, StructuredTable)


def test_tree_serialization_includes_version(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    tree, _ = sample_tree
    d = tree_to_dict(tree)
    assert d["version"] == CHUNK_TREE_CACHE_VERSION


def test_tree_deserialization_version_mismatch() -> None:
    bad_data = {"version": 999, "root_id": "x", "nodes": {}, "strategy": "single_pass"}
    with pytest.raises(ValueError, match="version mismatch"):
        tree_from_dict(bad_data)


def test_tree_heading_path_roundtrip(sample_tree: tuple[ChunkTree, dict[str, ChunkNode]]) -> None:
    _, nodes = sample_tree
    node = nodes["section-a"]
    restored = _node_from_dict(_node_to_dict(node))
    assert restored.heading_path == ("Chapter 1", "Section A")


# ── Cross-reference stubs ────────────────────────────────────────────────


def test_cross_reference_issue() -> None:
    issue = CrossReferenceIssue(
        issue_type="conflict",
        description="Term defined differently in two sections",
        entity="SNR",
        locations=["section-a", "section-b"],
    )
    assert issue.issue_type == "conflict"
    assert issue.entity == "SNR"
    assert len(issue.locations) == 2


def test_cross_reference_result() -> None:
    result = CrossReferenceResult(
        issues=[CrossReferenceIssue(issue_type="conflict", description="x")],
        entity_count=10,
        resolved_count=8,
    )
    assert len(result.issues) == 1
    assert result.entity_count == 10
    assert result.resolved_count == 8
