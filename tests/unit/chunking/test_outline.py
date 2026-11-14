"""Unit tests for the structural chunk-tree outline (no LLM)."""

from __future__ import annotations

import time
from pathlib import Path

from kompend.chunking.outline import (
    OutlineNode,
    build_outline,
    estimate_tokens,
    outline_to_prompt,
)
from kompend.core.models import Chunk, ChunkNode, ChunkStrategy, ChunkTree, ElementType
from kompend.core.models.structured import StructuredTable
from kompend.core.text.density import conservative_tokens


def test_leaf_only_tree() -> None:
    """One leaf: payload, subtree, kinds, and preview follow from the chunk."""
    entries = build_outline(_leaf_only_tree())
    assert len(entries) == 1
    entry = entries[0]
    assert entry.node_id == "r"
    assert entry.heading_chain == ()
    assert entry.depth == 0
    assert entry.child_count == 0
    assert entry.payload_chars == len("hello world")
    assert entry.subtree_chars == entry.payload_chars
    assert entry.token_estimate == conservative_tokens("hello world")
    assert entry.kinds == ()
    assert entry.preview == "hello world"


def _leaf_only_tree() -> ChunkTree:
    """A root that is itself the only leaf."""
    root = ChunkNode(id="r", depth=0, chunk=_make_chunk("r", "hello world"))
    return _make_tree([root], "r")


def test_single_child_chain() -> None:
    """Root → section → leaf comes back in document order with nested sizes."""
    leaf = ChunkNode(
        id="l",
        parent_id="s",
        depth=2,
        heading_chain=("Intro",),
        chunk=_make_chunk("l", "leaf text here"),
    )
    section = ChunkNode(
        id="s",
        parent_id="r",
        depth=1,
        heading_chain=("Intro",),
        children_ids=["l"],
    )
    root = ChunkNode(id="r", depth=0, children_ids=["s"])
    entries = build_outline(_make_tree([root, section, leaf], "r"))

    assert [entry.node_id for entry in entries] == ["r", "s", "l"]
    by_id = {entry.node_id: entry for entry in entries}
    assert by_id["l"].subtree_chars == len("leaf text here")
    assert by_id["s"].subtree_chars == len("leaf text here")
    assert by_id["s"].child_count == 1
    assert by_id["s"].kinds == ("chunk",)
    assert by_id["r"].subtree_chars == len("leaf text here")
    assert by_id["r"].kinds == ("internal",)


def test_wide_root() -> None:
    """A root with four chunk leaves counts children and sums payloads."""
    leaves = [
        ChunkNode(
            id=f"l{index}",
            parent_id="r",
            depth=1,
            chunk=_make_chunk(f"l{index}", f"leaf number {index}"),
        )
        for index in range(4)
    ]
    root = ChunkNode(id="r", depth=0, children_ids=[leaf.id for leaf in leaves])
    entries = build_outline(_make_tree([root, *leaves], "r"))

    by_id = {entry.node_id: entry for entry in entries}
    assert by_id["r"].child_count == 4
    assert by_id["r"].kinds == ("chunk",)
    expected = sum(len(f"leaf number {index}") for index in range(4))
    assert by_id["r"].subtree_chars == expected


def test_mixed_chunk_structured_children() -> None:
    """Chunk, structured, and internal children classify in document order."""
    chunk_leaf = ChunkNode(
        id="c",
        parent_id="s",
        depth=2,
        chunk=_make_chunk("c", "prose leaf"),
    )
    table_leaf = ChunkNode(
        id="t",
        parent_id="s",
        depth=2,
        structured=StructuredTable(columns=["A"], rows=[{"A": "1"}]),
    )
    inner_leaf = ChunkNode(
        id="n",
        parent_id="i",
        depth=3,
        chunk=_make_chunk("n", "nested prose"),
    )
    inner = ChunkNode(id="i", parent_id="s", depth=2, children_ids=["n"])
    section = ChunkNode(id="s", parent_id="r", depth=1, children_ids=["c", "t", "i"])
    root = ChunkNode(id="r", depth=0, children_ids=["s"])
    entries = build_outline(_make_tree([root, section, chunk_leaf, table_leaf, inner, inner_leaf], "r"))

    by_id = {entry.node_id: entry for entry in entries}
    assert by_id["s"].kinds == ("chunk", "structured", "internal")
    assert by_id["t"].payload_chars > 0
    assert by_id["i"].kinds == ("chunk",)


def test_subtree_chars_sums_all_payloads() -> None:
    """subtree_chars(root) equals the sum of every payload in the subtree."""
    inner_leaf = ChunkNode(
        id="n",
        parent_id="i",
        depth=3,
        heading_chain=("A", "B"),
        chunk=_make_chunk("n", "deep leaf payload"),
    )
    inner = ChunkNode(
        id="i",
        parent_id="s",
        depth=2,
        heading_chain=("A", "B"),
        children_ids=["n"],
        summary="inner summary",
    )
    section = ChunkNode(id="s", parent_id="r", depth=1, heading_chain=("A",), children_ids=["i"])
    root = ChunkNode(id="r", depth=0, children_ids=["s"])
    entries = build_outline(_make_tree([root, section, inner, inner_leaf], "r"))

    by_id = {entry.node_id: entry for entry in entries}
    assert by_id["r"].subtree_chars == sum(entry.payload_chars for entry in entries)
    assert by_id["n"].subtree_chars == by_id["n"].payload_chars
    # The inner node's own payload is its summary (summarizer projection).
    assert by_id["i"].payload_chars == len("inner summary")


def test_summary_preferred_over_chunk_text() -> None:
    """A summarized node projects its summary, like the summarizer input."""
    node = ChunkNode(
        id="s",
        depth=1,
        summary="the summary wins",
        chunk=_make_chunk("s", "the chunk text loses"),
    )
    root = ChunkNode(id="r", depth=0, children_ids=["s"])
    entries = build_outline(_make_tree([root, node], "r"))

    by_id = {entry.node_id: entry for entry in entries}
    assert by_id["s"].payload_chars == len("the summary wins")
    assert by_id["s"].preview == "the summary wins"


def test_preview_collapses_whitespace_unicode_safe_and_bounded() -> None:
    """Newlines/tabs collapse, unicode survives, preview_chars is respected."""
    text = "line one\n\tline  two   with  spaces ✓✓✓ and 日本語テキスト here"
    node = ChunkNode(id="l", depth=1, chunk=_make_chunk("l", text))
    root = ChunkNode(id="r", depth=0, children_ids=["l"])

    full = build_outline(_make_tree([root, node], "r"), preview_chars=200)
    assert full[1].preview == "line one line two with spaces ✓✓✓ and 日本語テキスト here"

    short = build_outline(_make_tree([root, node], "r"), preview_chars=10)
    assert short[1].preview == full[1].preview[:10]
    assert len(short[1].preview) == 10


def test_determinism() -> None:
    """Two builds over the same tree produce identical entries and bytes."""
    leaf = ChunkNode(
        id="l",
        parent_id="s",
        depth=2,
        heading_chain=("Intro",),
        chunk=_make_chunk("l", "stable text"),
    )
    section = ChunkNode(
        id="s",
        parent_id="r",
        depth=1,
        heading_chain=("Intro",),
        children_ids=["l"],
    )
    root = ChunkNode(id="r", depth=0, children_ids=["s"])
    tree = _make_tree([root, section, leaf], "r")

    first = build_outline(tree)
    second = build_outline(tree)
    assert first == second
    assert isinstance(first[0], OutlineNode)
    assert outline_to_prompt(first).encode("utf-8") == outline_to_prompt(second).encode("utf-8")


def test_prompt_rendering_lists_every_entry_untruncated() -> None:
    """One indented line per entry with all columns and the full preview."""
    leaf = ChunkNode(
        id="leaf-1",
        parent_id="s",
        depth=2,
        heading_chain=("Intro", "Detail"),
        chunk=_make_chunk("leaf-1", "some preview text"),
    )
    section = ChunkNode(
        id="s",
        parent_id="r",
        depth=1,
        heading_chain=("Intro",),
        children_ids=["leaf-1"],
    )
    root = ChunkNode(id="r", depth=0, children_ids=["s"])
    text = outline_to_prompt(build_outline(_make_tree([root, section, leaf], "r")))

    lines = text.split("\n")
    assert len(lines) == 3
    assert lines[0] == "r | 0 |  | 1 | 17 | internal | "
    assert lines[1] == "  s | 1 | Intro | 1 | 17 | chunk | "
    assert lines[2].startswith("    leaf-1 | 2 | Intro / Detail | 0 | 17 | - | some preview text")
    # Indentation follows depth; no entry is truncated.
    assert "leaf-1" in lines[2]
    assert "some preview text" in lines[2]


def test_estimate_tokens_reuses_the_shared_counter() -> None:
    """estimate_tokens is the shared density helper under one name."""
    assert estimate_tokens("hello world, this is a test") == conservative_tokens("hello world, this is a test")
    assert estimate_tokens("") == 0


def test_scale_guard() -> None:
    """A 26k-node synthetic tree builds in under a second."""
    leaves = [ChunkNode(id=f"l{index}", parent_id="r", depth=1, chunk=_make_chunk(f"l{index}", f"leaf {index}")) for index in range(26000)]
    root = ChunkNode(id="r", depth=0, children_ids=[leaf.id for leaf in leaves])
    tree = _make_tree([root, *leaves], "r")

    start = time.perf_counter()
    entries = build_outline(tree)
    elapsed = time.perf_counter() - start

    assert len(entries) == 26001
    assert elapsed < 1.0


def _make_chunk(cid: str, text: str) -> Chunk:
    """Build a Chunk with required fields."""
    return Chunk(
        id=cid,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[],
        source_file=Path("doc.md"),
        token_count=max(1, len(text) // 4),
    )


def _make_tree(nodes: list[ChunkNode], root_id: str) -> ChunkTree:
    """Build a ChunkTree from nodes keyed by id."""
    return ChunkTree(
        root_id=root_id,
        nodes={node.id: node for node in nodes},
        source_file=Path("doc.md"),
        strategy=ChunkStrategy.TWO_LEVEL,
    )
