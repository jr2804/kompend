"""Structural chunk-tree outline for decision-model traversal planning.

Pure structure, no LLM: one bottom-up pass over the tree yields per-node
size and shape facts the traversal planner batches by level. Deterministic
in document order, so a plan can be cached and diffed.

Layering: this module must stay importable without ``kompend.agents`` —
chunking must not pull pydantic-ai. Only core models, core text helpers,
and the standard library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from kompend.core.models.structured import structured_to_text
from kompend.core.models.trees import ChunkNode, ChunkTree
from kompend.core.text.density import conservative_tokens


#: Content kind of a child that carries a chunk.
KIND_CHUNK: Final[str] = "chunk"
#: Content kind of a child that carries structured data.
KIND_STRUCTURED: Final[str] = "structured"
#: Content kind of a child that carries neither (an inner node).
KIND_INTERNAL: Final[str] = "internal"


@dataclass(frozen=True)
class OutlineNode:
    """Size and shape facts for one chunk-tree node.

    Attributes:
        node_id: The node's id in the tree.
        heading_chain: The headings above the node, root-first.
        depth: Depth in the tree (root is 0).
        child_count: Children present in the tree, in document order.
        payload_chars: Characters of the node's own payload (summary, else
            chunk text, else structured text — the same projection the
            summarizer feeds the model).
        subtree_chars: ``payload_chars`` plus the ``subtree_chars`` of every
            child — the whole subtree rooted here, inclusive.
        token_estimate: The payload in tokens, counted once, with the shared
            density helper.
        kinds: Distinct content kinds of the present children, in
            first-appearance order: ``chunk`` (carries a chunk), ``structured``
            (carries structured data), ``internal`` (neither — an inner node
            whose content, if any, is its summary).
        preview: First ``preview_chars`` of the node's own payload,
            whitespace-collapsed.
    """

    node_id: str
    heading_chain: tuple[str, ...]
    depth: int
    child_count: int
    payload_chars: int
    subtree_chars: int
    token_estimate: int
    kinds: tuple[str, ...]
    preview: str


def build_outline(tree: ChunkTree, *, preview_chars: int = 200) -> list[OutlineNode]:
    """Build the structural outline of *tree* in one bottom-up pass.

    Walks the tree once (``O(nodes)``), accumulates subtree sizes from the
    leaves up, and returns one entry per node in document order. No cache,
    no I/O, no model.

    Args:
        tree: The chunk tree to outline.
        preview_chars: Characters of own payload kept in each preview.

    Returns:
        One :class:`OutlineNode` per node, in ``tree.walk()`` order.
    """
    ordered = list(tree.walk())
    subtree_chars: dict[str, int] = {}
    entries: dict[str, OutlineNode] = {}
    for node in reversed(ordered):
        payload = _own_content(node)
        children = [tree.nodes[child_id] for child_id in node.children_ids if child_id in tree.nodes]
        kinds: list[str] = []
        for child in children:
            kind = _child_kind(child)
            if kind not in kinds:
                kinds.append(kind)
        total = len(payload) + sum(subtree_chars[child.id] for child in children)
        subtree_chars[node.id] = total
        entries[node.id] = OutlineNode(
            node_id=node.id,
            heading_chain=node.heading_chain,
            depth=node.depth,
            child_count=len(children),
            payload_chars=len(payload),
            subtree_chars=total,
            token_estimate=estimate_tokens(payload),
            kinds=tuple(kinds),
            preview=_preview(payload, preview_chars),
        )
    return [entries[node.id] for node in ordered]


def estimate_tokens(text: str) -> int:
    """Count *text* in tokens with the shared density helper.

    This is :func:`kompend.core.text.density.conservative_tokens` under one
    name — the same counter the budgeting path uses. Do not invent a second
    counter here; two counters drift and the plan no longer fits the window.
    """
    return conservative_tokens(text)


def _own_content(node: ChunkNode) -> str:
    """The node's own payload, mirroring the summarizer's projection.

    Same priority as ``NodeSummarizer._child_content`` (summary, then chunk
    text, then structured text): the outline describes what the summarizer
    would feed the model for this node.
    """
    if node.summary:
        return node.summary
    if node.chunk is not None:
        return node.chunk.text
    if node.structured is not None:
        return structured_to_text(node.structured)
    return ""


def _child_kind(child: ChunkNode) -> str:
    """Classify a present child by what it carries."""
    if child.chunk is not None:
        return KIND_CHUNK
    if child.structured is not None:
        return KIND_STRUCTURED
    return KIND_INTERNAL


def _preview(payload: str, preview_chars: int) -> str:
    """Collapse whitespace and take the first *preview_chars* characters.

    ``str.split`` without arguments splits on any whitespace run (spaces,
    newlines, tabs) and drops empties, so the join is single-spaced.
    Slicing is code-point based — multi-byte characters are never split
    into undecodable bytes.
    """
    return " ".join(payload.split())[:preview_chars]


def outline_to_prompt(entries: Sequence[OutlineNode]) -> str:
    """Render outline entries as a compact indented listing.

    One line per entry: indentation by depth, then
    ``node_id | depth | heading | children | chars | kinds | preview``,
    where ``chars`` is the inclusive subtree size and ``kinds`` joins with
    ``+`` (``-`` when the node has no children).

    Never truncates an entry: the rendered text is exact, so the caller
    batches by level with ``rendered_chars = len(text)`` — plain ``len``,
    no second counter.
    """
    lines: list[str] = []
    for entry in entries:
        indent = "  " * entry.depth
        heading = _cell(" / ".join(entry.heading_chain))
        kinds = "+".join(entry.kinds) if entry.kinds else "-"
        lines.append(f"{indent}{_cell(entry.node_id)} | {entry.depth} | {heading} | {entry.child_count} | {entry.subtree_chars} | {kinds} | {entry.preview}")
    return "\n".join(lines)


def _cell(value: str) -> str:
    """Collapse a metadata cell to one line for the prompt rendering."""
    return " ".join(value.split())
