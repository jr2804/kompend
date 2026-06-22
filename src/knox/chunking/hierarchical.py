"""KNOX Hierarchical Chunker — builds a ChunkTree from a ParsedDocument.

Strategy:
1. Walk elements in reading order, mirroring heading hierarchy into a tree.
2. HEADING elements open/close internal ChunkNodes at the appropriate depth.
3. ATOMIC_TYPES (TABLE, IMAGE, FORMULA, CODE) become leaf ChunkNodes with
   ``.structured`` populated (tables/equations/figures) or ``.chunk`` (code).
4. TEXT / LIST_ITEM runs are accumulated under the current heading and split
   via Chonkie's RecursiveChunker when they exceed the effective token budget.
5. Token counts are computed bottom-up so each internal node knows its
   subtree size — the basis for adaptive retrieval (Phase 4).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from chonkie import RecursiveChunker, RecursiveLevel, RecursiveRules

from knox.chunking.base import BaseChunker
from knox.chunking.extractors import extract_structured
from knox.core.models import (
    ATOMIC_TYPES,
    Chunk,
    ChunkNode,
    ChunkStrategy,
    ChunkTree,
    DocumentElement,
    ElementType,
    ParsedDocument,
)

logger = logging.getLogger(__name__)


class HierarchicalChunker(BaseChunker):
    """Structure-aware chunker that builds a :class:`ChunkTree`.

    The tree mirrors the document's heading hierarchy.  Atomic elements
    (tables, figures, formulas) are preserved as :class:`StructuredData`
    on leaf nodes.  Text runs are split via Chonkie's RecursiveChunker
    when they exceed the effective token budget.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        reserve_ratio: float = 0.2,
        tokenizer_name: str = "cl100k_base",
        context_window: Any | None = None,
        chunk_size_min: int = 2048,
        chunk_size_max: int = 131072,
        summary_chunk_threshold: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            reserve_ratio=reserve_ratio,
            context_window=context_window,
            chunk_size_min=chunk_size_min,
            chunk_size_max=chunk_size_max,
        )
        self.tokenizer_name = tokenizer_name
        self._summary_chunk_threshold = summary_chunk_threshold

    def chunk(self, document: ParsedDocument) -> ChunkTree:
        """Build a hierarchical chunk tree from *document*."""
        if not document.elements:
            return self._empty_tree(document)

        builder = _TreeBuilder(self, document)
        return builder.build()

    def _choose_strategy(self, total_tokens: int) -> ChunkStrategy:
        """Select tree strategy based on document size vs context budget.

        Uses **chunk-count** as the scaling metric: root summary quality
        degrades with the number of chunks to compress, not total document
        size.  Threshold derived from summary budget (~150t per chunk
        summary, ~1500t root summary cap → 10 chunks max for two_level).
        """
        if total_tokens <= self._effective_max:
            return ChunkStrategy.SINGLE_PASS
        num_chunks = math.ceil(total_tokens / self._effective_max)
        if num_chunks <= self._summary_chunk_threshold:
            return ChunkStrategy.TWO_LEVEL
        return ChunkStrategy.MULTI_RESOLUTION

    def _split_text(self, text: str) -> list[str]:
        """Split *text* into pieces that fit the effective token budget."""
        try:
            rc = RecursiveChunker(
                chunk_size=self._effective_max,
                tokenizer=self.tokenizer_name,
                rules=_boundary_rules(),
            )
            result = rc.chunk(text)
        except Exception:
            logger.exception("Chonkie chunking failed; returning unsplit text")
            return [text]

        if not result:
            return [text] if text.strip() else []
        return [c.text for c in result]

    @staticmethod
    def _empty_tree(document: ParsedDocument) -> ChunkTree:
        root_id = f"{document.source_path.stem}_root"
        return ChunkTree(
            root_id=root_id,
            nodes={root_id: ChunkNode(id=root_id, depth=0)},
            source_file=document.source_path,
            strategy=ChunkStrategy.SINGLE_PASS,
        )


# ── helpers ─────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


def _boundary_rules() -> RecursiveRules:
    """Paragraph → line → sentence → whitespace fallback."""
    return RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=["\n\n"], include_delim="prev"),
            RecursiveLevel(delimiters=["\n"], include_delim="prev"),
            RecursiveLevel(delimiters=[". ", "? ", "! "], include_delim="prev"),
            RecursiveLevel(include_delim="prev"),
        ],
    )


class _TreeBuilder:
    """Internal state machine that walks elements and assembles a ChunkTree."""

    def __init__(self, chunker: HierarchicalChunker, document: ParsedDocument) -> None:
        self._chunker = chunker
        self._doc = document
        self._stem = document.source_path.stem

        self._root_id = f"{self._stem}_root"
        self._nodes: dict[str, ChunkNode] = {
            self._root_id: ChunkNode(id=self._root_id, depth=0),
        }
        # Stack of (heading_level, heading_text, node_id).
        self._stack: list[tuple[int, str, str]] = [(0, "", self._root_id)]
        self._text_buffer: list[tuple[int, str]] = []
        self._counter = 0

    def build(self) -> ChunkTree:
        """Walk all elements and return the assembled tree."""
        for idx, el in enumerate(self._doc.elements):
            if el.type == ElementType.HEADING:
                self._handle_heading(idx, el)
            elif el.type in ATOMIC_TYPES:
                self._handle_atomic(idx, el)
            elif el.type == ElementType.METADATA:
                continue
            else:
                self._text_buffer.append((idx, str(el.content)))

        self._flush_text()
        total = self._compute_subtree_tokens(self._root_id)
        strategy = self._chunker._choose_strategy(total)

        return ChunkTree(
            root_id=self._root_id,
            nodes=self._nodes,
            source_file=self._doc.source_path,
            strategy=strategy,
        )

    # ── element handlers ────────────────────────────────────────────────

    def _handle_heading(self, idx: int, el: DocumentElement) -> None:
        self._flush_text()
        level = min(el.heading_level or 1, 6)
        text = str(el.content)

        while len(self._stack) > 1 and self._stack[-1][0] >= level:
            self._stack.pop()

        pid = self._stack[-1][2]
        p_depth = self._nodes[pid].depth
        hpath = self._heading_path() + (text,)
        node_id = self._next_id("h")
        self._nodes[node_id] = ChunkNode(
            id=node_id,
            parent_id=pid,
            depth=p_depth + 1,
            heading_path=hpath,
        )
        self._nodes[pid].children_ids.append(node_id)
        self._stack.append((level, text, node_id))

    def _handle_atomic(self, idx: int, el: DocumentElement) -> None:
        self._flush_text()
        pid = self._stack[-1][2]
        p_depth = self._nodes[pid].depth
        hpath = self._heading_path()
        leaf_id = self._next_id("a")

        structured = extract_structured(el)
        text_repr = _element_to_text(el)
        chunk = Chunk(
            id=leaf_id,
            text=text_repr,
            element_types=[el.type],
            source_elements=[idx],
            source_file=self._doc.source_path,
            heading_context=" / ".join(hpath),
            metadata=dict(el.metadata) if el.metadata else {},
        )
        self._nodes[leaf_id] = ChunkNode(
            id=leaf_id,
            parent_id=pid,
            depth=p_depth + 1,
            heading_path=hpath,
            chunk=chunk,
            structured=structured,
            token_count=_estimate_tokens(text_repr),
        )
        self._nodes[pid].children_ids.append(leaf_id)

    # ── text accumulation ───────────────────────────────────────────────

    def _flush_text(self) -> None:
        if not self._text_buffer:
            return
        pid = self._stack[-1][2]
        p_depth = self._nodes[pid].depth
        hpath = self._heading_path()
        full_text = "\n\n".join(t for _, t in self._text_buffer)
        el_indices = [i for i, _ in self._text_buffer]
        est_tokens = _estimate_tokens(full_text)

        if est_tokens <= self._chunker._effective_max:
            self._add_text_leaf(pid, p_depth + 1, hpath, full_text, el_indices, est_tokens)
        else:
            for piece in self._chunker._split_text(full_text):
                self._add_text_leaf(
                    pid,
                    p_depth + 1,
                    hpath,
                    piece,
                    el_indices,  # ponytail: each piece gets ALL buffer indices (provenance ceiling). Fix would require tracking char ranges through Chonkie splits.
                    _estimate_tokens(piece),
                )
        self._text_buffer = []

    def _add_text_leaf(
        self,
        pid: str,
        depth: int,
        hpath: tuple[str, ...],
        text: str,
        el_indices: list[int],
        tokens: int,
    ) -> None:
        leaf_id = self._next_id("t")
        chunk = Chunk(
            id=leaf_id,
            text=text,
            element_types=[ElementType.TEXT],
            source_elements=el_indices,
            source_file=self._doc.source_path,
            heading_context=" / ".join(hpath),
        )
        self._nodes[leaf_id] = ChunkNode(
            id=leaf_id,
            parent_id=pid,
            depth=depth,
            heading_path=hpath,
            chunk=chunk,
            token_count=tokens,
        )
        self._nodes[pid].children_ids.append(leaf_id)

    # ── token computation ───────────────────────────────────────────────

    def _compute_subtree_tokens(self, node_id: str) -> int:
        node = self._nodes[node_id]
        if not node.children_ids:
            return node.token_count
        total = sum(self._compute_subtree_tokens(c) for c in node.children_ids)
        node.token_count = total
        return total

    # ── utilities ───────────────────────────────────────────────────────

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{self._stem}_{prefix}{self._counter}"

    def _heading_path(self) -> tuple[str, ...]:
        return tuple(text for _, text, _ in self._stack[1:])


# ── module-level conversion helpers ─────────────────────────────────────


def _element_to_text(el: DocumentElement) -> str:
    """Flatten any element content to a text representation."""
    if isinstance(el.content, str):
        return el.content
    if isinstance(el.content, (list, tuple)):
        return "\n".join(" | ".join(str(c) for c in row) for row in el.content)
    if isinstance(el.content, bytes):
        return f"[binary data: {len(el.content)} bytes]"
    return str(el.content)
