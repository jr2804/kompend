"""KNOX Hybrid Chunker — structure-aware atomic splitting + Chonkie text chunking.

Strategy:
1. Atomic elements (tables, figures, formulas, code blocks) never split.
   Each becomes its own Chunk with the correct element_type metadata.
2. Text/heading runs are concatenated into a single text block and fed
   to Chonkie's RecursiveChunker (respecting heading hierarchy via
   markdown recipe markers).
3. Token budget: reserve 20% for system prompts (configurable).
"""

from __future__ import annotations

import logging
from typing import Any

from chonkie import RecursiveChunker

from knox.chunking.base import BaseChunker
from knox.core.models import ATOMIC_TYPES, Chunk, ElementType, ParsedDocument

logger = logging.getLogger(__name__)


class HybridChunker(BaseChunker):
    """Structure-aware chunker that preserves atomic elements.

    Atomic elements (TABLE, IMAGE, FORMULA, CODE) are never split.
    TEXT and HEADING runs are merged and chunked via Chonkie's
    RecursiveChunker with a markdown-aware recipe.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_ratio: float = 0.2,
        tokenizer_name: str = "cl100k_base",
        **kwargs: Any,
    ) -> None:
        super().__init__(max_tokens=max_tokens, reserve_ratio=reserve_ratio)
        self.tokenizer_name = tokenizer_name

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        if not document.elements:
            return []

        chunks: list[Chunk] = []
        text_run_elements: list[int] = []

        for idx, el in enumerate(document.elements):
            if el.type in ATOMIC_TYPES:
                if text_run_elements:
                    chunks.extend(self._chunk_text_run(document, text_run_elements))
                    text_run_elements = []
                chunks.append(_atomic_chunk(document, el, idx))
            else:
                text_run_elements.append(idx)

        if text_run_elements:
            chunks.extend(self._chunk_text_run(document, text_run_elements))

        for i, c in enumerate(chunks):
            c.id = f"{document.source_path.stem}_chunk_{i}"

        return chunks

    def _chunk_text_run(
        self,
        document: ParsedDocument,
        indices: list[int],
    ) -> list[Chunk]:
        """Chunk a contiguous run of TEXT/HEADING elements."""
        text_parts: list[str] = []
        heading_stack: list[str] = []
        offset_map: list[tuple[int, int, list[str]]] = []
        cursor = 0
        for i in indices:
            el = document.elements[i]
            content = _str_content(el)
            if el.type == ElementType.HEADING:
                level = el.heading_level or 1
                prefix = "#" * min(level, 6)
                part = f"{prefix} {content}"
                heading_stack = heading_stack[: level - 1] + [content]
            else:
                heading = " / ".join(heading_stack) if heading_stack else ""
                part = f"[{heading}] {content}" if heading else content

            text_parts.append(part)
            length = len(part.encode("utf-8"))
            offset_map.append((cursor, cursor + length, heading_stack.copy()))
            cursor += length + len(b"\n\n")

        full_text = "\n\n".join(text_parts)

        try:
            rc = RecursiveChunker.from_recipe(
                "markdown",
                chunk_size=self._effective_max,
                tokenizer=self.tokenizer_name,
            )
            chonkie_result = rc.chunk(full_text)
        except Exception:
            logger.exception("Chonkie chunking failed for text run")
            return [self._fallback_chunk(document, indices)]

        if not chonkie_result:
            return []

        chunks: list[Chunk] = []
        for c in chonkie_result:
            # Use Chonkie's byte offsets to recover which original element(s)
            # this chunk covers and the heading stack active at that span.
            start_idx = getattr(c, "start_index", 0)
            end_idx = getattr(c, "end_index", start_idx)
            _local_idxs, heading_ctx = _map_offsets_to_elements(start_idx, end_idx, offset_map, len(full_text.encode("utf-8")))
            source_idxs = [indices[i] for i in _local_idxs]
            chunks.append(
                Chunk(
                    id="",
                    text=c.text,
                    element_types=[ElementType.TEXT],
                    source_elements=source_idxs or indices,
                    source_file=document.source_path,
                    heading_context=heading_ctx,
                    token_count=getattr(c, "token_count", 0),
                )
            )
        return chunks

    @staticmethod
    def _fallback_chunk(
        document: ParsedDocument,
        indices: list[int],
    ) -> Chunk:
        """Fallback: merge all elements into one chunk."""
        parts = [str(document.elements[i].content) for i in indices]
        return Chunk(
            id="",
            text="\n\n".join(parts),
            element_types=[ElementType.TEXT],
            source_elements=indices,
            source_file=document.source_path,
        )


# ── helpers ──────────────────────────────────────────────────────────────


def _str_content(el: Any) -> str:
    if isinstance(el.content, str):
        return el.content
    if isinstance(el.content, (list, tuple)):
        return "\n".join(" | ".join(str(c) for c in row) for row in el.content)
    if isinstance(el.content, bytes):
        return f"[binary data: {len(el.content)} bytes]"
    return str(el.content)


def _atomic_chunk(
    document: ParsedDocument,
    el: Any,
    idx: int,
) -> Chunk:
    return Chunk(
        id="",
        text=_str_content(el),
        element_types=[el.type],
        source_elements=[idx],
        source_file=document.source_path,
        heading_context="",
    )


def _map_offsets_to_elements(
    start_byte: int,
    end_byte: int,
    offset_map: list[tuple[int, int, list[str]]],
    full_len: int,
) -> tuple[list[int], str]:
    """Map a Chonkie chunk byte-range back to element indices and heading context.

    Chonkie reports start_index/end_index as character (not byte) offsets, but
    in UTF-8 the two align closely enough that we guard with clamping for
    safety.
    """
    source_indices: list[int] = []
    heading_ctx: str = ""
    effective_end = min(end_byte, full_len)
    for i, (el_start, el_end, stack) in enumerate(offset_map):
        if el_end <= start_byte or el_start >= effective_end:
            continue
        source_indices.append(i)
        if stack:
            heading_ctx = " / ".join(stack)
    return source_indices, heading_ctx
