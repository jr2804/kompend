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

from knox.chunking.base import BaseChunker
from knox.core.models import Chunk, ElementType, ParsedDocument

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
        self._chonkie = _lazy_load_chonkie()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        if not document.elements:
            return []

        chunks: list[Chunk] = []
        text_run_elements: list[int] = []

        for idx, el in enumerate(document.elements):
            if el.type in _ATOMIC_TYPES:
                # Flush any pending text run first
                if text_run_elements:
                    chunks.extend(self._chunk_text_run(document, text_run_elements))
                    text_run_elements = []
                # Atomic element → its own chunk
                chunks.append(_atomic_chunk(document, el, idx))
            else:
                # TEXT or HEADING → accumulate for batch chunking
                text_run_elements.append(idx)

        # Flush remaining text run
        if text_run_elements:
            chunks.extend(self._chunk_text_run(document, text_run_elements))

        # Assign stable IDs
        for i, c in enumerate(chunks):
            c.id = f"{document.source_path.stem}_chunk_{i}"

        return chunks

    def _chunk_text_run(
        self,
        document: ParsedDocument,
        indices: list[int],
    ) -> list[Chunk]:
        """Chunk a contiguous run of TEXT/HEADING elements."""
        if self._chonkie is None:
            return [self._fallback_chunk(document, indices)]

        # Build markdown-like text preserving heading hierarchy
        text_parts: list[str] = []
        heading_stack: list[str] = []
        for i in indices:
            el = document.elements[i]
            content = _str_content(el)
            if el.type == ElementType.HEADING:
                level = el.heading_level or 1
                prefix = "#" * min(level, 6)
                text_parts.append(f"{prefix} {content}")
                heading_stack = heading_stack[: level - 1] + [content]
            else:
                heading = " / ".join(heading_stack) if heading_stack else ""
                if heading:
                    text_parts.append(f"[{heading}] {content}")
                else:
                    text_parts.append(content)

        full_text = "\n\n".join(text_parts)

        try:
            # Use RecursiveChunker with markdown recipe for heading-aware splitting
            rc = self._chonkie.RecursiveChunker.from_recipe(
                "markdown",
                chunk_size=self._effective_max,
            )
            chonkie_result = rc.chunk(full_text)
        except Exception:
            logger.exception("Chonkie chunking failed for text run")
            return [self._fallback_chunk(document, indices)]

        if not chonkie_result:
            return []

        return [
            Chunk(
                id="",  # assigned later
                text=c.text,
                element_types=[ElementType.TEXT],
                source_elements=indices,
                source_file=document.source_path,
                heading_context=_extract_heading(c.text),
            )
            for c in chonkie_result
        ]

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

_ATOMIC_TYPES: frozenset[ElementType] = frozenset({ElementType.TABLE, ElementType.IMAGE, ElementType.FORMULA, ElementType.CODE})


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
    """Create a single-element chunk for an atomic element."""
    return Chunk(
        id="",  # assigned later
        text=_str_content(el),
        element_types=[el.type],
        source_elements=[idx],
        source_file=document.source_path,
        heading_context="",
    )


def _extract_heading(text: str) -> str:
    """Extract the first heading from text for context."""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped.startswith("[") and "]" in stripped:
            return stripped[1:].split("]")[0]
    return ""


def _lazy_load_chonkie() -> Any:
    """Lazy-import chonkie module for optional availability."""
    try:
        import chonkie  # noqa: PLC0415  — lazy import for optional dep
        from chonkie import RecursiveChunker  # noqa: PLC0415, F401

        return chonkie
    except ImportError:
        logger.warning("chonkie not available — using fallback chunking")
        return None
