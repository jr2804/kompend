"""Semantic chunking using the chonkie library."""

from __future__ import annotations

import logging
from typing import Any

from chonkie import RecursiveChunker, SemanticChunker

from knox.chunking.base import BaseChunker
from knox.core.models import Chunk, ElementType, ParsedDocument

logger = logging.getLogger(__name__)


class ChonkieChunker(BaseChunker):
    """Chunker adapter using chonkie library."""

    def __init__(
        self,
        chunker_type: str = "recursive",  # "recursive", "semantic", "table", "code"
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        **kwargs: Any,
    ) -> None:
        self.chunker_type = chunker_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

        if chunker_type == "recursive":
            self._chunker = RecursiveChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        elif chunker_type == "semantic":
            self._chunker = SemanticChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        else:
            self._chunker = RecursiveChunker(chunk_size=chunk_size, overlap=chunk_overlap)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        text = self._collect_text(document)
        if not text.strip():
            return []

        try:
            chonkie_chunks = self._chunker.chunk(text)
        except Exception:
            logger.exception("Chonkie chunking failed for %s", document.source_path)
            return []

        chunks: list[Chunk] = []
        for i, c in enumerate(chonkie_chunks):
            chunks.append(
                Chunk(
                    id=f"{document.source_path.stem}_chunk_{i}",
                    text=c.text if hasattr(c, "text") else str(c),
                    element_types=[ElementType.TEXT],
                    source_elements=[],
                    source_file=document.source_path,
                )
            )
        return chunks

    def _collect_text(self, document: ParsedDocument) -> str:
        parts: list[str] = []
        current_heading = ""
        for el in document.elements:
            if el.type == ElementType.HEADING:
                current_heading = str(el.content)
            elif el.type == ElementType.TEXT:
                text = str(el.content)
                if current_heading:
                    parts.append(f"# {current_heading}\n{text}")
                    current_heading = ""
                else:
                    parts.append(text)
            elif el.type == ElementType.CODE:
                parts.append(f"```\n{el.content}\n```")
            elif el.type == ElementType.TABLE and isinstance(el.content, list):
                parts.append(self._render_table(el.content))
        return "\n\n".join(parts)

    @staticmethod
    def _render_table(rows: list[list[str]]) -> str:
        lines = ["| " + " | ".join(row) + " |" for row in rows]
        if len(lines) > 1:
            sep = "|" + " --- |" * len(rows[0])
            lines.insert(1, sep)
        return "\n".join(lines)
