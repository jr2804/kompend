"""Development stub for chonkie-free environments."""

from __future__ import annotations

from knox.chunking.base import BaseChunker
from knox.core.models import Chunk, ParsedDocument


class DevChunker(BaseChunker):
    """Naive chunker for development without chonkie installed."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        text_parts = [str(el.content) for el in document.elements if isinstance(el.content, str)]
        full_text = "\n\n".join(text_parts)
        if not full_text:
            return chunks

        start = 0
        i = 0
        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))
            chunk_text = full_text[start:end]
            chunks.append(
                Chunk(
                    id=f"{document.source_path.stem}_chunk_{i}",
                    text=chunk_text,
                    element_types=[],
                    source_elements=[i],
                    source_file=document.source_path,
                )
            )
            start += self.chunk_size - self.chunk_overlap
            i += 1
        return chunks
