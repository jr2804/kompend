"""Knowledge graph compiler."""

from __future__ import annotations

from knox.compilation.base import BaseCompiler
from knox.core.models import CompiledWiki, ResolvedChunk, WikiNode


class GraphCompiler(BaseCompiler):
    """Stub: Assembles resolved chunks into a knowledge graph."""

    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        nodes = [
            WikiNode(
                id=f"node_{i}",
                title=f"Chunk {i}",
                content=rc.chunk.text,
                node_type="section",
                source_chunks=[rc.chunk.id],
            )
            for i, rc in enumerate(chunks)
        ]
        return CompiledWiki(
            nodes=nodes,
            edges=[],
            toc=[{"title": n.title, "id": n.id} for n in nodes],
            metadata={"chunk_count": len(chunks)},
        )
