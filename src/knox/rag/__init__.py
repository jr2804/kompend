"""KNOX Hybrid RAG module — sidecar vector store for semantic search."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import CompiledWiki
from knox.rag.zvec_store import ZvecWikiStore

__all__ = ["ZvecWikiStore", "build_vector_index"]


def build_vector_index(
    wiki: CompiledWiki,
    output_dir: Path,
    embed_model: str | None = None,
) -> int:
    """Build a vector index for semantic search over *wiki*.

    Returns the number of indexed nodes (0 if nothing was indexed).
    """
    store = ZvecWikiStore(db_path=output_dir / ".knox" / "vectors")
    count = store.build_index(wiki, embedding_model=embed_model)
    store.close()
    return count
