"""KNOX Hybrid RAG module — sidecar vector store for semantic search."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import CompiledWiki
from knox.core.paths import VaultPaths
from knox.rag.zvec_store import ZvecWikiStore

__all__ = ["ZvecWikiStore", "build_vector_index"]


def build_vector_index(
    wiki: CompiledWiki,
    output_dir: Path,
    embed_model: str,
    *,
    base_dir: Path | None = None,
    batch_size: int = 10,
) -> int:
    """Build a vector index for semantic search over *wiki*.

    Returns the number of indexed nodes (0 if nothing was indexed).
    """
    if not embed_model:
        raise ValueError("An explicit 'embed_model' must be provided to build the vector index.")

    resolved_base = base_dir if base_dir is not None else output_dir.parent
    store = ZvecWikiStore(db_file=VaultPaths(resolved_base).vectors_dir)
    count = store.build_index(wiki, embedding_model=embed_model, batch_size=batch_size)
    store.close()
    return count
