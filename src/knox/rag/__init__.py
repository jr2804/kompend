"""KNOX Hybrid RAG module — sidecar vector store for semantic search."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import CompiledWiki
from knox.core.models import CompiledWiki as _CW
from knox.core.paths import VaultPaths
from knox.rag.zvec_store import ZvecField, ZvecWikiStore

__all__ = ["ZvecField", "ZvecWikiStore", "build_vector_index"]


def build_vector_index(
    wiki: CompiledWiki,
    output_dir: Path,
    embed_model: str,
    *,
    base_dir: Path | None = None,
    batch_size: int = 10,
    embed_assets: bool = False,
) -> int:
    """Build a vector index for semantic search over *wiki*.

    Args:
        wiki: Compiled wiki to index.
        output_dir: Wiki output directory.
        embed_model: Embedding model string.
        base_dir: Vault base directory.
        batch_size: Embedding batch size.
        embed_assets: When ``False`` (default), nodes tagged
            ``source_asset=True`` are excluded from the index.

    Returns:
        Number of indexed nodes (0 if nothing was indexed).
    """
    if not embed_model:
        raise ValueError("An explicit 'embed_model' must be provided to build the vector index.")

    nodes = wiki.nodes
    if not embed_assets:
        nodes = [n for n in nodes if not (n.metadata or {}).get("source_asset", False)]

    if not nodes:
        return 0
    filtered = _CW(
        nodes=nodes,
        edges=wiki.edges,
        toc=wiki.toc,
        metadata=wiki.metadata,
    )

    resolved_base = base_dir if base_dir is not None else output_dir.parent
    store = ZvecWikiStore(db_file=VaultPaths(resolved_base).vectors_dir)
    count = store.build_index(filtered, embedding_model=embed_model, batch_size=batch_size, base_dir=resolved_base)
    store.close()
    return count
