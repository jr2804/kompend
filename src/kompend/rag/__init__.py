"""kompend Hybrid RAG module — sidecar vector store for semantic search."""

from __future__ import annotations

from pathlib import Path

import lazy_loader as lazy

from kompend.core.models import CompiledWiki
from kompend.core.paths import VaultPaths

# Runtime names arrive via lazy.attach's module __getattr__; these
# imports exist so type checkers and ruff see them defined.
from kompend.rag.sqlite_vec_store import SqliteVecWikiStore
from kompend.rag.vector_store import VectorField, indexable_nodes

# Everything heavy is attached lazily: importing ``kompend.rag`` — directly
# or as the package ``__init__`` of any submodule import — must not drag the
# pydantic-ai provider stack or the sqlite-vec backend into interpreter
# startup. ``SqliteVecWikiStore`` referenced inside :func:`build_vector_index`
# resolves through the module ``__getattr__`` on first call. The
# cross-encoder backend is an optional extra and pulls onnxruntime; it stays
# lazy for the same reason.
__getattr__, __dir__, _LAZY_EXPORTS = lazy.attach(
    __name__,
    submod_attrs={
        "embedding_pydantic": ["PydanticAIEmbeddingProvider", "build_embedder", "make_provider"],
        "flashrank_backend": ["build_request", "load_ranker"],
        "sqlite_vec_store": ["SqliteVecWikiStore"],
        "vector_store": ["VectorField", "indexable_nodes"],
    },
)

__all__ = ["SqliteVecWikiStore", "VectorField", "build_vector_index", "indexable_nodes"]


def build_vector_index(
    wiki: CompiledWiki,
    output_dir: Path,
    embed_model: str,
    *,
    base_dir: Path | None = None,
    batch_size: int = 10,
    embed_attachments: bool = False,
) -> int:
    """Build a vector index for semantic search over *wiki*.

    Build from the **same** node set the query path judges staleness against
    (``load_wiki_from_files``), or the index is rebuilt on the next query.

    Args:
        wiki: Compiled wiki to index.
        output_dir: Wiki output directory.
        embed_model: Embedding model string.
        base_dir: Vault base directory.
        batch_size: Embedding batch size.
        embed_attachments: When ``False`` (default), nodes tagged
            ``source_asset=True`` are excluded from the index.

    Returns:
        Number of indexed nodes (0 if nothing was indexed).
    """
    if not embed_model:
        raise ValueError("An explicit 'embed_model' must be provided to build the vector index.")

    resolved_base = base_dir if base_dir is not None else output_dir.parent
    store = SqliteVecWikiStore(db_file=VaultPaths(resolved_base).vectors_db_file)
    count = store.build_index(
        wiki,
        embedding_model=embed_model,
        batch_size=batch_size,
        base_dir=resolved_base,
        embed_attachments=embed_attachments,
    )
    store.close()
    return count
