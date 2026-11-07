"""Build the vector index — the entry point the compile and query paths call.

The function lives here rather than in the package ``__init__`` because it
needs the sqlite-vec backend: keeping it in ``kompend/rag/__init__.py`` would
mean importing the backend at package-import time, which is exactly the cost
the lazy boundary exists to avoid (measured: 2.3 s of interpreter startup).
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import CompiledWiki
from kompend.core.paths import VaultPaths
from kompend.rag import SqliteVecWikiStore


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
