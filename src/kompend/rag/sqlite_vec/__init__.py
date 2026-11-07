"""sqlite-vec backend — the vec0 implementation of :class:`VectorStoreLike`.

The package holds the store itself; ``kompend.rag.sqlite_vec_store`` remains
as a compatibility module because that path is a documented import, while
new code imports from the package root (``from kompend.rag import
SqliteVecWikiStore``).
"""

from __future__ import annotations

from kompend.rag.sqlite_vec.store import SqliteVecWikiStore, recorded_embedding_model

__all__ = ["SqliteVecWikiStore", "recorded_embedding_model"]
