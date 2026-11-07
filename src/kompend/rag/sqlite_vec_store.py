"""Compatibility surface for the sqlite-vec store.

The implementation lives in :mod:`kompend.rag.sqlite_vec`. This module path is
still imported by the published documentation
(``docs/integrations/open-webui.md``), so it keeps resolving to exactly the two
names it always exported. In-repo code imports from the package root
(``from kompend.rag import SqliteVecWikiStore``).
"""

from __future__ import annotations

from kompend.rag.sqlite_vec import SqliteVecWikiStore, recorded_embedding_model

__all__ = ["SqliteVecWikiStore", "recorded_embedding_model"]
