"""kompend Hybrid RAG module — sidecar vector store for semantic search."""

from __future__ import annotations

import lazy_loader as lazy

# Everything heavy is attached lazily: importing ``kompend.rag`` — directly
# or as the package ``__init__`` of any submodule import — must not drag the
# pydantic-ai provider stack or the sqlite-vec backend into interpreter
# startup (measured: the sqlite-vec backend alone cost 2.3 s). The
# cross-encoder backend is an optional extra and pulls onnxruntime; it stays
# lazy for the same reason.
#
# The names are NOT imported at module level to satisfy a type checker: an
# import here *is* the eager import this boundary exists to prevent. A
# function that needs a lazy name lives in the submodule that owns it — the
# index builder is in ``build.py``, next to the store it constructs.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "build": ["build_vector_index"],
        "embedding_pydantic": ["PydanticAIEmbeddingProvider", "build_embedder", "make_provider"],
        "flashrank_backend": ["build_request", "load_ranker"],
        "sqlite_vec": ["SqliteVecWikiStore", "recorded_embedding_model"],
        "vector_store": ["VectorField", "indexable_nodes"],
    },
)
