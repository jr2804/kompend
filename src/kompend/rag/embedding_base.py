"""Embedding-provider interface shared by the RAG layer and its consumers.

Deliberately import-light: the heavy implementations (pydantic-ai-backed
text embedder in ``vector_store``, httpx-backed media embedder in
``multimodal``) depend on this module, never the reverse. Code that only
handles providers abstractly — dedup matching, memory, test fixtures —
imports the interface from here without paying their import cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MediaEmbeddingUnsupported(RuntimeError):
    """The embedding model cannot consume images.

    Raised when the server answers that the model does not support media
    embeddings. Not transient: the same model will answer the same way, so
    the caller must not retry and must not pretend the image was embedded.
    """


class EmbeddingProvider(ABC):
    """Interface for embedding text into fixed-dimension vectors."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a normalized embedding vector for ``text``."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality."""
        ...

    @property
    def model_id(self) -> str:
        """The identity of the model behind the vectors.

        Recorded in the build manifest so a model change invalidates the
        index: vectors from one model do not describe the same space as the
        vectors of another. Providers that cannot name their model return
        the empty string — the index then cannot tell them apart.
        """
        return ""

    def embed_media(self, text: str, image: bytes) -> list[float]:
        """Return the joint embedding of ``text`` and one image.

        Only models with an image encoder can do this. Providers that cannot
        raise :class:`~kompend.rag.embedding_base.MediaEmbeddingUnsupported`,
        so the caller can index the text alone and say so, instead of
        silently dropping the image.
        """
        raise MediaEmbeddingUnsupported(f"{type(self).__name__} cannot embed images")
