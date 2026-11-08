"""The pydantic-ai-backed embedding provider.

pydantic-ai pulls the provider SDKs (anthropic, openai, typesafe, …) into
interpreter startup, so the provider that needs them lives here — apart from
the import-light contract in :mod:`kompend.rag.vector_store` and the
interface in :mod:`kompend.rag.embedding_base`.

Import this module where a provider is actually built; never from a module
that only needs the protocol.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from kompend.llm.constants import OLLAMA_DEFAULT_BASE_URL
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.multimodal import MediaEmbeddingUnsupported, OllamaMediaEmbedder
from kompend.rag.vector_store import to_pydantic_ai_embedding_string


class PydanticAIEmbeddingProvider(EmbeddingProvider):
    """Embedding via pydantic-ai's :class:`~pydantic_ai.Embedder`.

    Supports every remote embedding provider pydantic-ai ships (openai,
    ollama, google, cohere, bedrock).
    The *model_string* uses the same ``provider:model`` convention as
    :func:`kompend.llm.models.build_model`.
    """

    def __init__(self, model_string: str, *, dim: int | None = None) -> None:
        self._model_string = model_string
        self._embedder = build_embedder(model_string)
        #: Known vector dimensionality shortcut — set by callers (tests, factory
        #: sites) that already know the dimension, so ``.dim`` does not need a
        #: probe embed. ``None`` keeps the auto-detect path.
        self._dim: int | None = dim
        self._media: OllamaMediaEmbedder | None = None
        self._media_checked = False

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Determine dimensionality by embedding a short probe text.
            self._dim = len(self.embed("dimension probe"))
        return self._dim

    @property
    def model_id(self) -> str:
        """The model behind the vectors, in the ``provider/model`` config form."""
        provider, separator, model = self._model_string.partition(":")
        if not separator:
            return self._model_string
        return f"{provider}/{model}"

    def embed(self, text: str) -> list[float]:
        result = self._embedder.embed_documents_sync(text)
        vector = list(result.embeddings[0])
        if self._dim is None:
            self._dim = len(vector)
        return vector

    def embed_media(self, text: str, image: bytes) -> list[float]:
        """Embed text and one image jointly through Ollama's media endpoint."""
        if not self._media_checked:
            self._media = self._build_media_embedder()
            self._media_checked = True
        if self._media is None:
            raise MediaEmbeddingUnsupported(f"Embedding model '{self._model_string}' cannot embed images")
        return self._media.embed(text, image)

    def _build_media_embedder(self) -> OllamaMediaEmbedder | None:
        """The Ollama media embedder for this model, or ``None`` for other providers."""
        if not self._model_string.startswith("ollama:"):
            return None
        return OllamaMediaEmbedder(self._model_string.split(":", 1)[1])


def build_embedder(model_string: str) -> Any:
    """Build a pydantic-ai ``Embedder`` with Ollama base_url support.

    Mirrors the pattern in :mod:`kompend.llm.models`: when the provider is
    Ollama, explicitly construct ``OllamaProvider`` with the local default
    ``base_url`` so it works out-of-the-box without ``OLLAMA_BASE_URL``.

    Returns ``pydantic_ai.Embedder``; the annotation is ``Any`` because the
    pydantic-ai imports are call-site (``import_module``, per the repo-wide
    runtime-only-import rule) and ``TYPE_CHECKING`` blocks are banned — this
    module is reachable from lint report rendering, which must stay
    import-light.
    """
    pydantic_ai = import_module("pydantic_ai")
    if model_string.startswith("ollama:"):
        ollama_providers = import_module("pydantic_ai.providers.ollama")
        base_url = os.getenv("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL
        provider = ollama_providers.OllamaProvider(base_url=base_url)
        embeddings = import_module("pydantic_ai.embeddings")
        model = embeddings.infer_embedding_model(model_string, provider_factory=lambda _: provider)
        return pydantic_ai.Embedder(model)
    return pydantic_ai.Embedder(model_string)


def make_provider(
    embedding_model: str,
    dim: int | None = None,
) -> EmbeddingProvider:
    """Select an embedding provider based on model name.

    Uses the same ``provider/model`` or ``provider:model`` convention as
    :func:`kompend.llm.models.build_model`. An explicit provider prefix is
    required — there is no fallback to a local embedding provider.
    """
    pydantic_ai_string = to_pydantic_ai_embedding_string(embedding_model)
    return PydanticAIEmbeddingProvider(model_string=pydantic_ai_string, dim=dim)
