"""Multimodal embedding of text and an image through Ollama's media endpoint.

Ollama's ``POST /api/embed`` accepts a multimodal input object::

    {"model": "<model>", "input": {"text": "<text>", "image": "<raw base64>"}}

The shape below was pinned against a live Ollama 0.40 server:

* the image must be **raw** base64 — a ``data:image/png;base64,`` prefix is
  rejected with ``input.image: illegal base64 data at input byte 4``;
* the key is ``image`` (singular) and carries exactly one image; the embed
  route rejects ``images`` (``input.images must be a string`` /
  ``unknown input field "images"``), so a node with several figures is
  embedded from its first figure and the rest are reported, never dropped
  silently;
* a model that cannot consume images answers ``501 {"error": "model does not
  support media embeddings"}`` — a property of the model, not a transport
  failure, so it is raised as :class:`MediaEmbeddingUnsupported` and never
  retried;
* the response shape is ``{"embeddings": [[...]]}``.

Only Ollama exposes this endpoint today; every other provider keeps the
text-only path and reports the images as not embedded.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Final

import httpx

from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.constants import OLLAMA_DEFAULT_BASE_URL
from kompend.rag.embedding_base import MediaEmbeddingUnsupported

#: Ollama's own wording for "this model has no image encoder". Matched on the
#: message, not only the status: 501 alone is too weak a signal.
_MEDIA_UNSUPPORTED_MARKER: Final[str] = "media embeddings"

#: How much of a failed response body ends up in the error message.
_ERROR_BODY_LIMIT: Final[int] = 400


class OllamaMediaEmbedder:
    """Embed text and one image into a single vector via Ollama.

    The image bytes are the raw file content; they are base64-encoded here so
    the caller never has to know the wire format.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout

    @property
    def model(self) -> str:
        """The Ollama model name this embedder calls."""
        return self._model

    @property
    def endpoint(self) -> str:
        """The native Ollama media endpoint this embedder calls."""
        return _media_endpoint(self._base_url)

    def embed(self, text: str, image: bytes) -> list[float]:
        """Return the joint embedding of *text* and *image*.

        Raises:
            MediaEmbeddingUnsupported: the model has no image encoder.
            LLMOutputError: the request failed or the response carried no vector.
        """
        if not image:
            raise LLMOutputError(f"Media embedding for model '{self._model}' received an empty image")
        payload = {
            "model": self._model,  # nocheck: the Ollama API's own payload key
            "input": {
                "text": text,
                "image": base64.b64encode(image).decode(),
            },
        }
        try:
            response = httpx.post(
                self.endpoint,
                json=payload,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMOutputError(f"Media embedding failed for model '{self._model}': {exc}") from exc
        if _MEDIA_UNSUPPORTED_MARKER in response.text:
            raise MediaEmbeddingUnsupported(f"Embedding model '{self._model}' does not support media embeddings")
        if response.is_error:
            body = response.text[:_ERROR_BODY_LIMIT]
            raise LLMOutputError(f"Media embedding failed for model '{self._model}' (HTTP {response.status_code}): {body}")
        return _first_vector(response.json(), model=self._model)


def _media_endpoint(base_url: str) -> str:
    """The native Ollama media endpoint for *base_url*.

    The configured base URL is the OpenAI-compatible one (``.../v1``), but
    ``/api/embed`` lives at the server root — asking ``.../v1/api/embed``
    answers ``404 page not found``.
    """
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return f"{trimmed}/api/embed"


def _first_vector(payload: Any, *, model: str) -> list[float]:
    """Pull the first embedding out of an Ollama ``/api/embed`` response."""
    if not isinstance(payload, dict):
        raise LLMOutputError(f"Media embedding for model '{model}' returned a malformed response")
    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or not vectors:
        raise LLMOutputError(f"Media embedding for model '{model}' returned no vector")
    first = vectors[0]
    if not isinstance(first, list) or not first:
        raise LLMOutputError(f"Media embedding for model '{model}' returned an empty vector")
    return [float(value) for value in first]
