"""Model-based rerankers: re-score retrieved candidates with a cross-encoder.

A cross-encoder reads the query and the passage *together*, so it judges a
pair instead of two independent vectors. That is strictly better than the
bi-encoder re-scoring the vector store already did, at the cost of a model
in the loop — hence an opt-in stage.

The heavy dependency (flashrank pulls onnxruntime) is imported lazily, so a
vault that never enables the stage never pays for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import kompend.rag
from kompend.logging import get_logger

logger = get_logger(__name__)

#: The model flashrank uses when the user names none. A cross-encoder
#: distilled from ms-marco — ~3 MB, no torch.
DEFAULT_FLASHRANK_MODEL = "ms-marco-TinyBERT-L-2-v2"


class RerankerUnavailable(RuntimeError):
    """The reranker cannot run — the model or its dependency is missing."""


class Reranker(Protocol):
    """Re-scores passages against a query."""

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return one score per passage, aligned with the input order.

        Higher is better. The scores are comparable across passages of the
        same call, not across calls.
        """
        ...


class FlashRankReranker:
    """Cross-encoder reranker backed by flashrank.

    The model is downloaded on first use and cached under the vault's cache
    directory, so the download lands inside ``.kompend/`` rather than in the
    user's home directory.
    """

    def __init__(self, model_name: str = "", *, cache_dir: Path | None = None) -> None:
        """Prepare the reranker.

        Raises:
            RerankerUnavailable: when flashrank is not installed or the model
                cannot be loaded.
        """
        self._model_name = model_name or DEFAULT_FLASHRANK_MODEL
        self._cache_dir = cache_dir
        try:
            self._ranker = kompend.rag.load_ranker(self._model_name, cache_dir=cache_dir)
        except ImportError as exc:
            raise RerankerUnavailable(
                "The flashrank reranker needs the 'rerank' extra (uv add 'kompend[rerank]')"
            ) from exc
        except Exception as exc:
            raise RerankerUnavailable(f"Could not load the rerank model {self._model_name!r}: {exc}") from exc

    @property
    def model_name(self) -> str:
        """The model behind the scores."""
        return self._model_name

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Score every passage against *query*, in the input order."""
        if not passages:
            return []
        request = kompend.rag.build_request(query, passages)
        ranked = self._ranker.rerank(request)
        scores = [0.0] * len(passages)
        for item in ranked:
            index = int(item["id"])
            if 0 <= index < len(scores):
                scores[index] = float(item["score"])
        return scores
