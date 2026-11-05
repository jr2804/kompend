"""The flashrank cross-encoder — the optional cross-encoder backend.

flashrank pulls onnxruntime, so this module is imported only when the rerank
stage asks for it: a vault that never enables the stage never loads it.
"""

from __future__ import annotations

from pathlib import Path

from flashrank import Ranker, RerankRequest


def load_ranker(model_name: str, *, cache_dir: Path | None = None) -> Ranker:
    """Build a flashrank ranker for *model_name*.

    The model is downloaded on first use; *cache_dir* is where it lands.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return Ranker(model_name=model_name, cache_dir=str(cache_dir))
    return Ranker(model_name=model_name)


def build_request(query: str, passages: list[str]) -> RerankRequest:
    """Build the rerank request for *query* over *passages*."""
    return RerankRequest(
        query=query,
        passages=[{"id": str(index), "text": text} for index, text in enumerate(passages)],
    )
