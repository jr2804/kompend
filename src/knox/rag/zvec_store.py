"""zvec-based vector store for semantic search over compiled wikis.

zvec is an embedded vector database (Alibaba, Apache 2.0) — in-process,
no server required. This module wraps it as a sidecar for KNOX's compiled
knowledge base.

NOTE: zvec uses CPU-specific instructions (AVX) and may crash with SIGILL
on older x86 CPUs. A subprocess-based availability check at import time
prevents the crash from taking down the interpreter.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki

logger = logging.getLogger(__name__)


def _zvec_available() -> bool:
    """Check if zvec can be imported without crashing (SIGILL-safe)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import zvec; print('ok')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "ok"
    except Exception:
        return False


if _zvec_available():
    import zvec as _zvec  # noqa: PLC0415
    ZVEC_AVAILABLE = True
else:
    ZVEC_AVAILABLE = False
    _zvec = None  # type: ignore[assignment]


class ZvecWikiStore:
    """Sidecar vector store for semantic search over compiled wikis.

    The store is dormant until build_index() is called.
    """

    def __init__(
        self,
        db_path: str | Path = "./.knox/vectors",
        embedding_dim: int = 384,
    ) -> None:
        self.db_path = Path(db_path)
        self._embedding_dim = embedding_dim
        self._collection: Any = None

    @property
    def available(self) -> bool:
        return ZVEC_AVAILABLE and self._collection is not None

    def build_index(
        self,
        wiki: CompiledWiki,
        embedding_model: str | None = None,
    ) -> int:
        """Index compiled wiki nodes into the zvec store."""
        if not ZVEC_AVAILABLE or not _zvec:
            logger.warning("zvec not available — skipping index build")
            return 0
        if not wiki.nodes:
            return 0

        if self._collection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            schema = _zvec.CollectionSchema(
                name="knox_wiki",
                vectors=_zvec.VectorSchema(
                    "embedding", _zvec.DataType.VECTOR_FP32, self._embedding_dim,
                ),
            )
            self._collection = _zvec.create_and_open(
                path=str(self.db_path), schema=schema,
            )

        docs = [
            _zvec.Doc(
                id=node.id,
                vectors={"embedding": self._text_to_vec(f"{node.title}\n{node.content}")},
                metadata={
                    "title": node.title,
                    "node_type": node.node_type,
                    "content_preview": node.content[:200],
                },
            )
            for node in wiki.nodes
        ]
        self._collection.insert(docs)
        logger.info("Indexed %d nodes in zvec store at %s", len(docs), self.db_path)
        return len(docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed wiki content by semantic similarity."""
        if not self.available or not _zvec:
            return []
        results = self._collection.query(
            _zvec.VectorQuery("embedding", vector=self._text_to_vec(query)),
            topk=top_k,
        )
        return [
            {
                "id": r.get("id", ""),
                "score": r.get("score", 0.0),
                **(r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}),
            }
            for r in results
        ]

    def close(self) -> None:
        if self._collection is not None:
            self._collection.close()
            self._collection = None

    @staticmethod
    def _text_to_vec(text: str, dim: int = 384) -> list[float]:
        """Deterministic text→vector via SHA-256 (placeholder for testing)."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(digest[i % 32] - 128) / 128.0 for i in range(dim)]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm > 0 else vec