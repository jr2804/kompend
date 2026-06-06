"""LLM-based entity/relation resolver.

Tier 3 (Resolution) implementation using litellm for entity extraction
and reference resolution. Falls back gracefully to no-op when LLM is
unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from knox.core.models import Chunk, ElementType, ResolvedChunk
from knox.resolution.base import BaseResolver

logger = logging.getLogger(__name__)

# Element types that should skip LLM resolution (atomic/specialised chunks).
_ATOMIC_TYPES = frozenset({ElementType.TABLE, ElementType.IMAGE, ElementType.FORMULA, ElementType.CODE})


class LLMResolver(BaseResolver):
    """Uses an LLM (via litellm) to extract entities and resolve references.

    For text-based chunks the resolver calls the configured LLM with a JSON
    extraction prompt.  For atomic chunks (TABLE, IMAGE, FORMULA, CODE) the
    chunk is returned as-is because their structure is already explicit.

    The resolver is *resilient* by design — any failure (missing dependency,
    network error, bad response) falls back to returning the chunk unchanged.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        provider: str = "ollama",
        system_prompt: str | None = None,
    ) -> None:
        """Initialise the LLM resolver.

        Args:
            model: Model name to use (e.g. "llama3.2", "gpt-4o-mini").
            provider: Provider prefix for litellm (e.g. "ollama", "openai").
            system_prompt: Optional custom system prompt for entity extraction.
        """
        self.model = model
        self.provider = provider
        self._system_prompt = system_prompt or (
            "Extract named entities and resolve cross-references from the text below. "
            "Return a JSON object with these keys:\n"
            '  - "entities": list of {"name": str, "type": str, "context": str}\n'
            '  - "relations": list of {"source": str, "target": str, "type": str}\n'
            '  - "resolved_references": list of resolved reference strings\n'
            "If no entities are found return empty lists."
        )

        try:
            import litellm  # noqa: F401
            self._litellm = __import__("litellm")
        except ImportError:
            logger.warning("litellm not available — LLMResolver will operate in no-op (passthrough) mode")
            self._litellm = None

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        """Resolve entities and references in *chunk*.

        Returns a :class:`ResolvedChunk` — never raises.
        """
        # ------------------------------------------------------------------
        # Fast path: atomic chunks are returned as-is.
        # ------------------------------------------------------------------
        if _ATOMIC_TYPES.intersection(chunk.element_types):
            return ResolvedChunk(chunk=chunk)

        # ------------------------------------------------------------------
        # Passthrough path: litellm unavailable — no-op.
        # ------------------------------------------------------------------
        if not self._litellm:
            return ResolvedChunk(chunk=chunk)

        # ------------------------------------------------------------------
        # LLM path: try to extract entities via litellm.
        # ------------------------------------------------------------------
        try:
            response = self._litellm.completion(
                model=f"{self.provider}/{self.model}",
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": chunk.text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            raw = response.choices[0].message.content
            if not raw:
                return ResolvedChunk(chunk=chunk)

            data: dict[str, Any] = json.loads(raw)
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            resolved_refs = data.get("resolved_references", [])

            return ResolvedChunk(
                chunk=chunk,
                entities=entities if isinstance(entities, list) else [],
                relations=relations if isinstance(relations, list) else [],
                resolved_references=resolved_refs if isinstance(resolved_refs, list) else [],
                metadata={"resolved_by": f"{self.provider}/{self.model}"},
            )

        except Exception:
            logger.exception("LLM resolution failed for chunk %s — falling back to passthrough", chunk.id)
            return ResolvedChunk(chunk=chunk)