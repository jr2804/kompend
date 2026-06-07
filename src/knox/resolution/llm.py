"""LLM-based entity/relation resolver.

Tier 3 (Resolution) implementation using litellm for entity extraction
and reference resolution. Falls back gracefully to no-op when LLM is
unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from knox.core.models import ATOMIC_TYPES, Chunk, ElementType, ResolvedChunk
from knox.resolution.base import BaseResolver

logger = logging.getLogger(__name__)


class LLMResolver(BaseResolver):
    """Uses litellm to extract entities and resolve references.

    For text chunks the resolver calls the configured LLM with a JSON
    extraction prompt. Atomic chunks (TABLE, IMAGE, FORMULA, CODE) are
    returned as-is. Any failure falls back to passthrough.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        provider: str = "ollama",
        system_prompt: str | None = None,
    ) -> None:
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

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        """Resolve entities and references in chunk. Never raises."""

        if ATOMIC_TYPES.intersection(chunk.element_types):
            return ResolvedChunk(chunk=chunk)

        try:
            import litellm

            response = litellm.completion(
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
            return ResolvedChunk(
                chunk=chunk,
                entities=data.get("entities", []),
                relations=data.get("relations", []),
                resolved_references=data.get("resolved_references", []),
                metadata={"resolved_by": f"{self.provider}/{self.model}"},
            )

        except Exception:
            logger.exception("LLM resolution failed for chunk %s — falling back to passthrough", chunk.id)
            return ResolvedChunk(chunk=chunk)