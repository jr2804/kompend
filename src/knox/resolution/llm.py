"""LLM-based entity/relation resolver.

Tier 3 (Resolution) implementation routing LLM calls through the
shared :class:`knox.llm.LLMClient`. Atomic chunks (TABLE, IMAGE,
FORMULA, CODE) bypass the LLM entirely. On client failure the chunk
is returned as a passthrough :class:`ResolvedChunk` — the same
behaviour as before, but with structured logging that records which
model was tried and why it failed.
"""

from __future__ import annotations

import logging

from knox.core.models import ATOMIC_TYPES, Chunk, ResolvedChunk
from knox.core.template_loader import TemplateLoader
from knox.llm.client import LLMClient
from knox.resolution.base import BaseResolver

logger = logging.getLogger(__name__)

_loader = TemplateLoader()


class LLMResolver(BaseResolver):
    """Extract entities and resolve references via the shared LLM client.

    Parameters
    ----------
    client:
        Shared LLM client. When ``None``, a default client is built
        from the global default model. Inject a custom client to
        select a different model or fallback chain for the resolve
        stage.
    system_prompt:
        Optional override for the system prompt.
    role:
        Role key passed to :meth:`LLMClient.complete` (default
        ``"resolve"``). Per-stage configurations key off this value.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        system_prompt: str | None = None,
        role: str = "resolve",
    ) -> None:
        self._client = client  # None is OK — resolve() handles it gracefully
        self._role = role
        self._system_prompt = system_prompt or _loader.get_body("resolution/resolve_system.md.j2")

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        """Resolve entities and references in chunk. Never raises."""
        if ATOMIC_TYPES.intersection(chunk.element_types):
            return ResolvedChunk(chunk=chunk)

        if self._client is None:
            logger.debug("LLMResolver has no LLM client — passing chunk through")
            return ResolvedChunk(chunk=chunk)

        result = self._client.try_complete(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": chunk.text},
            ],
            role=self._role,
            response_format={"type": "json_object"},
        )

        if result is None or result.json_data is None:
            return ResolvedChunk(chunk=chunk)

        data = result.json_data
        return ResolvedChunk(
            chunk=chunk,
            entities=data.get("entities", []),
            relations=data.get("relations", []),
            resolved_references=data.get("resolved_references", []),
            metadata={
                "resolved_by": result.diagnostics.model_used,
                "llm_attempts": result.diagnostics.attempts,
            },
        )
