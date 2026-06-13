"""LLM-based entity and relation extractor for KNOX Tier 3.

Extracts typed relations with confidence from text chunks. The resolver is
intended to run early in the resolver chain so downstream compilers can emit
semantic edges such as ``uses``, ``depends_on``, and ``contradicts``.
"""

from __future__ import annotations

import logging
from typing import Any

from knox.core.models import Chunk, ResolvedChunk
from knox.core.template_loader import get_template_loader
from knox.llm.client import LLMClient
from knox.resolution.base import BaseResolver

logger = logging.getLogger(__name__)

_loader = get_template_loader()

# Relation types supported by the typed knowledge graph.
_RELATION_TYPES: frozenset[str] = frozenset({"uses", "depends_on", "contradicts", "caused", "fixed", "supersedes", "related"})


def _normalize_relation(rel: Any) -> dict[str, Any] | None:
    """Validate and normalize a relation dict from the LLM."""
    if not isinstance(rel, dict):
        return None
    source = rel.get("source") or rel.get("subject") or rel.get("from")
    target = rel.get("target") or rel.get("object") or rel.get("to")
    if not source or not target:
        return None
    rel_type = rel.get("type") or rel.get("relation") or "related"
    confidence = rel.get("confidence", 1.0)
    confidence = float(confidence) if isinstance(confidence, (int, float)) else 1.0
    return {
        "source": str(source),
        "target": str(target),
        "type": str(rel_type),
        "confidence": confidence,
    }


class LLMResolver(BaseResolver):
    """Extract entities and typed relations from text chunks.

    Implements :class:`BaseResolver` so it can be the first resolver in a
    :class:`ResolverChain`, converting a :class:`Chunk` into a
    :class:`ResolvedChunk`. It also implements :meth:`enrich` so later
    resolvers (e.g. :class:`VLMEnricher`) can add captions without losing
    the relations produced here.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        confidence_threshold: float = 0.0,
        relation_types: frozenset[str] | None = None,
    ) -> None:
        self._client = client
        self._confidence_threshold = confidence_threshold
        self._relation_types = relation_types or _RELATION_TYPES

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        """BaseResolver interface: wrap chunk then enrich."""
        resolved = ResolvedChunk(chunk=chunk)
        return self.enrich(resolved)

    def enrich(self, resolved: ResolvedChunk) -> ResolvedChunk:
        """Extract entities and relations from the chunk text."""
        text = resolved.chunk.text or ""
        if not text:
            return resolved

        prompt = _loader.render(
            "resolution/resolve_relations.md.j2",
            text=text,
            relation_types=sorted(self._relation_types),
            confidence_threshold=self._confidence_threshold,
        )
        result = self._client.try_complete(
            [{"role": "user", "content": prompt}],
            role="resolve.relations",
            response_format={"type": "json_object"},
        )
        if result is None or result.json_data is None:
            return resolved

        data = result.json_data
        entities = data.get("entities")
        if isinstance(entities, list):
            resolved.entities = [
                {
                    "name": str(ent.get("name", "")) if isinstance(ent, dict) else str(ent),
                    "type": str(ent.get("type", "")) if isinstance(ent, dict) else "",
                    "context": str(ent.get("context", "")) if isinstance(ent, dict) else "",
                }
                for ent in entities
            ]

        relations = data.get("relations")
        if isinstance(relations, list):
            normalized = (_normalize_relation(rel) for rel in relations)
            resolved.relations = [rel for rel in normalized if rel and rel["confidence"] >= self._confidence_threshold and rel["type"] in self._relation_types]

        return resolved
