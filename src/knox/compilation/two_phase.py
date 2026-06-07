"""N3 two-phase compiler — extract first, then generate with cross-source merging.

Phase 1 (extract): for each source, run the LLM once in extraction mode
and collect a structured payload of {summary, entities, concepts, tags}.

Phase 2 (generate): merge same-named entities/concepts across sources,
then ask the LLM to generate one page per unique entity/concept. Each
page carries ``^[source.pdf:42-58]`` provenance citations per N4.

When the LLM call fails, the compiler falls back to the underlying
``base`` compiler (typically :class:`GraphCompiler`) so the pipeline
remains useful offline.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import litellm

from knox.compilation.base import BaseCompiler
from knox.core.models import CompiledWiki, ResolvedChunk, WikiNode

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = (
    "You extract structured information from source documents. "
    "Return only valid JSON with the schema: "
    '{"summary": str, "entities": [{"name": str, "type": str}], '
    '"concepts": [{"name": str, "definition": str}], "tags": [str]}.'
)

_EXTRACTION_TEMPLATE = (
    "Read the following source and return a JSON object with: "
    '"summary" (1-3 sentence summary), "entities" (list of objects with name and type), '
    '"concepts" (list of objects with name and definition), "tags" (list of short tags). '
    "Source: {text}"
)

_GENERATION_SYSTEM = (
    "You generate knowledge-base pages. Return only valid JSON: "
    '{"pages": [{"title": str, "content": str, "provenance": [str]}]}'
)

_GENERATION_TEMPLATE = (
    "Generate one knowledge-base page for each concept. "
    "Each page's content must include claim-level provenance in the form "
    '"^[source.pdf:42-58]" appended to the claim it supports. '
    "Concepts (across all sources): {concepts_json}\n"
    "Source content (fair-share truncated): {content}"
)


@dataclass
class Extraction:
    """Phase-1 extraction result for a single source."""

    source: str
    summary: str = ""
    entities: list[dict[str, str]] = field(default_factory=list)
    concepts: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary,
            "entities": list(self.entities),
            "concepts": list(self.concepts),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Extraction:
        return cls(
            source=data.get("source", ""),
            summary=data.get("summary", ""),
            entities=list(data.get("entities", [])),
            concepts=list(data.get("concepts", [])),
            tags=list(data.get("tags", [])),
        )


@dataclass
class ExtractionSet:
    """All extractions across the corpus."""

    extractions: list[Extraction] = field(default_factory=list)

    def add(self, extraction: Extraction) -> None:
        self.extractions.append(extraction)

    def concept_pages(self) -> list[dict[str, Any]]:
        """Return merged concept dicts with provenance per source.

        Output schema: ``{name, definition, sources: [str], definitions: [str]}``
        """
        merged: dict[str, dict[str, Any]] = {}
        for ext in self.extractions:
            for concept in ext.concepts:
                name = concept.get("name", "").strip()
                if not name:
                    continue
                bucket = merged.setdefault(name, {"name": name, "sources": [], "definitions": []})
                bucket["sources"].append(ext.source)
                if concept.get("definition"):
                    bucket["definitions"].append(concept["definition"])
        return list(merged.values())

    def entity_index(self) -> dict[str, list[str]]:
        """Return ``entity_name -> [source, ...]``."""
        out: dict[str, list[str]] = defaultdict(list)
        for ext in self.extractions:
            for ent in ext.entities:
                name = ent.get("name", "").strip()
                if name:
                    out[name].append(ext.source)
        return dict(out)


def _try_litellm_json(prompt: str, system: str | None = None) -> Any | None:
    """Call litellm completion and return parsed JSON, or None on failure."""
    try:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = litellm.completion(
            model="ollama/llama3.2",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        return json.loads(content)
    except Exception:
        logger.exception("litellm call failed in two-phase compiler")
        return None


def _fallback_extraction(source: str, text: str) -> Extraction:
    """Heuristic extraction when the LLM is unavailable."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    summary = sentences[0] if sentences else ""
    return Extraction(
        source=source,
        summary=summary,
        entities=[],
        concepts=[{"name": "auto_concept", "definition": summary}] if summary else [],
        tags=[],
    )


def fair_share_truncate(
    sources: list[tuple[str, str]],
    total_budget: int,
) -> list[tuple[str, str, int]]:
    """Distribute ``total_budget`` chars across ``sources`` fairly.

    Returns ``(source_label, content, dropped_chars)`` per source. The
    dropped-chars value is logged so the user can adjust the budget.
    """
    if total_budget <= 0 or not sources:
        return [(label, "", len(content)) for label, content in sources]

    per_source = total_budget // len(sources)
    out: list[tuple[str, str, int]] = []
    for label, content in sources:
        if len(content) <= per_source:
            out.append((label, content, 0))
        else:
            out.append((label, content[:per_source], len(content) - per_source))
    return out


class TwoPhaseCompiler(BaseCompiler):
    """N3 two-phase compiler with cross-source merging and N10 fair-share budget.

    Parameters
    ----------
    base:
        Compiler used as a fallback when the LLM is unavailable.
    prompt_budget:
        Maximum total chars of source content used to generate a single
        concept page (N10 fair-share truncation).
    """

    def __init__(
        self,
        base: BaseCompiler | None = None,
        prompt_budget: int = 200_000,
    ) -> None:
        self._base = base
        self.prompt_budget = prompt_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Run the two-phase compile."""
        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={"chunk_count": 0, "two_phase": True},
            )

        extractions = self.extract(chunks)
        return self.generate(extractions, chunks)

    def extract(self, chunks: list[ResolvedChunk]) -> ExtractionSet:
        """Phase 1: run the LLM in extraction mode for each source."""
        _ = self  # noqa: PLR6301 — instance method for future per-compiler config
        extractions = ExtractionSet()
        # Group chunks by source file so each source is extracted once.
        by_source: dict[str, list[ResolvedChunk]] = defaultdict(list)
        for rc in chunks:
            label = str(rc.chunk.source_file or "unknown")
            by_source[label].append(rc)

        for source, source_chunks in by_source.items():
            text = "\n\n".join(rc.chunk.text for rc in source_chunks if rc.chunk.text)
            prompt = _EXTRACTION_TEMPLATE.format(text=text[:6000])
            data = _try_litellm_json(prompt, system=_EXTRACTION_SYSTEM)
            if isinstance(data, dict):
                extractions.add(
                    Extraction(
                        source=source,
                        summary=str(data.get("summary", "")),
                        entities=[dict(e) for e in data.get("entities", []) if isinstance(e, dict)],
                        concepts=[dict(c) for c in data.get("concepts", []) if isinstance(c, dict)],
                        tags=[str(t) for t in data.get("tags", []) if t],
                    )
                )
            else:
                extractions.add(_fallback_extraction(source, text))
        return extractions

    def generate(self, extractions: ExtractionSet, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Phase 2: merge extractions and generate per-concept pages."""
        concepts = extractions.concept_pages()
        if not concepts:
            # Nothing to merge — fall back to the base compiler.
            return self._fallback_compile(chunks)

        # Fair-share truncate source content for the generation prompt.
        sources: list[tuple[str, str]] = []
        by_source: dict[str, str] = defaultdict(str)
        for rc in chunks:
            label = str(rc.chunk.source_file or "unknown")
            if rc.chunk.text:
                by_source[label] += rc.chunk.text + "\n\n"
        for label, text in by_source.items():
            sources.append((label, text))

        truncated = fair_share_truncate(sources, self.prompt_budget)
        dropped_total = sum(d for _, _, d in truncated)
        if dropped_total > 0:
            logger.warning(
                "TwoPhase: prompt budget %d truncated %d chars across %d sources",
                self.prompt_budget,
                dropped_total,
                len(truncated),
            )

        content_block = "\n\n".join(
            f"### {label}\n{content}" for label, content, _ in truncated
        )
        prompt = _GENERATION_TEMPLATE.format(
            concepts_json=json.dumps(concepts),
            content=content_block,
        )
        data = _try_litellm_json(prompt, system=_GENERATION_SYSTEM)

        nodes: list[WikiNode] = []
        edges: list[dict[str, Any]] = []

        if isinstance(data, dict):
            for idx, page in enumerate(data.get("pages", [])):
                if not isinstance(page, dict):
                    continue
                nodes.append(
                    WikiNode(
                        id=f"page_{idx}",
                        title=str(page.get("title", f"Concept {idx}")),
                        content=str(page.get("content", "")),
                        node_type="concept",
                        source_chunks=[],
                        metadata={
                            "provenance": list(page.get("provenance", []) or []),
                            "two_phase": True,
                        },
                    )
                )
            # Cross-source edges: pair-wise connect pages that share an entity.
            entity_index = extractions.entity_index()
            entity_to_page: dict[str, set[int]] = defaultdict(set)
            for ent_name in entity_index:
                for idx, page in enumerate(data.get("pages", [])):
                    if not isinstance(page, dict):
                        continue
                    if ent_name.lower() in str(page.get("content", "")).lower():
                        entity_to_page[ent_name].add(idx)
            for ent_name, page_idxs in entity_to_page.items():
                sorted_idxs = sorted(page_idxs)
                for i in range(len(sorted_idxs)):
                    for j in range(i + 1, len(sorted_idxs)):
                        edges.append({
                            "source": nodes[sorted_idxs[i]].id,
                            "target": nodes[sorted_idxs[j]].id,
                            "type": "entity_reference",
                            "entity": ent_name,
                        })
        else:
            # LLM failed — fall back to base compiler for substance.
            return self._fallback_compile(chunks)

        toc = [{"title": n.title, "id": n.id, "level": 1} for n in nodes]
        return CompiledWiki(
            nodes=nodes,
            edges=edges,
            toc=toc,
            metadata={
                "chunk_count": len(chunks),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "two_phase": True,
                "concepts_merged": len(concepts),
                "prompt_budget": self.prompt_budget,
                "truncated_chars": dropped_total,
            },
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fallback_compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        if self._base is not None:
            wiki = self._base.compile(chunks)
            return CompiledWiki(
                nodes=list(wiki.nodes),
                edges=list(wiki.edges),
                toc=list(wiki.toc),
                quality_score=wiki.quality_score,
                iteration=wiki.iteration,
                metadata={**wiki.metadata, "two_phase": True, "fallback": True},
            )
        # Last-resort empty wiki.
        return CompiledWiki(
            nodes=[],
            edges=[],
            toc=[],
            metadata={"chunk_count": len(chunks), "two_phase": True, "fallback": True},
        )
