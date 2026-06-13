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

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from slugify import slugify as _slugify_impl

from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.core.cache_store import CacheStore
from knox.core.models import CompiledWiki, ElementType, ResolvedChunk, WikiNode
from knox.core.pipeline_defs import PipelineConfig
from knox.core.template_loader import get_template_loader
from knox.llm.client import _discover_context_window

logger = logging.getLogger(__name__)

_loader = get_template_loader()

_CHARS_PER_TOKEN = 4  # Approximate for English text.


def _tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * _CHARS_PER_TOKEN


def _chars_to_tokens(chars: int) -> int:
    """Convert character count to approximate token count."""
    return max(1, chars // _CHARS_PER_TOKEN)


def _slugify(text: str) -> str:
    """Convert a title to a lowercase hyphenated slug."""
    return _slugify_impl(text) or "concept"


def _dedup_id(node_id: str, seen: set[str]) -> str:
    """Append a numeric suffix if node_id already exists in *seen*, then add."""
    candidate = node_id
    counter = 2
    while candidate in seen:
        candidate = f"{node_id}-{counter}"
        counter += 1
    seen.add(candidate)
    return candidate


def _extract_images_from_chunks(chunks: list[ResolvedChunk]) -> list[dict[str, Any]]:
    """Extract image metadata from IMAGE chunks.

    Returns a list of dicts with keys: path, caption, element_type.
    Images are collected from:
    - chunk.metadata["image_path"] (file mode)
    - chunk.metadata["image_base64"] (base64 mode)
    - VLM captions from resolved.entities with type="image_description"
    """
    images: list[dict[str, Any]] = []

    for rc in chunks:
        if ElementType.IMAGE not in rc.chunk.element_types:
            continue

        # Get image path or base64 data
        img_path = rc.chunk.metadata.get("image_path") or rc.chunk.metadata.get("image_base64")
        if not img_path:
            # Try resolved metadata
            img_path = rc.metadata.get("image_path") or rc.metadata.get("image_base64")

        if not img_path:
            continue

        # Get VLM caption if available
        caption = ""
        for entity in rc.entities:
            if entity.get("type") == "image_description" and entity.get("context"):
                caption = entity["context"]
                break

        images.append(
            {
                "path": img_path,
                "caption": caption,
                "element_type": "image",
            }
        )

    return images


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown text into sections keyed by heading.

    Content before the first heading is stored under the empty-string key.
    """
    sections: dict[str, str] = {}
    current_heading = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("## "):
            if current_lines:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = stripped[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _merge_sections(existing: dict[str, str], incoming: dict[str, str]) -> str:
    """Merge two section dicts, keeping the richer version per heading."""
    merged: dict[str, str] = {}
    for heading in dict.fromkeys(list(existing) + list(incoming)):
        ex = existing.get(heading, "")
        inc = incoming.get(heading, "")
        merged[heading] = ex if len(ex) >= len(inc) else inc

    result = ""
    for heading, body in merged.items():
        if heading:
            result += f"## {heading}\n{body}\n\n"
        elif body:
            result += f"{body}\n\n"
    return result.strip()


def _compute_quality_score(nodes: list[WikiNode], edges: list[dict[str, Any]]) -> float:
    """Compute a 0–100 quality score for a compiled wiki.

    Weighted average of three sub-scores:
    - **Completeness** (40 %): fraction of nodes with non-trivial content (>50 chars)
    - **Connectivity** (30 %): edge density relative to a target of ~3 edges/node
    - **Richness** (30 %): average content length normalised against a 500-char target
    """
    n = len(nodes)
    if n == 0:
        return 0.0

    non_empty = sum(1 for node in nodes if len((node.content or "").strip()) > 50)
    completeness = non_empty / n

    target_edges = n * 3
    connectivity = min(len(edges) / target_edges, 1.0) if target_edges > 0 else 0.0

    avg_len = sum(len(node.content or "") for node in nodes) / n
    richness = min(avg_len / 500.0, 1.0)

    return round((completeness * 0.4 + connectivity * 0.3 + richness * 0.3) * 100, 1)


def _merge_node(
    existing: WikiNode,
    incoming: WikiNode,
) -> WikiNode:
    """Merge *incoming* node into *existing* using section-aware dedup.

    Sections with the same heading are de-duplicated, keeping the longer
    (richer) version.  Sections unique to either source are appended.
    Provenance lists are unioned.
    """
    existing_text = existing.content or ""
    incoming_text = incoming.content or ""

    if not existing_text.strip():
        merged_content = incoming_text
    elif not incoming_text.strip():
        merged_content = existing_text
    else:
        ex_sections = _split_sections(existing_text)
        inc_sections = _split_sections(incoming_text)
        merged_content = _merge_sections(ex_sections, inc_sections)

    merged_title = existing.title or incoming.title

    # Merge provenance
    existing_prov = set(existing.metadata.get("provenance", []) or [])
    incoming_prov = set(incoming.metadata.get("provenance", []) or [])
    merged_prov = sorted(existing_prov | incoming_prov)

    # Merge related
    existing_rel = set(existing.metadata.get("related", []) or [])
    incoming_rel = set(incoming.metadata.get("related", []) or [])
    merged_rel = sorted(existing_rel | incoming_rel)

    return WikiNode(
        id=existing.id,
        title=merged_title,
        content=merged_content,
        node_type=existing.node_type,
        source_chunks=list(dict.fromkeys(existing.source_chunks + incoming.source_chunks)),
        metadata={
            **existing.metadata,
            "provenance": merged_prov,
            "related": merged_rel,
        },
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


def _try_litellm_json(
    prompt: str,
    model: str,
    system: str | None = None,
    timeout: float | None = None,
) -> Any | None:
    """Call litellm completion and return parsed JSON, or None on failure."""
    start = time.monotonic()
    try:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = litellm.completion(**kwargs)
        elapsed = time.monotonic() - start
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            except (TypeError, ValueError):
                pt = 0
            try:
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
            except (TypeError, ValueError):
                ct = 0
            tt = pt + ct
            tps = f"{tt / elapsed:.1f}" if elapsed > 0 else "?"
            logger.info(
                "LLM ok model=%s in=%.1fs in=%d out=%d total=%d tok/s=%s",
                model,
                elapsed,
                pt,
                ct,
                tt,
                tps,
            )
        else:
            logger.info("LLM ok model=%s in=%.1fs (no token info)", model, elapsed)
        content = response.choices[0].message.content
        if not content:
            return None
        return json.loads(content)
    except Exception:
        elapsed = time.monotonic() - start
        logger.error("LLM fail model=%s in=%.1fs", model, elapsed, exc_info=True)
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


def _merge_extractions(source: str, parts: list[Extraction]) -> Extraction:
    """Merge multiple batch extractions for the same source into one."""
    if not parts:
        return Extraction(source=source)
    if len(parts) == 1:
        return parts[0]

    summary = " ".join(p.summary for p in parts if p.summary)
    entities: list[dict[str, str]] = []
    concepts: list[dict[str, str]] = []
    tags: list[str] = []
    seen_entities: set[str] = set()
    seen_concepts: set[str] = set()
    seen_tags: set[str] = set()
    for p in parts:
        for e in p.entities:
            key = e.get("name", "").strip().lower()
            if key and key not in seen_entities:
                entities.append(e)
                seen_entities.add(key)
        for c in p.concepts:
            key = c.get("name", "").strip().lower()
            if key and key not in seen_concepts:
                concepts.append(c)
                seen_concepts.add(key)
        for t in p.tags:
            if t not in seen_tags:
                tags.append(t)
                seen_tags.add(t)
    return Extraction(source=source, summary=summary, entities=entities, concepts=concepts, tags=tags)


def _segment_sources(
    sources: list[tuple[str, str]],
    segment_size: int,
) -> list[list[tuple[str, str]]]:
    """Split source content into segments of at most *segment_size* chars.

    Returns a list of segments, where each segment is a list of
    ``(label, content)`` pairs.  No data is dropped — sources that
    exceed *segment_size* are split across multiple segments.
    """
    if not sources:
        return []

    segments: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_len = 0

    for label, content in sources:
        remaining = content
        while remaining:
            space = segment_size - current_len
            if space <= 0:
                segments.append(current)
                current = []
                current_len = 0
                space = segment_size
            if len(remaining) <= space:
                current.append((label, remaining))
                current_len += len(remaining)
                remaining = ""
            else:
                # Split at the last newline within budget, or hard-cut.
                cut = remaining.rfind("\n", 0, space)
                if cut <= 0:
                    cut = space
                current.append((label, remaining[:cut]))
                current_len += cut
                remaining = remaining[cut:]

    if current:
        segments.append(current)
    return segments


class TwoPhaseCompiler(BaseCompiler):
    """N3 two-phase compiler with cross-source merging and adaptive budgets.

    Parameters
    ----------
    config:
        Pipeline configuration including budget and context window settings.
    base:
        Compiler used as a fallback when the LLM is unavailable.
    """

    def __init__(
        self,
        config: PipelineConfig,
        base: BaseCompiler | None = None,
    ) -> None:
        self._base = base or GraphCompiler(model=config.llm.default_model)
        self._budget = config.budget
        self._model = config.llm.default_model
        self._config = config
        self._timeout = config.llm.timeout
        self._cache_store = CacheStore(config.knox_dir() / "cache")

        # Discover context window for adaptive sizing.
        cw = _discover_context_window(
            self._model,
            fallback_max_input=config.llm.fallback_max_input,
            fallback_max_output=config.llm.fallback_max_output,
            num_ctx_min=config.llm.num_ctx_min,
            num_ctx_max=config.llm.num_ctx_max,
        )
        self._context_window = cw

        # Estimate system prompt token overhead (rough: 4 chars/token).
        extraction_system = _loader.get_body("compilation/extraction_system.md.j2")
        generation_system = _loader.get_body("compilation/generation_system.md.j2")
        self._extract_system_tokens = max(1, len(extraction_system) // 4)
        self._gen_system_tokens = max(1, len(generation_system) // 4)

        logger.info(
            "TwoPhase: context_window=%d in, %d out (source=%s)",
            cw.max_input,
            cw.max_output,
            cw.source,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Run the two-phase compile with generation cache (Tier 3 Phase 2)."""
        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={"chunk_count": 0, "two_phase": True},
            )

        extractions = self.extract(chunks)

        # Generation cache: keyed by all extractions + model + gen config.
        gen_deps = self._gen_cache_deps(extractions)
        if not self._config.no_cache:
            cached = self._cache_store.get("generate", gen_deps)
            if cached is not None:
                logger.info(
                    "TwoPhase: generation cache hit (%d nodes, %d edges)",
                    len(cached.nodes),
                    len(cached.edges),
                )
                return cached

        wiki = self.generate(extractions, chunks)

        if not self._config.no_cache:
            self._cache_store.set("generate", gen_deps, wiki)
        return wiki

    def extract(self, chunks: list[ResolvedChunk]) -> ExtractionSet:
        """Phase 1: run the LLM in extraction mode for each source.

        Source text is processed in batches sized to fill the context window
        (minus system prompt and output budget) so that no content is silently
        dropped.  Results from each batch are merged into a single
        :class:`Extraction` per source.
        """
        _ = self  # noqa: PLR6301 — instance method for future per-compiler config
        extractions = ExtractionSet()
        # Group chunks by source file so each source is extracted once.
        by_source: dict[str, list[ResolvedChunk]] = defaultdict(list)
        for rc in chunks:
            label = str(rc.chunk.source_file or "unknown")
            by_source[label].append(rc)

        extraction_system = _loader.get_body("compilation/extraction_system.md.j2")
        num_sources = len(by_source)
        logger.info("TwoPhase: extracting %d source(s)", num_sources)

        # Adaptive extract batch size: fill context window, clamped to safety rails.
        cw = self._context_window
        budget = self._budget
        extract_char_budget = _tokens_to_chars(
            max(1, cw.max_input - self._extract_system_tokens - budget.extract_output_max),
        )
        extract_char_budget = max(_tokens_to_chars(budget.chunk_size_min), min(extract_char_budget, _tokens_to_chars(budget.chunk_size_max)))

        for source_idx, (source, source_chunks) in enumerate(by_source.items(), start=1):
            logger.info("TwoPhase: extract [%d/%d] %s", source_idx, num_sources, source)
            texts = [rc.chunk.text for rc in source_chunks if rc.chunk.text]
            if not texts:
                extractions.add(Extraction(source=source))
                continue

            # Batch texts into budget-sized groups so nothing is dropped.
            batch_parts: list[str] = []
            batch_len = 0
            batch_extractions: list[Extraction] = []

            for text in texts:
                if batch_len + len(text) > extract_char_budget and batch_parts:
                    combined = "\n\n".join(batch_parts)
                    batch_extractions.append(self._extract_one(source, combined, extraction_system))
                    batch_parts = []
                    batch_len = 0
                batch_parts.append(text)
                batch_len += len(text)

            if batch_parts:
                combined = "\n\n".join(batch_parts)
                batch_extractions.append(self._extract_one(source, combined, extraction_system))

            # Merge all batch extractions for this source.
            extractions.add(_merge_extractions(source, batch_extractions))
        return extractions

    def generate(self, extractions: ExtractionSet, chunks: list[ResolvedChunk]) -> CompiledWiki:  # noqa: PLR0915
        """Phase 2: merge extractions and generate per-concept pages.

        Batches concepts into groups of ``concepts_per_batch`` to avoid
        overwhelming the LLM with a single massive prompt. Source segments
        are sized adaptively based on the discovered context window.
        """
        concepts = extractions.concept_pages()
        if not concepts:
            # Nothing to merge — fall back to the base compiler.
            return self._fallback_compile(chunks)

        # Collect images from IMAGE chunks for embedding in output
        all_images = _extract_images_from_chunks(chunks)

        # Per-source content, collected in full.
        by_source: dict[str, str] = defaultdict(str)
        for rc in chunks:
            label = Path(str(rc.chunk.source_file or "unknown")).stem
            if rc.chunk.text:
                by_source[label] += rc.chunk.text + "\n\n"
        sources: list[tuple[str, str]] = list(by_source.items())

        # Adaptive segment size: fill context window, clamped to safety rails.
        cw = self._context_window
        budget = self._budget
        segment_char_size = _tokens_to_chars(
            max(1, cw.max_input - self._gen_system_tokens - budget.gen_output_max),
        )
        segment_char_size = max(_tokens_to_chars(budget.chunk_size_min), min(segment_char_size, _tokens_to_chars(budget.chunk_size_max)))

        source_segments = _segment_sources(sources, segment_char_size)

        generation_system = _loader.get_body("compilation/generation_system.md.j2")
        nodes: list[WikiNode] = []
        edges: list[dict[str, Any]] = []
        slug_to_idx: dict[str, int] = {}  # slug → index in nodes (for merging)

        def _upsert_node(new_node: WikiNode) -> None:
            """Insert a new node, or merge with existing if slug matches."""
            slug = new_node.id
            if slug in slug_to_idx:
                idx = slug_to_idx[slug]
                nodes[idx] = _merge_node(nodes[idx], new_node)
            else:
                slug_to_idx[slug] = len(nodes)
                nodes.append(new_node)

        num_concepts = len(concepts)
        num_segments = len(source_segments)
        batch_size = budget.concepts_per_batch
        logger.info("TwoPhase: generating %d concept(s) across %d source segment(s) (batch_size=%d)", num_concepts, num_segments, batch_size)
        for batch_start in range(0, num_concepts, batch_size):
            batch = concepts[batch_start : batch_start + batch_size]

            # Iterate through ALL source segments for this concept batch,
            # generating pages from each segment and merging.
            for seg_idx, segment in enumerate(source_segments):
                logger.info("TwoPhase: gen batch %d-%d/%d segment %d/%d", batch_start, len(batch), num_concepts, seg_idx + 1, num_segments)
                content_block = "\n\n".join(f"### {label}\n{content}" for label, content in segment)

                prompt = _loader.render(
                    "compilation/generation_prompt.md.j2",
                    concepts_json=json.dumps(batch),
                    content=content_block,
                )
                data = _try_litellm_json(prompt, system=generation_system, model=self._model, timeout=self._timeout)

                if not isinstance(data, dict):
                    logger.warning(
                        "TwoPhase: LLM returned non-dict for batch %d-%d, segment %d — using fallback",
                        batch_start,
                        batch_start + len(batch),
                        seg_idx,
                    )
                    if seg_idx == 0:
                        # Only create fallback nodes from the first segment
                        # to avoid duplicates.
                        for idx, concept in enumerate(batch):
                            node_id = _slugify(concept.get("name", f"concept-{batch_start + idx}"))
                            _upsert_node(
                                WikiNode(
                                    id=node_id,
                                    title=concept.get("name", "Unknown"),
                                    content=concept.get("definition", ""),
                                    node_type="concept",
                                    source_chunks=[],
                                    metadata={
                                        "provenance": concept.get("sources", []),
                                        "two_phase": True,
                                        "images": all_images,
                                    },
                                )
                            )
                    continue

                for idx, page in enumerate(data.get("pages", [])):
                    if not isinstance(page, dict):
                        continue
                    # Always derive slug from title for consistency — ignore LLM slug
                    node_id = _slugify(str(page.get("title", f"concept-{batch_start + idx}")))
                    _upsert_node(
                        WikiNode(
                            id=node_id,
                            title=str(page.get("title", f"Concept {batch_start + idx}")),
                            content=str(page.get("content", "")),
                            node_type="concept",
                            source_chunks=[],
                            metadata={
                                "provenance": list(page.get("provenance", []) or []),
                                "related": list(page.get("related", []) or []),
                                "two_phase": True,
                                "images": all_images,
                                "frontmatter": {
                                    "title": str(page.get("title", f"Concept {batch_start + idx}")),
                                },
                            },
                        )
                    )

        if not nodes:
            return self._fallback_compile(chunks)

        # Cross-source edges: pair-wise connect pages that share an entity.
        # Only match entities against title + summary (first 200 chars), not
        # the entire page content — full-content matching creates near-complete
        # graphs in narrow domains.
        entity_index = extractions.entity_index()
        entity_to_node: dict[str, set[str]] = defaultdict(set)
        for ent_name in entity_index:
            ent_lower = ent_name.lower()
            for node in nodes:
                match_text = (f"{node.title} {(node.content or '')[:200]}").lower()
                if ent_lower in match_text:
                    entity_to_node[ent_name].add(node.id)

        # Deduplicate edges — same pair may appear from both entity and related links
        seen_edges: set[tuple[str, str]] = set()
        for ent_name, node_ids in entity_to_node.items():
            sorted_ids = sorted(node_ids)
            for i in range(len(sorted_ids)):
                for j in range(i + 1, len(sorted_ids)):
                    pair = (sorted_ids[i], sorted_ids[j])
                    if pair in seen_edges:
                        continue
                    seen_edges.add(pair)
                    edges.append(
                        {
                            "source": sorted_ids[i],
                            "target": sorted_ids[j],
                            "type": "entity_reference",
                            "entity": ent_name,
                        }
                    )

        # Also wire up explicit "related" links from the LLM response.
        title_to_id = {n.title: n.id for n in nodes}
        for node in nodes:
            for related_title in node.metadata.get("related", []):
                target_id = title_to_id.get(related_title)
                if target_id and target_id != node.id:
                    pair = tuple(sorted([node.id, target_id]))
                    if pair in seen_edges:
                        continue
                    seen_edges.add(pair)
                    edges.append({"source": node.id, "target": target_id, "type": "related", "entity": related_title})

        quality = _compute_quality_score(nodes, edges)

        toc = [{"title": n.title, "id": n.id, "level": 1} for n in nodes]
        return CompiledWiki(
            nodes=nodes,
            edges=edges,
            toc=toc,
            quality_score=quality,
            metadata={
                "chunk_count": len(chunks),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "two_phase": True,
                "concepts_merged": len(concepts),
                "context_window": cw.max_input,
            },
        )

    def _extract_one(self, source: str, combined: str, extraction_system: str) -> Extraction:
        """Run a single extraction LLM call with cache (Tier 3 Phase 1)."""
        budget = self._budget
        extract_deps = {
            "source_text": hashlib.sha256(combined.encode()).hexdigest(),
            "model": self._model,
            "chunk_size_min": str(budget.chunk_size_min),
            "chunk_size_max": str(budget.chunk_size_max),
        }
        if not self._config.no_cache:
            cached = self._cache_store.get("extract", extract_deps)
            if cached is not None:
                return cached

        prompt = _loader.render("compilation/extraction_prompt.md.j2", text=combined)
        data = _try_litellm_json(prompt, system=extraction_system, model=self._model, timeout=self._timeout)
        if isinstance(data, dict):
            result = Extraction(
                source=source,
                summary=str(data.get("summary", "")),
                entities=[dict(e) for e in data.get("entities", []) if isinstance(e, dict)],
                concepts=[dict(c) for c in data.get("concepts", []) if isinstance(c, dict)],
                tags=[str(t) for t in data.get("tags", []) if t],
            )
            if not self._config.no_cache:
                self._cache_store.set("extract", extract_deps, result)
            return result
        return _fallback_extraction(source, combined)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gen_cache_deps(self, extractions: ExtractionSet) -> dict[str, str]:
        """Compute dependency hashes for generation cache."""
        blob = "|".join(json.dumps(e.to_dict(), sort_keys=True) for e in extractions.extractions)
        ext_hash = hashlib.sha256(blob.encode()).hexdigest()
        budget = self._budget
        return {
            "extractions": ext_hash,
            "model": self._model,
            "chunk_size_min": str(budget.chunk_size_min),
            "chunk_size_max": str(budget.chunk_size_max),
            "concepts_per_batch": str(budget.concepts_per_batch),
        }

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
