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
from collections import defaultdict
from pathlib import Path
from typing import Any

from knox.chunking.summarizer import (
    _structured_to_text,
)  # ponytail: cross-module private import. Make public or move to knox.core.models if other callers appear.
from knox.compilation._two_phase_core import (
    Extraction,
    ExtractionSet,
    _chars_to_tokens,
    _compute_quality_score,
    _extract_images_from_chunks,
    _fallback_extraction,
    _merge_extractions,
    _segment_sources,
    _slugify,
    _tokens_to_chars,
    _try_litellm_json,
    _upsert_node,
)
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.core.cache_store import CacheStore
from knox.core.models import ChunkTree, CompiledWiki, ResolvedChunk, WikiNode
from knox.core.pipeline_defs import PipelineConfig
from knox.core.template_loader import get_template_loader
from knox.llm.client import LLMClient, _discover_context_window

logger = logging.getLogger(__name__)

_loader = get_template_loader()

# Re-export internal helpers for backward compatibility.
__all__ = [
    "Extraction",
    "ExtractionSet",
    "TwoPhaseCompiler",
    "_chars_to_tokens",
    "_tokens_to_chars",
]


class TwoPhaseCompiler(BaseCompiler):
    """N3 two-phase compiler with cross-source merging and adaptive budgets.

    Parameters
    ----------
    config:
        Pipeline configuration including budget and context window settings.
    base:
        Compiler used as a fallback when the LLM is unavailable.
    llm_client:
        LLM client whose ``context_window`` is reused when available. When
        ``None``, the context window is discovered independently from litellm
        or Ollama as a fallback.
    """

    def __init__(
        self,
        config: PipelineConfig,
        base: BaseCompiler | None = None,
        *,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._base = base or GraphCompiler(model=config.llm.default_model)
        self._budget = config.budget
        self._model = config.llm.default_model
        self._config = config
        self._timeout = config.llm.timeout
        self._cache_store = CacheStore(config.paths.cache_dir)

        # Reuse the client's discovered context window when available; only
        # discover independently as a fallback (e.g. no LLM client provided).
        if llm_client is not None and llm_client.context_window is not None:
            cw = llm_client.context_window
        else:
            cw = _discover_context_window(
                self._model,
                fallback_max_input=config.llm.fallback_max_input,
                fallback_max_output=config.llm.fallback_max_output,
                num_ctx_min=config.llm.num_ctx_min,
                num_ctx_max=config.llm.num_ctx_max,
                context_window_cap=config.llm.context_window_cap,
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

    def compile(self, chunks: list[ResolvedChunk], *, trees: list[ChunkTree] | None = None) -> CompiledWiki:
        """Run the two-phase compile with generation cache (Tier 3 Phase 2)."""
        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={"chunk_count": 0, "two_phase": True},
            )

        extractions = self.extract(chunks, trees=trees)

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

        wiki = self.generate(extractions, chunks, trees=trees)

        if not self._config.no_cache:
            self._cache_store.set("generate", gen_deps, wiki)
        return wiki

    def extract(self, chunks: list[ResolvedChunk], *, trees: list[ChunkTree] | None = None) -> ExtractionSet:
        """Phase 1: run the LLM in extraction mode for each source.

        When *trees* are provided, extraction input is built from the
        ChunkTree hierarchy (internal node summaries + leaf text) rather
        than flat chunk text. This gives the LLM a structured view of
        the document with multi-resolution context.

        Source text is processed in batches sized to fill the context window
        (minus system prompt and output budget) so that no content is silently
        dropped.  Results from each batch are merged into a single
        :class:`Extraction` per source.
        """
        extractions = ExtractionSet()
        # Group chunks by source file so each source is extracted once.
        by_source: dict[str, list[ResolvedChunk]] = defaultdict(list)
        for rc in chunks:
            label = str(rc.chunk.source_file or "unknown")
            by_source[label].append(rc)

        # Build tree lookup when available.
        tree_by_source: dict[str, ChunkTree] = {}
        if trees:
            for t in trees:
                if t.source_file is not None:
                    tree_by_source[str(t.source_file)] = t

        extraction_system = _loader.get_body("compilation/extraction_system.md.j2")
        num_sources = len(by_source)
        logger.info("TwoPhase: extracting %d source(s)", num_sources)

        # Adaptive extract batch size: fill context window (no clamping).
        cw = self._context_window
        budget = self._budget
        extract_char_budget = _tokens_to_chars(
            max(1, cw.max_input - self._extract_system_tokens - budget.extract_output_max),
        )

        for source_idx, (source, source_chunks) in enumerate(by_source.items(), start=1):
            logger.info("TwoPhase: extract [%d/%d] %s", source_idx, num_sources, source)

            # Build hierarchical text from tree when available.
            tree = tree_by_source.get(source)
            texts = self._build_tree_texts(tree) if tree is not None else [rc.chunk.text for rc in source_chunks if rc.chunk.text]

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

    def generate(self, extractions: ExtractionSet, chunks: list[ResolvedChunk], *, trees: list[ChunkTree] | None = None) -> CompiledWiki:  # noqa: PLR0915
        """Phase 2: merge extractions and generate per-concept pages.

        Batches concepts into groups of ``concepts_per_batch`` to avoid
        overwhelming the LLM with a single massive prompt. Source segments
        are sized adaptively based on the discovered context window.
        """
        concepts = extractions.concept_pages()
        if not concepts:
            # Nothing to merge — fall back to the base compiler.
            return self._fallback_compile(chunks, trees=trees)

        # Collect images from IMAGE chunks for embedding in output
        all_images = _extract_images_from_chunks(chunks)

        # Per-source content, collected in full.
        by_source: dict[str, str] = defaultdict(str)
        for rc in chunks:
            label = Path(str(rc.chunk.source_file or "unknown")).stem
            if rc.chunk.text:
                by_source[label] += rc.chunk.text + "\n\n"
        sources: list[tuple[str, str]] = list(by_source.items())

        # Adaptive segment size: fill context window (no clamping).
        cw = self._context_window
        budget = self._budget
        segment_char_size = _tokens_to_chars(
            max(1, cw.max_input - self._gen_system_tokens - budget.gen_output_max),
        )

        source_segments = _segment_sources(sources, segment_char_size)

        generation_system = _loader.get_body("compilation/generation_system.md.j2")
        nodes, slug_to_idx = self._generate_all_pages(
            concepts,
            source_segments,
            generation_system,
            all_images,
        )

        if not nodes:
            return self._fallback_compile(chunks, trees=trees)

        self.disambiguate_duplicate_titles(nodes, chunks)
        nodes = self.filter_empty_nodes(
            nodes,
            min_content_length=self._config.budget.min_content_length,
        )

        if not nodes:
            return self._fallback_compile(chunks)

        edges = self._build_edges(nodes, extractions)
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

    def _generate_all_pages(
        self,
        concepts: list[dict[str, Any]],
        source_segments: list[list[tuple[str, str]]],
        generation_system: str,
        all_images: list[dict[str, Any]],
    ) -> tuple[list[WikiNode], dict[str, int]]:
        """Generate (or fallback-generate) one node per concept/segment pair."""
        nodes: list[WikiNode] = []
        slug_to_idx: dict[str, int] = {}

        num_concepts = len(concepts)
        num_segments = len(source_segments)
        batch_size = self._budget.concepts_per_batch
        logger.info(
            "TwoPhase: generating %d concept(s) across %d source segment(s) (batch_size=%d)",
            num_concepts,
            num_segments,
            batch_size,
        )
        for batch_start in range(0, num_concepts, batch_size):
            batch = concepts[batch_start : batch_start + batch_size]

            # Iterate through ALL source segments for this concept batch,
            # generating pages from each segment and merging.
            for seg_idx, segment in enumerate(source_segments):
                logger.info(
                    "TwoPhase: gen batch %d-%d/%d segment %d/%d",
                    batch_start,
                    len(batch),
                    num_concepts,
                    seg_idx + 1,
                    num_segments,
                )
                content_block = "\n\n".join(f"### {label}\n{content}" for label, content in segment)

                prompt = _loader.render(
                    "compilation/generation_prompt.md.j2",
                    concepts_json=json.dumps(batch),
                    content=content_block,
                )
                data = _try_litellm_json(
                    prompt,
                    system=generation_system,
                    model=self._model,
                    timeout=self._timeout,
                )

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
                                ),
                                nodes,
                                slug_to_idx,
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
                        ),
                        nodes,
                        slug_to_idx,
                    )

        return nodes, slug_to_idx

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
        data = _try_litellm_json(
            prompt,
            system=extraction_system,
            model=self._model,
            timeout=self._timeout,
        )
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

    def _fallback_compile(self, chunks: list[ResolvedChunk], *, trees: list[ChunkTree] | None = None) -> CompiledWiki:
        if self._base is not None:
            wiki = self._base.compile(chunks, trees=trees)
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

    @staticmethod
    def _build_tree_texts(tree: ChunkTree) -> list[str]:
        """Build hierarchical text blocks from a ChunkTree.

        Internal nodes produce heading + summary. Leaf nodes produce
        full text. The result is a list of text blocks that preserve
        the document's heading hierarchy.
        """
        texts: list[str] = []
        for node in tree.walk():
            if node.chunk is None and not node.structured and not node.summary:
                continue
            heading_prefix = ""
            if node.heading_path:
                heading_prefix = f"{'#' * min(node.depth + 1, 6)} {' / '.join(node.heading_path)}\n\n"
            if node.chunk is not None:
                # Leaf prose node — full text.
                texts.append(f"{heading_prefix}{node.chunk.text}")
            elif node.structured is not None:
                # Leaf structured node — text representation.
                texts.append(f"{heading_prefix}{_structured_to_text(node.structured)}")
            elif node.summary:
                # Internal node — summary only.
                texts.append(f"{heading_prefix}[summary: {node.summary}]")
        return texts

    @staticmethod
    def _build_edges(
        nodes: list[WikiNode],
        extractions: ExtractionSet,
    ) -> list[dict[str, Any]]:
        """Build cross-source entity and explicit related edges."""
        edges: list[dict[str, Any]] = []
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
                    pair = (min(node.id, target_id), max(node.id, target_id))
                    if pair in seen_edges:
                        continue
                    seen_edges.add(pair)
                    edges.append(
                        {"source": node.id, "target": target_id, "type": "related", "entity": related_title},
                    )

        return edges
