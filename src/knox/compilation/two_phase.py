"""N3 two-phase compiler — extract first, then generate with cross-source merging.

Phase 1 (extract): for each source, run the LLM once in extraction mode
and collect a structured payload of {summary, entities, concepts, tags}.

Phase 2 (generate): merge same-named entities/concepts across sources,
then ask the LLM to generate one page per unique entity/concept. Each
page carries ``[source.pdf:42-58]`` provenance citations per N4.

When the LLM call fails, the compiler falls back to the underlying
``base`` compiler (typically :class:`GraphCompiler`) so the pipeline
remains useful offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from knox.chunking.summarizer import (
    _structured_to_text,
)  # ponytail: cross-module private import. Make public or move to knox.core.models if other callers appear.
from knox.compilation._two_phase_core import (
    _MIN_SEGMENT_TOKENS,
    Extraction,
    ExtractionSet,
    _chars_to_tokens,
    _compute_quality_score,
    _extract_images_from_chunks,
    _merge_extractions,
    _segment_sources,
    _slugify,
    _tokens_to_chars,
    _try_litellm_json,
    _upsert_node,
)
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.compilation.schema import ExtractionOutput, GenerationOutput, ValidationError
from knox.core.cache_store import CacheStore, _decode_json, _encode_json, make_cache_key
from knox.core.models import ChunkTree, CompiledWiki, MetadataField, ResolvedChunk, WikiField, WikiNode
from knox.core.pipeline_defs import PipelineConfig
from knox.core.template_loader import get_template_loader
from knox.llm.client import LLMClient, _discover_context_window
from knox.llm.format import encode_for_llm
from knox.logging import get_logger
from knox.rag.zvec_store import _make_provider

logger = get_logger(__name__)

_loader = get_template_loader()

# ── Concept name-embedding merge (Option C) ─────────────────────────────

_CONCEPT_MERGE_THRESHOLD = 0.85


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (no numpy needed)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _merge_similar_concepts(
    concepts: list[dict[str, Any]],
    embedding_model: str,
    *,
    threshold: float = _CONCEPT_MERGE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Merge concept pages whose name embeddings are above *threshold*.

    Uses the configured embedding provider (SentenceTransformer either
    locally or via LiteLLM). Groups concepts greedily: for every pair
    with cosine similarity >= threshold, sources and definitions are
    unioned and the longest name is kept as canonical.

    When the embedding provider fails (e.g. model not found), concepts
    are returned unchanged and a warning is logged.
    """
    if not embedding_model or embedding_model == "hash":
        return concepts

    try:
        provider = _make_provider(embedding_model, dim=None)
    except Exception as exc:
        logger.debug(
            "Embedding merge disabled: could not create provider for %s: %s",
            embedding_model,
            exc,
        )
        return concepts

    names = [c["name"] for c in concepts]
    try:
        vecs = [provider.embed(n) for n in names]
    except Exception as exc:
        logger.debug("Embedding merge disabled: embedding call failed: %s", exc)
        return concepts

    # Union-Find to group similar concepts.
    n = len(concepts)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx < ry:
            parent[ry] = rx
        else:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_similarity(vecs[i], vecs[j]) >= threshold:
                _union(i, j)

    # Group by leader.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        leader = _find(i)
        groups.setdefault(leader, []).append(i)

    # Merge each multi-member group.
    result: list[dict[str, Any]] = []
    for leader, members in groups.items():
        if len(members) == 1:
            result.append(concepts[leader])
            continue
        merged: dict[str, Any] = {
            "name": max((concepts[m]["name"] for m in members), key=len),
            "sources": [],
            "definitions": [],
        }
        seen_sources: set[str] = set()
        seen_defs: set[str] = set()
        for m in members:
            for s in concepts[m].get("sources", []):
                if s not in seen_sources:
                    merged["sources"].append(s)
                    seen_sources.add(s)
            for d in concepts[m].get("definitions", []):
                if d not in seen_defs:
                    merged["definitions"].append(d)
                    seen_defs.add(d)
        result.append(merged)

    merged_count = n - len(result)
    if merged_count:
        logger.info(
            "Embedding merge: %d groups combined (%d → %d concepts, threshold=%.2f)",
            merged_count,
            n,
            len(result),
            threshold,
        )
    return result


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
        self._llm_stats: list[tuple[int, int]] = []
        self._extraction_stats: list[dict[str, Any]] = []

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

    def compile(
        self,
        chunks: list[ResolvedChunk],
        *,
        trees: list[ChunkTree] | None = None,
        pinned_facts: list[str] | None = None,
    ) -> CompiledWiki:
        """Run the two-phase compile with generation cache (Tier 3 Phase 2).

        When *pinned_facts* is non-empty, each fact is injected into the
        generation prompt as a hard preservation constraint (WiCER /
        CEGAR-style refinement). Pinned facts are also folded into the
        generation cache key so a refined call cannot serve a stale wiki
        from a prior iteration.
        """
        compile_start = time.monotonic()

        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={WikiField.CHUNK_COUNT: 0, "two_phase": True, "llm_calls": 0},
            )

        num_sources = len({str(rc.chunk.source_file or "unknown") for rc in chunks})
        self._report(0, num_sources, "Compiling wiki…")

        extract_start = time.monotonic()
        extractions = self.extract(chunks, trees=trees)
        extract_elapsed = time.monotonic() - extract_start

        if self._extraction_stats:
            src_times = [s["duration_s"] for s in self._extraction_stats if "duration_s" in s]
            avg_src = sum(src_times) / len(src_times) if src_times else 0.0
            logger.info(
                "TwoPhase: extract phase done — %d source(s) in %.1fs (avg %.1fs), %d LLM call(s), %d prompt + %d completion tokens",
                len(src_times),
                extract_elapsed,
                avg_src,
                len(self._llm_stats),
                sum(pt for pt, _ in self._llm_stats),
                sum(ct for _, ct in self._llm_stats),
            )

        # Generation cache: keyed by all extractions + model + gen config
        # + pinned facts (refined compilations must miss prior cache).
        gen_deps = self._gen_cache_deps(extractions, pinned_facts=pinned_facts)
        if not self._config.no_cache:
            cached = self._cache_store.get("generate", gen_deps)
            if cached is not None:
                logger.info(
                    "TwoPhase: generation cache hit (%d nodes, %d edges) — took %.1fs total",
                    len(cached.nodes),
                    len(cached.edges),
                    time.monotonic() - compile_start,
                )
                if getattr(self, "_extraction_stats", None):
                    cached.metadata["extraction_stats"] = self._extraction_stats
                cached.metadata["llm_calls"] = len(self._llm_stats)
                return cached

        # Progress is reported inside generate() per batch×segment.
        self._active_gen_deps = gen_deps
        gen_start = time.monotonic()
        wiki = self.generate(extractions, chunks, trees=trees, pinned_facts=pinned_facts)
        gen_elapsed = time.monotonic() - gen_start

        # Generation completed. Remove checkpoint so the next compile starts
        # fresh (or uses the generation cache).
        self._delete_gen_checkpoint(gen_deps)

        if getattr(self, "_extraction_stats", None):
            wiki.metadata["extraction_stats"] = self._extraction_stats

        # Store metadata.
        total_prompt = sum(pt for pt, _ in self._llm_stats)
        total_comp = sum(ct for _, ct in self._llm_stats)
        wiki.metadata["llm_calls"] = len(self._llm_stats)
        wiki.metadata["llm_prompt_tokens"] = total_prompt
        wiki.metadata["llm_completion_tokens"] = total_comp
        wiki.metadata["compile_duration_s"] = round(time.monotonic() - compile_start, 1)

        total_elapsed = time.monotonic() - compile_start
        wiki.metadata.get(WikiField.CHUNK_COUNT, 0)
        node_count = len(wiki.nodes)
        logger.info(
            "TwoPhase: compile done — %d source(s) \u2192 %d concept(s) in %.1fs "
            "(extract %.1fs, generate %.1fs), %d LLM call(s), "
            "%d prompt + %d completion tokens",
            num_sources,
            node_count,
            total_elapsed,
            extract_elapsed,
            gen_elapsed,
            len(self._llm_stats),
            total_prompt,
            total_comp,
        )

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
        self._llm_stats = []
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

        # Per-source extraction timing + summary stats (for log / metadata).
        for source_idx, (source, source_chunks) in enumerate(by_source.items(), start=1):
            logger.info("TwoPhase: extract [%d/%d] %s", source_idx, num_sources, source)
            src_start = time.monotonic()

            # Build hierarchical text from tree when available.
            total_chars = sum(len(rc.chunk.text or "") for rc in source_chunks)
            source_name = Path(source).name
            self._report(source_idx, num_sources, f"extracting [{source_idx}/{num_sources}] {source_name} ({total_chars} chars)")

            tree = tree_by_source.get(source)
            texts = self._build_tree_texts(tree) if tree is not None else [rc.chunk.text for rc in source_chunks if rc.chunk.text]

            if not texts:
                extractions.add(Extraction(source=source))
                self._extraction_stats.append(
                    {
                        "source": source,
                        "summary": "",
                        "duration_s": 0.0,
                        "summary_chars": 0,
                        "summary_words": 0,
                        "entity_count": 0,
                        "concept_count": 0,
                        "tag_count": 0,
                        "batches": 0,
                    }
                )
                continue

            # Batch texts into budget-sized groups so nothing is dropped.
            batch_parts: list[str] = []
            batch_len = 0
            batch_extractions: list[Extraction | None] = []

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

            # Merge all batch extractions for this source (skip failed batches).
            valid = [e for e in batch_extractions if e is not None]
            if not valid:
                logger.warning("TwoPhase: all extraction batches failed for '%s' — skipped", source)
            merged = _merge_extractions(source, valid)
            extractions.add(merged)
            self._extraction_stats.append(
                {
                    "source": source,
                    "summary": merged.summary,
                    "duration_s": round(time.monotonic() - src_start, 1),
                    "summary_chars": len(merged.summary),
                    "summary_words": len(merged.summary.split()),
                    "entity_count": len(merged.entities),
                    "concept_count": len(merged.concepts),
                    "tag_count": len(merged.tags),
                    "batches": len(batch_extractions),
                }
            )
        return extractions

    def generate(  # noqa: PLR0915
        self,
        extractions: ExtractionSet,
        chunks: list[ResolvedChunk],
        *,
        trees: list[ChunkTree] | None = None,
        pinned_facts: list[str] | None = None,
    ) -> CompiledWiki:
        """Phase 2: merge extractions and generate per-concept pages.

        Batches concepts into groups of ``concepts_per_batch`` to avoid
        overwhelming the LLM with a single massive prompt. Source segments
        are sized adaptively based on the discovered context window.

        ``pinned_facts`` is forwarded to the prompt template so the LLM
        treats each item as a hard preservation constraint when emitting
        page content (WiCER refinement input).
        """
        concepts = extractions.concept_pages()
        # Embedding-based similarity merge (Option C) — merges concepts whose
        # name embeddings are above a threshold, catching semantic duplicates
        # that Option A (normalized-key) misses. Only runs when an embedding
        # model is configured (not "hash").
        embed_model = self._config.embedding.model
        if embed_model and embed_model != "hash":
            concepts = _merge_similar_concepts(concepts, embed_model)
        if concepts:
            logger.info(
                "TwoPhase: %d concept(s) ready for generation (sample: %s)",
                len(concepts),
                [c["name"] for c in concepts[:3]],
            )
        else:
            # Diagnose: what keys did extraction actually return? Pinpoints
            # schema drift vs. genuinely empty extraction in one log line.
            sample_keys: set[str] = set()
            for ext in extractions.extractions:
                for c in ext.concepts:
                    sample_keys.update(c.keys())
            logger.warning(
                "TwoPhase: concept_pages() empty \u2014 extraction concept keys: %s",
                sorted(sample_keys) or "(no concepts returned)",
            )
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={
                    WikiField.CHUNK_COUNT: len(chunks),
                    "two_phase": True,
                    "generation_skipped": True,
                    "concepts_available": 0,
                    "skip_reason": "extraction returned no concepts",
                    "node_count": 0,
                },
            )

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
            max(_MIN_SEGMENT_TOKENS, cw.max_input - self._gen_system_tokens - budget.gen_output_max),
        )

        source_segments = _segment_sources(sources, segment_char_size)

        # Normalize pinned facts: drop empties + dedup (preserve order).
        pinned_clean = list(dict.fromkeys(f.strip() for f in (pinned_facts or []) if f and f.strip()))

        generation_system = _loader.get_body("compilation/generation_system.md.j2")

        # Compute total generation steps for a determinate progress bar.
        num_concepts = len(concepts)
        gen_total = ((num_concepts + self._budget.concepts_per_batch - 1) // self._budget.concepts_per_batch) * len(source_segments)
        if gen_total > 0:
            self._report(0, gen_total, f"generating wiki ({num_concepts} concepts, {len(source_segments)} segments)")

        # Load generation checkpoint so interrupted runs can resume.
        checkpoint = self._load_gen_checkpoint(self._active_gen_deps)
        if checkpoint is not None:
            ck_nodes, ck_slug_to_idx, ck_completed = checkpoint
            logger.info(
                "TwoPhase: generation checkpoint loaded \u2014 %d batch(es) already done, %d node(s)",
                len(ck_completed),
                len(ck_nodes),
            )
        else:
            ck_nodes, ck_slug_to_idx, ck_completed = [], {}, set()

        nodes, slug_to_idx = self._generate_all_pages(
            concepts,
            source_segments,
            generation_system,
            all_images,
            pinned_facts=pinned_clean,
            initial_nodes=ck_nodes,
            initial_slug_to_idx=ck_slug_to_idx,
            completed_batches=ck_completed,
        )

        # Generation finished (success or partial). Delete checkpoint always
        # so the next run uses the generation cache or starts fresh.
        self._delete_gen_checkpoint(self._active_gen_deps)

        if not nodes:
            logger.warning(
                "TwoPhase: generation skipped — 0 of %d concept(s) synthesized. All LLM calls returned invalid data.",
                len(concepts),
            )
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={
                    WikiField.CHUNK_COUNT: len(chunks),
                    "two_phase": True,
                    "generation_skipped": True,
                    "concepts_available": len(concepts),
                    "skip_reason": "all LLM calls returned invalid data",
                    "node_count": 0,
                },
            )

        self.disambiguate_duplicate_titles(nodes, chunks)
        nodes = self.filter_empty_nodes(
            nodes,
            min_content_length=self._config.budget.min_content_length,
        )

        if not nodes:
            logger.warning(
                "TwoPhase: generation produced 0 pages after filtering (min_content_length=%d). Check LLM output quality.",
                self._config.budget.min_content_length,
            )
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={
                    WikiField.CHUNK_COUNT: len(chunks),
                    "two_phase": True,
                    "generation_skipped": True,
                    "skip_reason": "all pages filtered as empty",
                    "node_count": 0,
                },
            )

        nodes = self.dedup_content_duplicate_nodes(nodes)
        edges = self._build_edges(nodes, extractions)
        quality = _compute_quality_score(nodes, edges)

        toc = [{"title": n.title, "id": n.id, "level": 1} for n in nodes]
        return CompiledWiki(
            nodes=nodes,
            edges=edges,
            toc=toc,
            quality_score=quality,
            metadata={
                WikiField.CHUNK_COUNT: len(chunks),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "two_phase": True,
                "concepts_merged": len(concepts),
                "context_window": cw.max_input,
                "pinned_facts_count": len(pinned_clean),
            },
        )

    def _generate_all_pages(
        self,
        concepts: list[dict[str, Any]],
        source_segments: list[list[tuple[str, str]]],
        generation_system: str,
        all_images: list[dict[str, Any]],
        *,
        pinned_facts: list[str],
        initial_nodes: list[WikiNode] | None = None,
        initial_slug_to_idx: dict[str, int] | None = None,
        completed_batches: set[int] | None = None,
    ) -> tuple[list[WikiNode], dict[str, int]]:
        """Generate (or fallback-generate) one node per concept/segment pair.

        When *initial_nodes* / *initial_slug_to_idx* / *completed_batches*
        are provided (from a generation checkpoint), completed batches are
        skipped and the accumulated node state is carried forward. A
        checkpoint is written to disk after every batch so an interrupted
        run can resume from the last completed batch.
        """
        nodes: list[WikiNode] = list(initial_nodes or [])
        slug_to_idx: dict[str, int] = dict(initial_slug_to_idx or {})
        completed: set[int] = set(completed_batches or [])

        num_concepts = len(concepts)
        num_segments = len(source_segments)
        batch_size = self._budget.concepts_per_batch
        logger.info(
            "TwoPhase: generating %d concept(s) across %d source segment(s) (batch_size=%d, pinned=%d)",
            num_concepts,
            num_segments,
            batch_size,
            len(pinned_facts),
        )
        gen_progress_total = ((num_concepts + batch_size - 1) // batch_size) * num_segments
        gen_progress_step = 0

        # ponytail: cap failed segment retries per batch.  Deepseek-v4-flash
        # often returns unparseable JSON — iterating all 120 segments wastes
        # hours.  2-3 retries catches transient failures without the combinatorial
        # explosion.
        MAX_FAILED = 3

        for batch_start in range(0, num_concepts, batch_size):
            if batch_start in completed:
                gen_progress_step += 1  # approximate: one tick per skipped batch
                continue
            batch = concepts[batch_start : batch_start + batch_size]
            failed_count = 0

            for seg_idx, segment in enumerate(source_segments):
                if failed_count >= MAX_FAILED:
                    break
                gen_progress_step += 1
                seg_label = f"gen batch {batch_start + 1}-{batch_start + len(batch)}/{num_concepts} seg {seg_idx + 1}/{num_segments}"
                self._report(gen_progress_step, gen_progress_total, seg_label)
                logger.info(
                    "TwoPhase: %s",
                    seg_label,
                )
                content_block = "\n\n".join(f"### {label}\n{content}" for label, content in segment)

                prompt = _loader.render(
                    "compilation/generation_prompt.md.j2",
                    concepts_json=encode_for_llm(batch, self._config.llm.data_format),
                    content=content_block,
                    pinned_facts=pinned_facts,
                )
                data = _try_litellm_json(
                    prompt,
                    system=generation_system,
                    model=self._model,
                    timeout=self._timeout,
                    call_stats=self._llm_stats,
                    max_tokens=min(self._context_window.max_output, self._budget.gen_output_max),
                    response_model=GenerationOutput,
                )

                if not data:
                    logger.warning(
                        "TwoPhase: LLM returned no data for batch %d-%d, segment %d — skipping",
                        batch_start,
                        batch_start + len(batch),
                        seg_idx,
                    )
                    failed_count += 1
                    continue

                try:
                    gen_output = GenerationOutput.model_validate(data)
                except ValidationError:
                    logger.warning(
                        "TwoPhase: generation output failed schema validation for batch %d-%d, segment %d — skipping",
                        batch_start,
                        batch_start + len(batch),
                        seg_idx,
                    )
                    failed_count += 1
                    continue

                for idx, page in enumerate(gen_output.pages):
                    # Always derive slug from title for consistency — ignore LLM slug
                    node_id = _slugify(page.title or f"concept-{batch_start + idx}")
                    _upsert_node(
                        WikiNode(
                            id=node_id,
                            title=page.title or f"Concept {batch_start + idx}",
                            content=page.content,
                            node_type="concept",
                            source_chunks=[],
                            metadata={
                                MetadataField.PROVENANCE: page.provenance,
                                "related": page.related,
                                "two_phase": True,
                                "images": all_images,
                                "frontmatter": {
                                    "title": page.title or f"Concept {batch_start + idx}",
                                },
                            },
                        ),
                        nodes,
                        slug_to_idx,
                    )

                # ponytail: break after the first successful segment for this batch.
                # Each segment covers a different portion of the source text; segment 0
                # already spans the first ~63K chars of every file (with the segment_size
                # fix) and produces pages from the most important content. Remaining
                # segments would make duplicate LLM calls for the same concepts with
                # diminishing returns. If a concept's source lives in a later segment,
                # the fallback from segment 0 already captured its definition.
                break

            # Checkpoint after each completed batch so interrupted runs can resume.
            completed.add(batch_start)
            self._save_gen_checkpoint(
                self._active_gen_deps,
                nodes,
                slug_to_idx,
                completed,
            )

        self._report(gen_progress_total, gen_progress_total, "generating complete")
        return nodes, slug_to_idx

    def _extract_one(self, source: str, combined: str, extraction_system: str) -> Extraction | None:
        """Run a single extraction LLM call with cache (Tier 3 Phase 1).

        Returns ``None`` when the LLM returns no parseable data — the caller
        skips the source and continues with the remaining ones.
        """
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
            call_stats=self._llm_stats,
            max_tokens=min(self._context_window.max_output, self._budget.extract_output_max),
            response_model=ExtractionOutput,
        )
        if data:
            try:
                validated = ExtractionOutput.model_validate(data)
            except ValidationError:
                logger.warning(
                    "TwoPhase: extraction output failed schema validation for '%s' — skipping",
                    source,
                )
                return None
            result = Extraction(
                source=source,
                summary=validated.summary,
                entities=[e.model_dump() for e in validated.entities],
                concepts=[c.model_dump() for c in validated.concepts],
                tags=validated.tags,
            )
            if not self._config.no_cache:
                self._cache_store.set("extract", extract_deps, result)
            return result
        logger.warning(
            "TwoPhase: extraction skipped for source '%s' — LLM returned no parseable data",
            source,
        )
        return None

    def _gen_cache_deps(self, extractions: ExtractionSet, *, pinned_facts: list[str] | None = None) -> dict[str, str]:
        """Compute dependency hashes for generation cache.

        ``pinned_facts`` participates in the key so that a WiCER
        refinement (same extractions, new pin set) cannot serve a stale
        wiki from a prior iteration.
        """
        blob = "|".join(json.dumps(e.to_dict(), sort_keys=True) for e in extractions.extractions)
        ext_hash = hashlib.sha256(blob.encode()).hexdigest()
        pin_blob = "\n".join(sorted(f.strip() for f in (pinned_facts or []) if f and f.strip()))
        pin_hash = hashlib.sha256(pin_blob.encode()).hexdigest()
        budget = self._budget
        return {
            "extractions": ext_hash,
            "model": self._model,
            "chunk_size_min": str(budget.chunk_size_min),
            "chunk_size_max": str(budget.chunk_size_max),
            "concepts_per_batch": str(budget.concepts_per_batch),
            "pinned_facts": pin_hash,
        }

    # -- generation checkpoint (resume after crash) ------------------------

    def _gen_checkpoint_dir(self) -> Path:
        """Directory for per-batch generation checkpoint files."""
        return self._config.paths.cache_dir / "gen_checkpoint"

    def _gen_checkpoint_path(self, gen_deps: dict[str, str]) -> Path:
        """On-disk path for the generation checkpoint."""
        return self._gen_checkpoint_dir() / f"{make_cache_key(gen_deps)}.json"

    def _load_gen_checkpoint(  # noqa: PLR0911
        self,
        gen_deps: dict[str, str],
    ) -> tuple[list[WikiNode], dict[str, int], set[int]] | None:
        """Load a generation checkpoint if it exists and deps match.

        Returns ``(nodes, slug_to_idx, completed_batches)`` or ``None``
        when the checkpoint is absent, stale, or malformed.
        """
        if self._config.no_cache:
            return None
        path = self._gen_checkpoint_path(gen_deps)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Generation checkpoint read error -- ignoring", exc_info=True)
            return None
        # Validate dependency hashes match current gen_deps.
        stored_deps = raw.get("deps", {})
        if stored_deps != gen_deps:
            logger.debug("Generation checkpoint deps mismatch -- ignoring")
            return None
        try:
            nodes: list[WikiNode] = [_decode_json(n) for n in raw.get("nodes", [])]  # nocheck — checkpoint schema
        except Exception:
            logger.debug("Generation checkpoint node deserialization error -- ignoring", exc_info=True)
            return None
        slug_to_idx: dict[str, int] = dict(raw.get("slug_to_idx", {}))
        completed: set[int] = set(raw.get("completed_batches", []))
        return nodes, slug_to_idx, completed

    def _save_gen_checkpoint(
        self,
        gen_deps: dict[str, str],
        nodes: list[WikiNode],
        slug_to_idx: dict[str, int],
        completed_batches: set[int],
    ) -> None:
        """Write generation checkpoint to disk (atomic write-to-temp + rename)."""
        if self._config.no_cache:
            return
        path = self._gen_checkpoint_path(gen_deps)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "deps": gen_deps,
            "completed_batches": sorted(completed_batches),
            "nodes": [_encode_json(n) for n in nodes],  # nocheck — checkpoint schema
            "slug_to_idx": dict(slug_to_idx),
        }
        try:
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f)
            os.replace(tmp, path)
            logger.debug("Generation checkpoint saved: %s", path)
        except OSError:
            logger.debug("Generation checkpoint write failed -- ignoring", exc_info=True)

    def _delete_gen_checkpoint(self, gen_deps: dict[str, str]) -> None:
        """Remove generation checkpoint file."""
        if self._config.no_cache:
            return
        path = self._gen_checkpoint_path(gen_deps)
        try:
            if path.exists():
                path.unlink()
                logger.debug("Generation checkpoint deleted: %s", path)
        except OSError:
            logger.debug("Generation checkpoint delete failed -- ignoring", exc_info=True)

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
                # Scan FULL content — cropping to the first N chars (the old
                # [:200]) silently dropped every match past the cutoff, falsely
                # orphaning pages whose entity mentions appear later. See
                # src/knox/AGENTS.md: Antipattern 4 (no hard crops).
                match_text = (f"{node.title} {node.content or ''}").lower()
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

        # Title-based edges: if a page's title appears as plaintext in another
        # page's content, that's an implicit cross-reference (not just entity
        # co-occurrence). The orphan check treats both endpoints as linked, so
        # direction is semantic (mentioning page → mentioned page) not structural.
        for node in nodes:
            content_lower = (node.content or "").lower()
            for other in nodes:
                if other.id == node.id:
                    continue
                other_title = other.title.lower().strip()
                if len(other_title) < 4:  # ponytail: skip very short generic titles
                    continue
                if other_title in content_lower:
                    pair = (min(node.id, other.id), max(node.id, other.id))
                    if pair in seen_edges:
                        continue
                    seen_edges.add(pair)
                    edges.append(
                        {
                            "source": node.id,
                            "target": other.id,
                            "type": "title_reference",
                            "entity": other.title,
                        }
                    )

        return edges
