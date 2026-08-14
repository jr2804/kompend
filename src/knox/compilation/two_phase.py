"""N3 two-phase compiler — extract first, then generate with cross-source merging.

Phase 1 (extract): for each source, run the LLM once in extraction mode
and collect a structured payload of {summary, entities, concepts, tags}.

Phase 2 (generate): merge same-named entities/concepts across sources,
then ask the LLM to generate one page per unique entity/concept. Each
page carries ``[source.pdf:42-58]`` provenance citations per N4.

When the LLM call fails, the compiler falls back to the underlying
``base`` compiler (typically :class:`GraphCompiler`) so the pipeline
remains useful offline.

Implementation split into :mod:`knox.compilation.flows` submodules:
- :mod:`flows.extraction` — extraction phase
- :mod:`flows.generation` — generation phase + wiki assembly
- :mod:`flows.assembly` — pure helpers (edges, tags, tree texts, concept merge)
"""

from __future__ import annotations

import time
from typing import Any

from knox.compilation._checkpoints import delete_gen_checkpoint
from knox.compilation._two_phase_core import (
    BatchFailureTracker,
    Extraction,
    ExtractionSet,
    chars_to_tokens,
    segment_sources,
    tokens_to_chars,
)
from knox.compilation.asset_registry import AssetIdMap, assign_asset_ids
from knox.compilation.base import BaseCompiler
from knox.compilation.flows.assembly import (
    cosine_similarity,
    merge_similar_concepts,
)
from knox.compilation.flows.extraction import (
    batch_texts,
    combine_extractions,
    reduce_extractions,
    run_extraction,
)
from knox.compilation.flows.generation import (
    gen_cache_deps,
    run_generation,
)
from knox.compilation.flows.references import run_reference_extraction
from knox.compilation.graph import GraphCompiler
from knox.core.cache.store import CacheStore
from knox.core.models import ChunkTree, CompiledWiki, MetadataField, ResolvedChunk, WikiField
from knox.core.paths import rel_to_vault
from knox.core.pipeline.assets import AssetStore
from knox.core.pipeline.config import PipelineConfig
from knox.core.pipeline.references import ReferenceStore
from knox.core.templates import get_template_loader
from knox.llm.context_window import ContextWindow, discover_context_window
from knox.logging import get_logger

# Re-export for backward compatibility.
__all__ = [
    "Extraction",
    "ExtractionSet",
    "TwoPhaseCompiler",
    "chars_to_tokens",
    "cosine_similarity",
    "merge_similar_concepts",
    "segment_sources",
    "tokens_to_chars",
]

logger = get_logger(__name__)

_loader = get_template_loader()


class TwoPhaseCompiler(BaseCompiler):
    """N3 two-phase compiler with cross-source merging and adaptive budgets.

    Parameters
    ----------
    config:
        Pipeline configuration including budget and context window settings.
    base:
        Compiler used as a fallback when the LLM is unavailable.
    context_window:
        Pre-discovered context window. When ``None``, discovered independently
        from Ollama or models.dev as a fallback.
    """

    def __init__(
        self,
        config: PipelineConfig,
        base: BaseCompiler | None = None,
        *,
        context_window: ContextWindow | None = None,
    ) -> None:
        self._model = config.llm.default_model or ""
        self._base = base or GraphCompiler(model=self._model)
        self._budget = config.budget
        self._config = config
        self._timeout = config.llm.timeout
        self._cache_store = CacheStore(config.paths.cache_dir)
        self._llm_stats: list[tuple[int, int]] = []
        self._extraction_stats: list[dict[str, Any]] = []
        self._asset_store: AssetStore | None = None
        self._asset_id_maps: dict[str, AssetIdMap] = {}
        self._batch_tracker = BatchFailureTracker(
            ratio=self._config.llm.batch_failure_threshold_ratio,
            min_batches=self._config.llm.batch_failure_threshold_min,
        )

        if context_window is not None:
            cw = context_window
        else:
            if not self._model:
                msg = "LLM model required for TwoPhaseCompiler"
                raise ValueError(msg)
            cw = discover_context_window(
                self._model,
                fallback_max_input=config.llm.fallback_max_input,
                fallback_max_output=config.llm.fallback_max_output,
                context_window_cap=config.llm.context_window_cap,
            )
        self._context_window = cw

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

    def compile(  # noqa: PLR0915
        self,
        chunks: list[ResolvedChunk],
        *,
        trees: list[ChunkTree] | None = None,
        pinned_facts: list[str] | None = None,
    ) -> CompiledWiki:
        """Run the two-phase compile with generation cache.

        When *pinned_facts* is non-empty, each fact is injected into the
        generation prompt as a hard preservation constraint (WiCER /
        CEGAR-style refinement).
        """
        compile_start = time.monotonic()

        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={WikiField.CHUNK_COUNT: 0, MetadataField.TWO_PHASE: True, "llm_calls": 0},
            )

        num_sources = len({str(rc.chunk.source_file or "unknown") for rc in chunks})
        self._report(0, num_sources, "Compiling wiki…")

        # ── Reference extraction (pre-extraction pass) ─────────────────
        # Detect and parse bibliographic references before the main
        # extraction phase so the LLM sees content, not bibliographies.
        ref_store = ReferenceStore(self._config.paths.references_file)
        ref_start = time.monotonic()
        try:
            run_reference_extraction(self, chunks, ref_store)
        except Exception as exc:  # noqa: BLE001 — reference extraction is non-fatal
            logger.warning("TwoPhase: reference extraction failed (non-fatal): %s", exc)
        ref_elapsed = time.monotonic() - ref_start
        if ref_store.total_count() > 0:
            logger.info(
                "TwoPhase: references parsed in %.1fs — %d from %d source(s)",
                ref_elapsed,
                ref_store.total_count(),
                ref_store.source_count(),
            )

        # ── Asset separation (pre-extraction pass) ───────────────────
        # Register tables/figures under content-addressed ids so the
        # extraction input references them by id + summary instead of
        # inlining full (potentially huge) content. See
        # ``.agents/plans/structured-asset-separation.md``.
        self._asset_store = AssetStore(self._config.paths.assets_file)
        self._asset_id_maps = {}
        if trees:
            for t in trees:
                if t.source_file is not None:
                    src_key = rel_to_vault(t.source_file, self._config.paths.base_dir)
                    id_map = assign_asset_ids(t, self._asset_store, source=src_key)
                    if len(id_map) > 0:
                        self._asset_id_maps[src_key] = id_map
            self._asset_store.save()
            if self._asset_store.total_count() > 0:
                logger.info(
                    "TwoPhase: %d structured asset(s) separated across %d source(s)",
                    self._asset_store.total_count(),
                    len(self._asset_id_maps),
                )

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

        gen_deps = gen_cache_deps(self, extractions, pinned_facts=pinned_facts)
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

        self._active_gen_deps = gen_deps
        gen_start = time.monotonic()
        wiki = self.generate(extractions, chunks, trees=trees, pinned_facts=pinned_facts)
        gen_elapsed = time.monotonic() - gen_start

        has_failures = self._batch_tracker.failed > 0
        # When batches failed, keep the checkpoint so a re-run can resume and
        # retry only the failed batches. Don't cache the incomplete wiki.
        if has_failures:
            logger.info(
                "TwoPhase: %d/%d batch(es) failed — checkpoint preserved, wiki not cached. Re-run to retry failed batches only.",
                self._batch_tracker.failed,
                self._batch_tracker.total,
            )
        else:
            delete_gen_checkpoint(self._config, gen_deps)

        if getattr(self, "_extraction_stats", None):
            wiki.metadata["extraction_stats"] = self._extraction_stats

        total_prompt = sum(pt for pt, _ in self._llm_stats)
        total_comp = sum(ct for _, ct in self._llm_stats)
        wiki.metadata["llm_calls"] = len(self._llm_stats)
        wiki.metadata["llm_prompt_tokens"] = total_prompt
        wiki.metadata["llm_completion_tokens"] = total_comp
        wiki.metadata["compile_duration_s"] = round(time.monotonic() - compile_start, 1)

        if self._batch_tracker.should_stop() or self._batch_tracker.failed > 0:
            wiki.metadata["compile_incomplete"] = self._batch_tracker.should_stop()
            wiki.metadata["batch_failures"] = self._batch_tracker.failed
            wiki.metadata["batch_total"] = self._batch_tracker.total

        total_elapsed = time.monotonic() - compile_start
        node_count = len(wiki.nodes)
        logger.info(
            "TwoPhase: compile done — %d source(s) → %d concept(s) in %.1fs (extract %.1fs, generate %.1fs), %d LLM call(s), %d prompt + %d completion tokens",
            num_sources,
            node_count,
            total_elapsed,
            extract_elapsed,
            gen_elapsed,
            len(self._llm_stats),
            total_prompt,
            total_comp,
        )

        if not self._config.no_cache and not has_failures:
            self._cache_store.set("generate", gen_deps, wiki)
        return wiki

    def extract(self, chunks: list[ResolvedChunk], *, trees: list[ChunkTree] | None = None) -> ExtractionSet:
        """Phase 1: run the LLM in extraction mode for each source."""
        return run_extraction(
            self,
            chunks,
            trees=trees,
            asset_id_maps=self._asset_id_maps,
        )

    def generate(
        self,
        extractions: ExtractionSet,
        chunks: list[ResolvedChunk],
        *,
        trees: list[ChunkTree] | None = None,
        pinned_facts: list[str] | None = None,
    ) -> CompiledWiki:
        """Phase 2: merge extractions and generate per-concept pages."""
        return run_generation(self, extractions, chunks, trees=trees, pinned_facts=pinned_facts)

    def _combine_extractions(self, source: str, parts: list[Extraction]) -> Extraction:
        """Combine per-batch extractions into one document-level Extraction."""
        return combine_extractions(self, source, parts)

    def _reduce_extractions(self, source: str, parts: list[Extraction]) -> Extraction:
        """LLM-powered reduce: merge partial extractions into a coherent whole."""
        return reduce_extractions(self, source, parts)

    @staticmethod
    def _batch_texts(texts: list[str], char_budget: int) -> list[str]:
        """Group texts into batches that fit within *char_budget* (chars)."""
        return batch_texts(texts, char_budget)
