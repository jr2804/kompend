"""Pipeline orchestrator for the KNOX 5-tier architecture.

Tier identifiers, tier descriptions, and :class:`PipelineConfig` live in
:mod:`knox.core.pipeline_defs` so they can be shared with the CLI and
other callers without dragging in the orchestrator. This module
re-exports the legacy ``TIER_*`` string constants for backward
compatibility with downstream callers that still import them from
``knox.core.pipeline``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from knox.chunking.hierarchical import HierarchicalChunker
from knox.chunking.summarizer import NodeSummarizer
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.compilation.two_phase import TwoPhaseCompiler
from knox.compilation.wicer import WiCERCompiler
from knox.core.cache.store import CacheStore
from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_TIER_SAMPLE_SIZE
from knox.core.models import (
    Chunk,
    ChunkTree,
    CompiledWiki,
    CrossReferenceResult,
    DocumentElement,
    ElementType,
    MetadataField,
    ParsedDocument,
    ResolvedChunk,
    WikiField,
    WikiNode,
)
from knox.core.paths import VaultPaths
from knox.core.pipeline.config import (
    ALL_TIERS,
    TIER_DESCRIPTIONS,
    TIER_LEVEL,
    Compiler,
    PipelineConfig,
    Tier,
)
from knox.core.pipeline.ingest import parse_cache_name
from knox.core.pipeline.state import TierState, TierStateStore
from knox.core.resolution.entity_registry import EntityRegistry
from knox.core.resolution.xref_checker import CrossReferenceChecker
from knox.core.templates import get_template_loader
from knox.llm.client import LLMClient
from knox.logging import get_logger
from knox.memory.lifecycle import update_memory_for_compile
from knox.output.base import BaseOutputWriter
from knox.preprocessing.dedup import DedupStore
from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.parsers._registry import ParserBuildOptions, build_default_router, get_all_extensions
from knox.preprocessing.parsers.base import ParserRouter
from knox.resolution.base import BaseResolver
from knox.resolution.chain import ResolverChain
from knox.resolution.vlm_enricher import VLMEnricher
from knox.trust import filter_sensitive, filter_wiki

logger = get_logger(__name__)

# Type for progress event callbacks. Parameters:
#   phase   – pipeline phase ("index", "parse", "resolve", "compile", "output")
#   current – current item (0 = phase starting, >0 = progress)
#   total   – total items in this phase
#   detail  – human detail (filename, status, etc.)
ProgressCallback = Callable[[str, int, int, str], None]

# Legacy string-constant re-exports (kept for backward compatibility with
# downstream callers that imported these names from ``knox.core.pipeline``).
# New code should import :class:`Tier` / :class:`Mode` from
# :mod:`knox.core.pipeline_defs` instead.
TIER_INDEX: Tier = Tier.INDEX
TIER_CHUNK: Tier = Tier.CHUNK
TIER_RESOLVE: Tier = Tier.RESOLVE
TIER_COMPILE: Tier = Tier.COMPILE
TIER_OUTPUT: Tier = Tier.OUTPUT

# Backward-compat alias for the previously-private description map.
_TIER_DESCRIPTIONS = TIER_DESCRIPTIONS


class WikiPipeline:
    """Orchestrates the 5-tier KNOX pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        downloader: BaseDownloader | None = None,
        router: ParserRouter | None = None,
        chunker: HierarchicalChunker | None = None,
        resolver: BaseResolver | None = None,
        compiler: BaseCompiler | None = None,
        output_writer: BaseOutputWriter | None = None,
        state_store: TierStateStore | None = None,
        llm_client: LLMClient | None = None,
        summarizer: NodeSummarizer | None = None,
    ) -> None:
        self.config = config
        self.downloader = downloader
        self.router = router
        self.chunker = chunker
        self.resolver = resolver
        self.compiler = compiler
        self.output_writer = output_writer
        self.llm_client = llm_client
        self.summarizer = summarizer
        self._base_dir = config.resolved_base_dir()
        self.state_store = state_store or TierStateStore(self._default_state_path(), self._base_dir)
        self._progress_callback: ProgressCallback | None = None
        self._cache_store = CacheStore(self._cache_dir())
        self._tier_timings: dict[str, float] = {}
        self.last_chunk_count: int = 0
        self.xref_result: CrossReferenceResult | None = None

    @property
    def state(self) -> TierState:
        """Lazy-loaded persistent state — call sites should treat as read-only."""
        if not hasattr(self, "_state"):
            self._state = self.state_store.load()
        return self._state

    def save_state(self) -> None:
        """Persist any in-memory state mutations."""
        if hasattr(self, "_state"):
            self.state_store.save(self._state)

    def start_level(self) -> int:
        """Numeric level of the first tier to run."""
        return TIER_LEVEL[self.config.start_at_tier]

    def stop_level(self) -> int:
        """Numeric level of the last tier to run."""
        return TIER_LEVEL[self.config.stop_at_tier]

    def run(self, progress_callback: ProgressCallback | None = None) -> CompiledWiki:  # noqa: PLR0915
        """Execute the full pipeline from sources to compiled wiki.

        Parameters
        ----------
        progress_callback:
            Optional ``(phase, current, total, detail)`` callable.
            Purely informational — core logic never depends on it.
        """
        self._progress_callback = progress_callback

        if self.config.notes_file and self.config.notes_file.exists():
            notes_content = self.config.notes_file.read_text(encoding="utf-8")
            get_template_loader().set_global_context(user_notes=notes_content)
            logger.info("Loaded user notes from %s", self.config.notes_file)
        else:
            get_template_loader().clear_global_context()

        logger.info(
            "Starting KNOX pipeline (tiers %s→%s): %s",
            self.config.start_at_tier.value,
            self.config.stop_at_tier.value,
            self.config.source_dir,
        )

        # Auto-resume: when start_at_tier is at default (index) and a previous
        # run exists, jump to the tier where progress stopped. Use --force to
        # re-run from scratch, or --start-at-tier index to override.
        if self.config.start_at_tier == Tier.INDEX and not self.config.force:
            auto_tier = self._detect_resume_tier()
            if auto_tier is not None:
                min_tier = min(e.current_tier for e in self.state.entries.values())
                logger.info(
                    "Auto-resume: previous run detected (sources at tier %d). Resuming at tier '%s'. Use --start-at-tier index or --force for full re-run.",
                    min_tier,
                    auto_tier.value,
                )
                self.config.start_at_tier = auto_tier

        self._validate_prerequisites()

        # --force: invalidate all caches before starting.
        if self.config.force:
            logger.info("Force mode enabled — invalidating all caches")
            self._cache_store.invalidate()

        all_trees: list[ChunkTree] = []
        resolved: list[Any] = []
        wiki = CompiledWiki(nodes=[], edges=[], toc=[], metadata={})
        self._tier_timings.clear()

        if self._should_run(Tier.INDEX):
            self._emit("index", 0, 1, "starting")
            _t0 = time.monotonic()
            self.run_tier(TIER_INDEX)
            self._tier_timings["index"] = time.monotonic() - _t0
            self._emit("index", 1, 1, "done")
        if self._should_run(Tier.CHUNK):
            _t0 = time.monotonic()
            all_trees = self._parse_and_chunk()
            self.last_chunk_count = sum(1 for t in all_trees for _ in t.leaves())
            self._tier_timings["chunk"] = time.monotonic() - _t0
            self._record_tier_state(1)
        if self._should_run(Tier.RESOLVE):
            # When T1 was skipped (start_at_tier >= resolve), warm-load
            # chunks from the T1 cache so the resolver has input.
            if not all_trees:
                all_trees = self._parse_and_chunk()
            leaf_count = sum(1 for t in all_trees for _ in t.leaves())
            self._emit("resolve", 0, leaf_count, "starting")
            _t0 = time.monotonic()
            resolved = self._resolve_chunks(all_trees)
            self._tier_timings["resolve"] = time.monotonic() - _t0
            self._record_tier_state(2)
            self._emit("resolve", len(resolved), len(resolved), "done")
        if self._should_run(Tier.COMPILE):
            # When T2 was skipped (start_at_tier >= compile), warm-load
            # chunks and resolve them from cache.
            if not resolved:
                all_trees = self._parse_and_chunk()
                resolved = self._resolve_chunks(all_trees)
            self._emit("compile", 0, 1, "starting")
            _t0 = time.monotonic()
            if self.compiler:
                # Attach the progress hook so the compiler reports real,
                # countable work units (source extractions, WiCER steps) to
                # a determinate bar instead of an indeterminate spinner.
                self.compiler.on_progress = lambda c, t, d: self._emit("compile", c, t, d)
                try:
                    wiki = self.compiler.compile(resolved, trees=all_trees)
                finally:
                    self.compiler.on_progress = None
            if self.config.privacy_filter:
                wiki, _ = filter_wiki(wiki)
            self._tier_timings["compile"] = time.monotonic() - _t0
            self._record_tier_state(3)
            self._emit("compile", 1, 1, "done")
        else:
            wiki = self._load_compiled_wiki() or wiki
        if self._should_run(Tier.OUTPUT) and self.output_writer:
            self._emit("output", 0, 1, "starting")
            if self.config.privacy_filter:
                wiki, _ = filter_wiki(wiki)
            _t0 = time.monotonic()
            self.output_writer.write(wiki, self.config.output_dir)
            self._tier_timings["output"] = time.monotonic() - _t0
            self._record_tier_state(4)
            self._emit("output", 1, 1, "done")

        wiki.metadata[str(WikiField.TIER_TIMINGS)] = {k: round(v, 2) for k, v in self._tier_timings.items()}
        if self.llm_client is not None:
            wiki.metadata[str(WikiField.LLM_USAGE)] = self.llm_client.usage.to_dict()
            wiki.metadata[str(WikiField.LLM_MODEL)] = self.config.llm.default_model
            report = self.llm_client.degradation_report()
            wiki.metadata[str(WikiField.LLM_DEGRADATION)] = report
            if report["failed_calls"] > 0:
                rate_pct = report["degradation_rate"] * 100
                logger.warning(
                    "LLM degradation: %d/%d calls failed (%.1f%%) — output quality may be reduced",
                    report["failed_calls"],
                    report["total_calls"],
                    rate_pct,
                )

        self.save_state()
        self._progress_callback = None
        get_template_loader().clear_global_context()

        if self.config.memory_enabled and wiki.nodes:
            base_dir = self.config.resolved_base_dir()
            update_memory_for_compile(
                wiki,
                base_dir,
                half_life_days=self.config.memory_half_life_days,
                decay_threshold=self.config.memory_decay_threshold,
            )

        logger.info("KNOX pipeline complete: %s -> %s (%d nodes)", self.config.source_dir, self.config.output_dir, len(wiki.nodes))
        return wiki

    def run_tier(self, tier: str) -> None:
        """Run a single tier by name.

        Valid names: ``index``, ``chunk``, ``resolve``, ``compile``, ``output``.
        Tiers are cumulative: running ``compile`` first ensures ``chunk``
        and ``index`` state is up to date.
        """
        if tier not in TIER_LEVEL:
            msg = f"Unknown tier {tier!r}. Valid: {ALL_TIERS}"
            raise ValueError(msg)

        target_level = TIER_LEVEL[Tier(tier)]

        if target_level >= TIER_LEVEL[TIER_INDEX]:
            self._run_index_tier()
        if target_level >= TIER_LEVEL[TIER_CHUNK]:
            self._parse_and_chunk()
        if target_level >= TIER_LEVEL[TIER_COMPILE]:
            self._run_compile_tier()
        if target_level >= TIER_LEVEL[TIER_OUTPUT]:
            self._run_output_tier()

        self.save_state()

    def iter_sources(self) -> Any:
        """Yield source files to process.

        Always scans ``config.source_dir`` for supported file types.
        The ``config.input_files`` list is only used by the ingest
        command to copy files into the source directory — it does
        *not* scope which files the pipeline processes.

        Excludes the URL list file — it is processed via
        ``--url-file``, not a document to parse.
        """
        extensions = get_all_extensions()
        for f in self.config.source_dir.rglob("*"):
            if f.suffix.lower() in extensions:
                yield f

    def list_tiers(self) -> list[dict[str, Any]]:
        """Return the available tiers with their current persisted state.

        Used by ``knox compile --list-tiers`` to show progress and by
        tests to assert the tier contract. Each entry contains the
        tier name, level, the current persisted max level per source,
        and a sample of recent sources.
        """
        state = self.state
        distribution = state.tier_distribution()
        sample_limit = DEFAULTS.get(DISPLAY_TIER_SAMPLE_SIZE, 5)
        sample = list(state.entries.values())[:sample_limit]
        return [
            {
                "tier": name,
                "level": TIER_LEVEL[name],
                "sources_at_or_above": sum(1 for e in state.entries.values() if e.current_tier >= TIER_LEVEL[name]),
                "description": _TIER_DESCRIPTIONS[name],
            }
            for name in ALL_TIERS
        ] + [
            {
                "tier": "_distribution",
                "level": -1,
                "sources_at_or_above": 0,
                "description": f"tier distribution: {distribution}",
            },
            {
                "tier": "_sample",
                "level": -2,
                "sources_at_or_above": 0,
                "description": f"sample sources: {[e.stem for e in sample]}",
            },
        ]

    def _rel_path(self, path: Path) -> str:
        """Convert absolute path to relative POSIX string for serialization."""
        try:
            return path.relative_to(self._base_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def _abs_path(self, path_str: str) -> Path:
        """Reconstruct absolute path from potentially relative string."""
        p = Path(path_str) if path_str else Path()
        if p != Path() and not p.is_absolute():
            return self._base_dir / p
        return p

    def _normalize_metadata_paths(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert known absolute path values in metadata to relative."""
        if MetadataField.SOURCE_FILE in metadata:
            meta = dict(metadata)
            meta[MetadataField.SOURCE_FILE] = self._rel_path(Path(meta[MetadataField.SOURCE_FILE]))
            return meta
        return metadata

    def _detect_resume_tier(self) -> Tier | None:
        """Check compile_state.json for previous progress.

        Returns the tier to resume at, or ``None`` if no previous run
        exists or all sources are at tier 0 (no meaningful progress).
        """
        state = self.state
        if not state.entries:
            return None
        min_tier = min(e.current_tier for e in state.entries.values())
        if min_tier < 1:
            return None
        resume_level = min_tier + 1
        for tier, level in TIER_LEVEL.items():
            if level == resume_level:
                return tier
        return None

    def _should_run(self, tier: Tier) -> bool:
        """Check whether ``tier`` falls within [start, stop]."""
        level = TIER_LEVEL[tier]
        return self.start_level() <= level <= self.stop_level()

    def _validate_prerequisites(self) -> None:
        """Ensure previous tiers are cached when starting at a later one."""
        start = self.start_level()
        if start == 0:
            return

        state = self.state
        required_level = start - 1
        if not state.entries:
            msg = f"Cannot start at tier '{self.config.start_at_tier.value}' — no previous run found. Run 'knox ingest' or 'knox compile' first."
            raise ValueError(msg)

        ready = [e for e in state.entries.values() if e.current_tier >= required_level]
        if not ready:
            msg = (
                f"Cannot start at tier '{self.config.start_at_tier.value}' "
                f"— no sources have completed tier {required_level}. "
                f"Run 'knox ingest' or 'knox compile' first."
            )
            raise ValueError(msg)

    def _emit(self, phase: str, current: int, total: int, detail: str = "") -> None:
        """Emit a progress event if a callback is registered.

        A progress callback is arbitrary user code; it may raise anything. Per
        src/knox/AGENTS.md this is a best-effort enrichment site: a buggy
        callback must NOT abort the pipeline, so the broad catch is intentional.
        The failure is logged at WARNING (not swallowed) so it stays visible.
        """
        if self._progress_callback is not None:
            try:
                self._progress_callback(phase, current, total, detail)
            except Exception:  # best-effort: callback is user code, must not abort pipeline
                logger.warning("Progress callback raised in phase %s — continuing", phase, exc_info=True)

    def _default_state_path(self) -> Path:
        """Resolve the persistent state file path."""
        if self.config.tier_state_file is not None:
            return self.config.tier_state_file
        return self.config.paths.compile_state_path

    def _dedup_store_path(self) -> Path:
        """Resolve the dedup fingerprint store path."""
        base_dir = self.config.resolved_base_dir()
        return VaultPaths(base_dir).dedup_dir / "fingerprints.json"

    def _record_tier_state(self, tier: int) -> None:
        """Update current_tier for every known source file."""
        state = self.state
        for source_file in self.iter_sources():
            stem = source_file.stem
            try:
                content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            except OSError:
                logger.warning("Could not read %s for tier-state hashing — skipping", source_file, exc_info=True)
                continue
            self.state_store.record(state, stem, Path(self._rel_path(source_file)), content_hash, tier=tier)
        self.state_store.save(state)

    def _run_index_tier(self) -> None:
        """Tier 0: download online sources and record state for local files."""
        if self.downloader and self.config.online_sources:
            for url in self.config.online_sources:
                self.downloader.download(url, self.config.source_dir)

    def _run_compile_tier(self) -> None:
        """Tier 2+3: resolve leaf chunks from ChunkTree and compile into CompiledWiki."""
        trees = self._parse_and_chunk()
        resolved = self._resolve_chunks(trees)
        if not self.compiler:
            return
        wiki = self.compiler.compile(resolved, trees=trees)
        self._build_wiki_hierarchy(wiki, trees)
        if self.config.privacy_filter:
            wiki, _ = filter_wiki(wiki)
        self._save_compiled_wiki(wiki)

    def _run_output_tier(self) -> None:
        """Tier 4: write the compiled wiki to the output directory."""
        wiki = self._load_compiled_wiki()
        if wiki is None or not self.output_writer:
            return
        self.output_writer.write(wiki, self.config.output_dir)

    def _compiled_wiki_path(self) -> Path:
        return self.config.paths.compiled_wiki_path

    def _save_compiled_wiki(self, wiki: CompiledWiki) -> None:

        path = self._compiled_wiki_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            WikiField.NODES: [
                {
                    "id": n.id,
                    "title": n.title,
                    "content": n.content,
                    "node_type": n.node_type,
                    "source_chunks": n.source_chunks or [],
                    "metadata": self._normalize_metadata_paths(n.metadata),
                }
                for n in wiki.nodes
            ],
            WikiField.EDGES: wiki.edges,
            WikiField.TOC: wiki.toc,
            WikiField.QUALITY_SCORE: wiki.quality_score,
            WikiField.ITERATION: wiki.iteration,
            "metadata": wiki.metadata,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _load_compiled_wiki(self) -> CompiledWiki | None:
        path = self._compiled_wiki_path()
        if not path.exists():
            return None
        # Corrupt snapshot is a bug — let it raise so the real cause surfaces
        # instead of silently recompiling from scratch.
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = [WikiNode(**n) for n in data.get(WikiField.NODES, [])]
        return CompiledWiki(
            nodes=nodes,
            edges=data.get(WikiField.EDGES, []),
            toc=data.get(WikiField.TOC, []),
            quality_score=data.get(WikiField.QUALITY_SCORE, 0.0),
            iteration=data.get(WikiField.ITERATION, 0),
            metadata=data.get(WikiField.METADATA, {}),
        )

    def _load_dedup_state(self) -> tuple[set[str], dict[str, dict[str, Any]]]:
        """Best-effort load of dedup store.

        Returns ``(dedup_skip, dedup_similar)``.
        """
        dedup_skip: set[str] = set()
        dedup_similar: dict[str, dict[str, Any]] = {}
        try:
            ds = DedupStore(self._dedup_store_path())
            ds.load()
            if ds.is_fresh():
                for slug in ds.slugs:
                    entry = ds.get(slug, {})
                    if entry.get("verdict") == "identical":
                        dedup_skip.add(slug)
                    elif entry.get("verdict") == "similar":
                        dedup_similar[slug] = entry
        except Exception:
            logger.debug("Dedup store not available — skipping dedup check", exc_info=True)
        return dedup_skip, dedup_similar

    def _parse_and_chunk(self) -> list[Any]:  # noqa: PLR0915
        """Parse documents and chunk them, with optional caching and parallelism.

        When tier state is available, sources that already match the
        persisted hash and tier are skipped to keep the per-tier
        execution incremental.

        When content-level duplicate detection has run during a prior ingest,
        the DedupStore is checked. Sources flagged as ``identical`` are
        silently skipped. Sources flagged as ``similar`` get a duplication
        warning attached to their metadata.
        """
        source_files = list(self.iter_sources())
        if not source_files:
            return []

        self._emit("parse", 0, len(source_files), "starting")

        dedup_skip, dedup_similar = self._load_dedup_state()
        if dedup_skip:
            logger.info("Dedup: %d source(s) flagged as identical — will skip", len(dedup_skip))
            source_files = [f for f in source_files if f.stem not in dedup_skip]
            if not source_files:
                self._emit("parse", 0, 0, "all skipped by dedup")
                return []

        workers = max(1, self.config.workers)
        use_cache = not self.config.no_cache

        state = self.state

        def _process_one(source_file: Path) -> ChunkTree | None:
            """Parse a single file (cache-aware) and chunk it."""
            stem = source_file.stem
            content_hash = ""

            if use_cache and not self.config.force:
                content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

            parsed = self._load_cache(source_file) if use_cache else None

            if parsed is None and use_cache and not self.config.force:
                cached_entry = state.entries.get(stem)
                if cached_entry and cached_entry.content_hash == content_hash and cached_entry.current_tier >= 1:
                    parsed = self._load_cache(source_file)

            if parsed is None:
                if self.router is None:
                    return None
                try:
                    selected = self.router.route(source_file)
                    if selected is None:
                        logger.warning("No parser backend for %s, skipping", source_file)
                        return None
                    parsed = selected.parse(source_file)
                    _ = len(parsed.elements)
                except Exception:
                    logger.exception("Failed to parse %s", source_file)
                    return None
                if use_cache:
                    self._save_cache(source_file, parsed)
                    if not content_hash:
                        content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

            # ── Normalize source_file to relative POSIX on parsed doc ────
            # All downstream code (chunking, extraction, generation, wiki
            # metadata, LLM prompts) gets clean portable paths.
            parsed.source_file = Path(self._rel_path(parsed.source_file))
            for el in parsed.elements:
                if el.source_file is not None:
                    el.source_file = Path(self._rel_path(el.source_file))

            if self.config.privacy_filter:
                self._apply_privacy_filter(parsed.elements)

            if not self.chunker:
                return None

            # ── Attach dedup similarity warning if applicable ────────
            similar_entry = dedup_similar.get(stem)
            if similar_entry:
                warning = (
                    f"> [!warning] Possible duplicate\n"
                    f"> This document is similar to **{similar_entry.get('closest_match', 'unknown')}** "
                    f"> (score={similar_entry.get('closest_score', 0):.2f}).\n"
                )
                if not parsed.metadata:
                    parsed.metadata = {}
                parsed.metadata["_dedup_warning"] = warning

            # 2a. Chunk cache lookup (Tier 1).
            if use_cache and content_hash:
                chunk_deps = {
                    "source": content_hash,
                    "chunk_effective_max": str(getattr(self.chunker, "_effective_max", 0)),
                }
                cached = self._cache_store.get("chunk", chunk_deps)
                if cached is not None:
                    if content_hash:
                        self.state_store.record(state, stem, Path(self._rel_path(source_file)), content_hash, tier=1)
                    tree = ChunkTree.from_dict(cached)
                    self._apply_tags_to_tree(tree, source_file)
                    return tree

            # 2b. Chunk and cache.
            tree = self.chunker.chunk(parsed)
            self._apply_tags_to_tree(tree, source_file)

            if content_hash:
                self.state_store.record(state, stem, Path(self._rel_path(source_file)), content_hash, tier=1)
                chunk_deps = {
                    "source": content_hash,
                    "chunk_effective_max": str(getattr(self.chunker, "_effective_max", 0)),
                }
                self._cache_store.set("chunk", chunk_deps, tree.to_dict())
            return tree

        trees: list[ChunkTree] = []
        if workers > 1 and len(source_files) > 1:
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_one, sf): sf for sf in source_files}
                for future in as_completed(futures):
                    sf = futures[future]
                    try:
                        tree = future.result()
                        if tree is not None:
                            trees.append(tree)
                    except Exception:
                        logger.exception("Worker failed for %s", sf)
                    finally:
                        completed += 1
                        self._emit("parse", completed, len(source_files), sf.name)
        else:
            for idx, sf in enumerate(source_files):
                tree = _process_one(sf)
                if tree is not None:
                    trees.append(tree)
                self._emit("parse", idx + 1, len(source_files), sf.name)

        self._emit("parse", len(source_files), len(source_files), "done")

        if self.summarizer is not None:
            for tree in trees:
                self.summarizer.summarize(tree)

        return trees

    def _cache_dir(self) -> Path:
        """Resolve the cache directory."""
        if self.config.cache_dir:
            return self.config.cache_dir
        return self.config.paths.cache_dir

    def _load_cache(self, source_file: Path) -> ParsedDocument | None:
        """Load a cached ParsedDocument if available."""
        cache_path = self._cache_dir() / f"{self._cache_key(source_file)}.json"
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            elements = [
                DocumentElement(
                    type=ElementType(el["type"]),
                    content=el["content"],
                    page=el.get("page"),
                    reading_order=el.get(MetadataField.READING_ORDER),
                    heading_level=el.get("heading_level"),
                    metadata=el.get("metadata", {}),
                )
                for el in data.get("elements", [])
            ]
            return ParsedDocument(
                source_file=self._abs_path(data.get(MetadataField.SOURCE_FILE) or data.get(MetadataField.SOURCE_PATH, "")),
                elements=elements,
                title=data.get("title", ""),
                author=data.get("author", ""),
                language=data.get("language", "en"),
                page_count=data.get("page_count", 0),
                metadata=data.get(WikiField.METADATA, {}),
            )
        except (json.JSONDecodeError, KeyError, OSError):
            # Corrupt/unreadable cache entry = miss: re-parse. Warn so it's visible.
            logger.warning("Corrupt parse cache at %s — reparsing", cache_path, exc_info=True)
            return None

    def _save_cache(self, source_file: Path, doc: ParsedDocument) -> None:
        """Write a ParsedDocument to cache."""
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self._cache_key(source_file)}.json"
        try:
            data = {
                str(MetadataField.SOURCE_FILE): self._rel_path(doc.source_file),
                "title": doc.title,
                "author": doc.author,
                "language": doc.language,
                "page_count": doc.page_count,
                "metadata": doc.metadata,
                "elements": [
                    {
                        "type": str(el.type),
                        "content": str(el.content) if not isinstance(el.content, list) else el.content,
                        "page": el.page,
                        MetadataField.READING_ORDER: el.reading_order,
                        "heading_level": el.heading_level,
                        "metadata": el.metadata,
                    }
                    for el in doc.elements
                ],
            }
            cache_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception:
            logger.debug("Cache write failed for %s — ignoring", cache_path, exc_info=True)

    def _resolve_chunks(self, trees: list[ChunkTree]) -> list[Any]:
        """Resolve leaf chunks from trees and build entity registry + xref.

        VLM enrichment (image captions) runs per-leaf when configured.
        Entity registry and cross-reference checks run programmatically
        (no LLM) across the full tree set.

        Resolved chunks are cached under the ``"resolve"`` layer so VLM
        captioning survives a re-run after a mid-pipeline failure.
        """
        registry = EntityRegistry()
        for tree in trees:
            registry.scan_tree(tree)
        checker = CrossReferenceChecker(registry)
        self.xref_result = checker.check_all(trees)
        logger.info(
            "Cross-reference check: %d entities, %d issues",
            self.xref_result.entity_count,
            len(self.xref_result.issues),
        )

        if not self.resolver:
            return [node.chunk for t in trees for node in t.leaves() if node.chunk]

        leaf_chunks = [node.chunk for t in trees for node in t.leaves() if node.chunk]
        resolved: list[Any] = []
        total = len(leaf_chunks)

        # Two-pass resolution: non-image chunks first (no VLM), then all
        # image chunks batched together. This keeps the VLM loaded for one
        # contiguous burst instead of reloading it for scattered image
        # chunks across the 1000+ chunk stream (Ollama keep-alive churn).
        non_image = [c for c in leaf_chunks if ElementType.IMAGE not in c.element_types]
        image_chunks = [c for c in leaf_chunks if ElementType.IMAGE in c.element_types]
        if image_chunks:
            logger.info(
                "Resolving %d image chunks in batch (VLM), %d non-image first",
                len(image_chunks),
                len(non_image),
            )

        vlm_model = self.config.llm.vlm_model if self.config.llm.vlm_model else ""

        def _resolve_one(chunk: Chunk) -> ResolvedChunk:
            deps = {"chunk_id": chunk.id, "vlm_model": vlm_model}
            cached = self._cache_store.get("resolve", deps)
            if cached is not None:
                return ResolvedChunk(
                    chunk=chunk,
                    entities=cached.get("entities", []),
                    relations=cached.get("relations", []),
                    resolved_references=cached.get("resolved_references", []),
                    metadata=cached.get("metadata", {}),
                )
            result = self.resolver.resolve(chunk)
            self._cache_store.set(
                "resolve",
                deps,
                {
                    "entities": result.entities,
                    "relations": result.relations,
                    "resolved_references": result.resolved_references,
                    "metadata": result.metadata,
                },
            )
            return result

        done = 0
        for i, chunk in enumerate(non_image):
            resolved.append(_resolve_one(chunk))
            done = i + 1
            self._emit("resolve", done, total, "")
        if image_chunks:
            # Signal a distinct VLM sub-phase so the CLI bar switches to
            # "Captioning images" — makes the (slow) VLM run visible instead
            # of looking like a stalled resolve progress bar.
            self._emit("resolve", 0, len(image_chunks), "vlm")
        for i, chunk in enumerate(image_chunks):
            resolved.append(_resolve_one(chunk))
            self._emit("resolve", i + 1, len(image_chunks), "captioning")
        return resolved

    def _apply_tags_to_tree(self, tree: ChunkTree, source_file: Path) -> None:
        """Attach directory-based tags to every leaf chunk in *tree*."""
        tags = self._extract_source_tags(source_file, self.config.source_dir)
        if tags:
            for node in tree.leaves():
                if node.chunk:
                    node.chunk.metadata["tags"] = tags

    @staticmethod
    def _extract_source_tags(source_file: Path, source_dir: Path) -> list[str]:
        """Extract directory names as tags from a source file's path within source_dir.

        ``sources/2024/conferences/iclr/paper.pdf`` → ``['2024', 'conferences', 'iclr']``

        Returns an empty list when the file is not inside source_dir or sits
        at the top level (no parent directories).
        """
        try:
            relative = source_file.relative_to(source_dir)
        except ValueError:
            return []
        parts = list(relative.parts)
        if len(parts) <= 1:
            return []
        return parts[:-1]

    @staticmethod
    def _apply_privacy_filter(elements: list[Any]) -> None:
        """Apply privacy filter in-place to document elements."""
        for el in elements:
            if isinstance(el.content, str):
                el.content, _ = filter_sensitive(el.content)
            elif isinstance(el.content, list):
                el.content = [filter_sensitive(item)[0] if isinstance(item, str) else item for item in el.content]

    @staticmethod
    def _build_wiki_hierarchy(wiki: CompiledWiki, trees: list[ChunkTree]) -> None:
        """Populate WikiNode hierarchy fields from ChunkTree data.

        Maps ChunkTree internal node structure onto WikiNodes by matching
        heading paths. Sets parent_id, children_ids, depth, and heading_path
        on each WikiNode.
        # ponytail: matching by (heading_context, source_file) from metadata
        # is fragile — resolver/VLM may use different keys. Silent skip on
        # mismatch, no warning. Fix by having ChunkTree expose a stable
        # source_key that all tiers agree on.
        """
        if not wiki.nodes or not trees:
            return

        tree_map: dict[str, ChunkTree] = {}
        for t in trees:
            if t.source_file is not None:
                tree_map[str(t.source_file)] = t

        node_by_heading: dict[tuple[str, str | None], WikiNode] = {}
        for node in wiki.nodes:
            heading = node.metadata.get(MetadataField.HEADING_CONTEXT, "_root")
            source = node.metadata.get(MetadataField.SOURCE_FILE)
            node_by_heading[(heading, source)] = node

        def _link_parent(
            node: WikiNode,
            chunk_node: Any,
            tree: ChunkTree,
            source: str,
            node_by_heading: dict[tuple[str, str | None], WikiNode],
        ) -> None:
            """Link *node* to its parent based on chunk tree hierarchy."""
            if not chunk_node.parent_id:
                return
            parent_chunk = tree.nodes.get(chunk_node.parent_id)
            if not parent_chunk or not parent_chunk.chunk:
                return
            parent_heading = parent_chunk.chunk.heading_context or "_root"
            parent_node = node_by_heading.get((parent_heading, source))
            if parent_node is None:
                return
            node.parent_id = parent_node.id
            if node.id not in parent_node.children_ids:
                parent_node.children_ids.append(node.id)

        for node in wiki.nodes:
            heading = node.metadata.get(MetadataField.HEADING_CONTEXT, "_root")
            source = node.metadata.get(MetadataField.SOURCE_FILE)
            if source is None:
                continue
            tree = tree_map.get(source)
            if tree is None:
                continue

            for chunk_node in tree.walk():
                if chunk_node.chunk is None:
                    continue
                if (chunk_node.chunk.heading_context or "_root") != heading:
                    continue
                node.depth = chunk_node.depth
                node.heading_path = chunk_node.heading_path
                _link_parent(node, chunk_node, tree, source, node_by_heading)
                break

    @staticmethod
    def _cache_key(source_file: Path) -> str:
        """Content-discriminated cache key — collision-proof across same-stem files."""
        return parse_cache_name(source_file).removesuffix(".json")


def build_llm_client(config: PipelineConfig) -> LLMClient:
    """Construct an :class:`LLMClient` from a :class:`PipelineConfig`.

    The model is constructed
    from ``llm.default_model``. The fallback chain is
    taken verbatim from ``config.llm.fallback_chain`` and the retry
    budget from ``config.llm.fallback_max_retries``.
    """
    default_model = config.llm.default_model

    return LLMClient(
        default_model=default_model,
        data_format=config.llm.data_format,
        fallback_chain=list(config.llm.fallback_chain),
        max_retries_per_model=config.llm.fallback_max_retries,
        timeout=config.llm.timeout,
        _fallback_max_input=config.llm.fallback_max_input,
        _fallback_max_output=config.llm.fallback_max_output,
        _num_ctx_min=config.llm.num_ctx_min,
        _num_ctx_max=config.llm.num_ctx_max,
        _context_window_cap=config.llm.context_window_cap,
        _api_base=config.llm.api_base,
        _api_key=config.llm.api_key,
    )


def build_pipeline(
    config: PipelineConfig,
) -> WikiPipeline:
    """Shared factory used by both CLI and MCP.

    Constructs a WikiPipeline with default components:
    - Parser chain selected by ``config.parser`` (comma-separated backend names;
      default ``"docling"``; first matching backend per file wins)
    - HierarchicalChunker
    - Compiler selected by ``config.compiler``:
      - ``"graph"``: TwoPhaseCompiler wrapping GraphCompiler (default)
      - ``"wicer"``: WiCERCompiler wrapping TwoPhaseCompiler (so the
        CEGAR loop refines real synthesized pages, not raw chunk dumps)
    - Output writer selected by ``config.compile_target``

    Parameters
    ----------
    config:
        Pipeline configuration.
    """
    writer_cls = BaseOutputWriter.get(config.output.format)
    if writer_cls is None:
        available = ", ".join(BaseOutputWriter.available_formats())
        msg = f"Unknown output format {config.output.format!r}. Available: {available}"
        raise ValueError(msg)

    router = build_default_router(
        ParserBuildOptions(
            images_dir=config.paths.images_dir,
            image_mode=config.image_mode,
            images_scale=config.images_scale,
            pdf_config=config.pdf,
        ),
        names=config.parser,
    )

    # T0 (index) and T1 (parse) run without an LLM — `knox ingest`
    # stops here. T2 (chunk) and beyond need the LLM for context-window
    # discovery and adaptive chunk sizing.
    needs_llm = TIER_LEVEL[config.stop_at_tier] >= TIER_LEVEL[Tier.CHUNK]

    resolver: BaseResolver | None = None
    base_compiler: BaseCompiler | None = None
    chunker_context_window = None
    if needs_llm:
        llm_client = build_llm_client(config)
        chunker_context_window = llm_client.context_window
        resolver = _build_resolver_chain(config, llm_client)

        if config.compiler == Compiler.WICER:
            # WiCER refines through its base compiler: each iteration calls
            # base.compile(..., pinned_facts=F_cumulative). The base must
            # therefore be synthesis-capable (extract → generate) so the
            # pin set is honored in the generation prompt. Wrap TwoPhase
            # — not raw GraphCompiler, which only rearranges chunk text
            # and would silently drop the pin set.
            base_compiler = WiCERCompiler(
                config=config,
                base_compiler=TwoPhaseCompiler(
                    config=config,
                    base=GraphCompiler(
                        model=config.llm.default_model,
                        min_content_length=config.budget.min_content_length,
                        min_concept_sources=config.budget.min_concept_sources,
                        max_concept_pages=config.budget.max_concept_pages,
                    ),
                    llm_client=llm_client,
                ),
                llm_client=llm_client,
            )
        elif config.compiler == Compiler.GRAPH:
            base_compiler = TwoPhaseCompiler(
                config=config,
                base=GraphCompiler(
                    model=config.llm.default_model,
                    min_content_length=config.budget.min_content_length,
                    min_concept_sources=config.budget.min_concept_sources,
                    max_concept_pages=config.budget.max_concept_pages,
                ),
                llm_client=llm_client,
            )
        else:
            msg = f"Unknown compiler: {config.compiler!r}"
            raise ValueError(msg)

    return WikiPipeline(
        config=config,
        router=router,
        chunker=HierarchicalChunker(
            context_window=chunker_context_window,
            chunk_size_min=config.budget.chunk_size_min,
            chunk_size_max=config.budget.chunk_size_max,
            summary_chunk_threshold=config.budget.summary_chunk_threshold,
        ),
        resolver=resolver,
        compiler=base_compiler,
        output_writer=writer_cls(
            flavor=config.output.flavor,
            frontmatter_defaults=config.output.frontmatter or {},
            rumdl_config=config.output.rumdl_config,
            max_related_links=config.output.max_related_links,
        ),
        llm_client=llm_client if needs_llm else None,
        summarizer=NodeSummarizer(
            llm_client=llm_client if needs_llm else None,
            cache_store=CacheStore(config.paths.cache_dir),
            max_content_chars=config.budget.summary_max_content_chars,
            target_chars=config.budget.summary_target_chars,
            skip_single_child=config.budget.summary_skip_single_child,
        ),
    )


def _build_resolver_chain(config: PipelineConfig, llm_client: LLMClient) -> BaseResolver:
    """Construct the resolver chain.

    When a VLM model is configured, a :class:`VLMEnricher` enriches
    image chunks with generated captions.
    """
    resolvers: list[BaseResolver] = []

    if config.llm.vlm_model:
        try:
            vlm_client = LLMClient(
                default_model=config.llm.vlm_model,
                fallback_chain=[],
                max_retries_per_model=0,
                timeout=config.llm.timeout,
                _fallback_max_input=config.llm.fallback_max_input,
                _fallback_max_output=config.llm.fallback_max_output,
                _num_ctx_min=config.llm.num_ctx_min,
                _num_ctx_max=config.llm.num_ctx_max,
                _context_window_cap=config.llm.context_window_cap,
                _api_base=config.llm.api_base,
                _api_key=config.llm.api_key,
            )
            vlm_client.usage = llm_client.usage
            resolvers.append(
                VLMEnricher(
                    client=vlm_client,
                    min_image_bytes=config.vlm_min_image_bytes,
                    min_image_dimension=config.vlm_min_image_dimension,
                )
            )
            logger.debug("VLMEnricher added to resolver chain (model=%s)", config.llm.vlm_model)
        except Exception as exc:
            logger.debug("VLMEnricher not available: %s", exc, exc_info=True)

    if not resolvers:
        logger.info("No resolvers configured — chunks will pass through without resolution")

    return ResolverChain(resolvers)
