"""Pipeline orchestrator — WikiPipeline class.

This module exports :class:`WikiPipeline` and legacy re-exports.
Tier-specific implementation lives in sibling ``_index``, ``_parse``,
``_resolve``, and ``_compile`` modules.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from knox.chunking.hierarchical import HierarchicalChunker
from knox.chunking.summarizer import NodeSummarizer
from knox.compilation.base import BaseCompiler
from knox.core.cache.store import CacheStore
from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_TIER_SAMPLE_SIZE
from knox.core.models import (
    CompiledWiki,
    MetadataField,
    WikiField,
)
from knox.core.paths import VaultPaths
from knox.core.pipeline.config import (
    ALL_TIERS,
    TIER_DESCRIPTIONS,
    TIER_LEVEL,
    PipelineConfig,
    Tier,
)
from knox.core.pipeline.stages._compile import load_compiled_wiki, run_compile_tier, run_output_tier
from knox.core.pipeline.stages._index import detect_resume_tier, iter_sources, run_index_tier
from knox.core.pipeline.stages._parse import parse_and_chunk
from knox.core.pipeline.stages._resolve import apply_tags_to_tree, resolve_chunks
from knox.core.pipeline.state import TierState, TierStateStore
from knox.core.templates import get_template_loader
from knox.llm.models import set_user_providers
from knox.logging import get_logger
from knox.memory.lifecycle import update_memory_for_compile
from knox.output.base import BaseOutputWriter
from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.parsers.base import ParserRouter
from knox.resolution.base import BaseResolver
from knox.trust import filter_wiki

logger = get_logger(__name__)

# Type for progress event callbacks. Parameters:
#   phase   — pipeline phase ("index", "parse", "resolve", "compile", "output")
#   current — current item (0 = phase starting, >0 = progress)
#   total   — total items in this phase
#   detail  — human detail (filename, status, etc.)
ProgressCallback = Callable[[str, int, int, str], None]

# Legacy string-constant re-exports (kept for backward compatibility).
TIER_INDEX: Tier = Tier.INDEX
TIER_CHUNK: Tier = Tier.CHUNK
TIER_RESOLVE: Tier = Tier.RESOLVE
TIER_COMPILE: Tier = Tier.COMPILE
TIER_OUTPUT: Tier = Tier.OUTPUT

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
        summarizer: NodeSummarizer | None = None,
    ) -> None:
        self.config = config
        set_user_providers(config.providers)
        self.downloader = downloader
        self.router = router
        self.chunker = chunker
        self.resolver = resolver
        self.compiler = compiler
        self.output_writer = output_writer
        self.summarizer = summarizer
        self._base_dir = config.resolved_base_dir()
        self.state_store = state_store or TierStateStore(self._default_state_path(), self._base_dir)
        self._progress_callback: ProgressCallback | None = None
        self._cache_store = CacheStore(self._cache_dir())
        self._tier_timings: dict[str, float] = {}
        self.last_chunk_count: int = 0
        self.xref_result: Any = None

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

    # ── Public API ──────────────────────────────────────────────

    def run(self, progress_callback: ProgressCallback | None = None) -> CompiledWiki:  # noqa: PLR0915
        """Execute the full pipeline from sources to compiled wiki."""
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

        # Auto-resume.
        if self.config.start_at_tier == Tier.INDEX and not self.config.force:
            auto_tier = detect_resume_tier(self)
            if auto_tier is not None:
                logger.info(
                    "Auto-resume: previous run detected. Resuming at tier '%s'. Use --start-at-tier index or --force for full re-run.",
                    auto_tier,
                )
                self.config.start_at_tier = Tier(auto_tier)

        self._validate_prerequisites()

        # --force: invalidate all caches.
        if self.config.force:
            logger.info("Force mode enabled — invalidating all caches")
            self._cache_store.invalidate()

        all_trees: list[Any] = []
        resolved: list[Any] = []
        wiki = CompiledWiki(nodes=[], edges=[], toc=[], metadata={})
        self._tier_timings.clear()

        if self._should_run(Tier.INDEX):
            self._emit("index", 0, 1, "starting")
            _t0 = time.monotonic()
            self.run_tier(Tier.INDEX)
            self._tier_timings["index"] = time.monotonic() - _t0
            self._emit("index", 1, 1, "done")
        if self._should_run(Tier.CHUNK):
            _t0 = time.monotonic()
            all_trees = parse_and_chunk(self)
            self.last_chunk_count = sum(1 for t in all_trees for _ in t.leaves())
            self._tier_timings["chunk"] = time.monotonic() - _t0
            self._record_tier_state(1)
        if self._should_run(Tier.RESOLVE):
            if not all_trees:
                all_trees = parse_and_chunk(self)
            leaf_count = sum(1 for t in all_trees for _ in t.leaves())
            self._emit("resolve", 0, leaf_count, "starting")
            _t0 = time.monotonic()
            resolved = resolve_chunks(self, all_trees)
            self._tier_timings["resolve"] = time.monotonic() - _t0
            self._record_tier_state(2)
            self._emit("resolve", len(resolved), len(resolved), "done")
        if self._should_run(Tier.COMPILE):
            if not resolved:
                all_trees = parse_and_chunk(self)
                resolved = resolve_chunks(self, all_trees)
            self._emit("compile", 0, 1, "starting")
            _t0 = time.monotonic()
            if self.compiler:
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
            wiki = load_compiled_wiki(self) or wiki
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
        wiki.metadata[str(WikiField.LLM_MODEL)] = self.config.llm.default_model

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

        logger.info(
            "KNOX pipeline complete: %s -> %s (%d nodes)",
            self.config.source_dir,
            self.config.output_dir,
            len(wiki.nodes),
        )
        return wiki

    def run_tier(self, tier: str) -> None:
        """Run a single tier by name.

        Valid names: ``index``, ``chunk``, ``resolve``, ``compile``, ``output``.
        """
        if tier not in TIER_LEVEL:
            msg = f"Unknown tier {tier!r}. Valid: {ALL_TIERS}"
            raise ValueError(msg)

        target_level = TIER_LEVEL[Tier(tier)]

        if target_level >= TIER_LEVEL[TIER_INDEX]:
            run_index_tier(self)
        if target_level >= TIER_LEVEL[TIER_CHUNK]:
            parse_and_chunk(self)
        if target_level >= TIER_LEVEL[TIER_COMPILE]:
            run_compile_tier(self)
        if target_level >= TIER_LEVEL[TIER_OUTPUT]:
            run_output_tier(self)

        self.save_state()

    def iter_sources(self) -> list:
        """Yield source files to process."""
        return iter_sources(self)

    def list_tiers(self) -> list[dict[str, Any]]:
        """Return the available tiers with their current persisted state."""
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

    # ── Internal helpers ────────────────────────────────────────

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

    def _should_run(self, tier: Tier) -> bool:
        """Check whether *tier* falls within [start, stop]."""
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

    def _cache_dir(self) -> Path:
        """Resolve the cache directory."""
        if self.config.cache_dir:
            return self.config.cache_dir
        return self.config.paths.cache_dir

    def _apply_tags_to_tree(self, tree: Any, source_file: Path) -> None:
        """Attach directory-based tags and CLI-provided tags to every leaf chunk in *tree*."""
        apply_tags_to_tree(tree, source_file, self.config.source_dir, self.config.source_tags)
