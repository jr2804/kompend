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
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from knox.chunking.base import BaseChunker
from knox.chunking.hybrid import HybridChunker
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.compilation.two_phase import TwoPhaseCompiler
from knox.compilation.wicer import WiCERCompiler
from knox.core.cache_store import CacheStore
from knox.core.hooks import HookEvent, fire_hook
from knox.core.memory import update_memory_for_compile
from knox.core.models import CompiledWiki, DocumentElement, ElementType, ParsedDocument, WikiNode
from knox.core.paths import CACHE_SUBDIR, COMPILE_STATE_FILE
from knox.core.pipeline_defs import (
    ALL_TIERS,
    TIER_DESCRIPTIONS,
    TIER_LEVEL,
    Compiler,
    PipelineConfig,
    Tier,
)
from knox.core.template_loader import get_template_loader
from knox.core.tier_state import TierState, TierStateStore
from knox.llm.client import LLMClient
from knox.output.base import BaseOutputWriter
from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.parsers.base import BaseParser
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.resolution.base import BaseResolver
from knox.resolution.chain import ResolverChain
from knox.resolution.llm_resolver import LLMResolver
from knox.resolution.vlm_enricher import VLMEnricher
from knox.trust import filter_sensitive, filter_wiki

logger = logging.getLogger(__name__)

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
        parser: BaseParser | None = None,
        chunker: BaseChunker | None = None,
        resolver: BaseResolver | None = None,
        compiler: BaseCompiler | None = None,
        output_writer: BaseOutputWriter | None = None,
        state_store: TierStateStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.downloader = downloader
        self.parser = parser
        self.chunker = chunker
        self.resolver = resolver
        self.compiler = compiler
        self.output_writer = output_writer
        self.llm_client = llm_client
        self.state_store = state_store or TierStateStore(self._default_state_path())
        self._progress_callback: ProgressCallback | None = None
        self._cache_store = CacheStore(self._cache_dir())
        self._tier_timings: dict[str, float] = {}
        self.last_chunk_count: int = 0

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

        # Load user notes and inject into template global context
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
        fire_hook(
            HookEvent.PRE_COMPILE,
            source_dir=str(self.config.source_dir),
            output_dir=str(self.config.output_dir),
            start_at_tier=self.config.start_at_tier.value,
            stop_at_tier=self.config.stop_at_tier.value,
        )

        self._validate_prerequisites()

        # --force: invalidate all caches before starting.
        if self.config.force:
            logger.info("Force mode enabled — invalidating all caches")
            self._cache_store.invalidate()

        all_chunks: list[Any] = []
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
            all_chunks = self._parse_and_chunk()
            self.last_chunk_count = len(all_chunks)
            self._tier_timings["chunk"] = time.monotonic() - _t0
            self._record_tier_state(1)
        if self._should_run(Tier.RESOLVE):
            # When T1 was skipped (start_at_tier >= resolve), warm-load
            # chunks from the T1 cache so the resolver has input.
            if not all_chunks:
                all_chunks = self._parse_and_chunk()
            self._emit("resolve", 0, len(all_chunks), "starting")
            _t0 = time.monotonic()
            resolved = self._resolve_chunks(all_chunks)
            self._tier_timings["resolve"] = time.monotonic() - _t0
            self._record_tier_state(2)
            self._emit("resolve", len(resolved), len(resolved), "done")
        if self._should_run(Tier.COMPILE):
            # When T2 was skipped (start_at_tier >= compile), warm-load
            # chunks and resolve them from cache.
            if not resolved:
                all_chunks = self._parse_and_chunk()
                resolved = self._resolve_chunks(all_chunks)
            self._emit("compile", 0, 1, "starting")
            _t0 = time.monotonic()
            if self.compiler:
                wiki = self.compiler.compile(resolved)
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

        wiki.metadata["tier_timings"] = {k: round(v, 2) for k, v in self._tier_timings.items()}
        if self.llm_client is not None:
            wiki.metadata["llm_usage"] = self.llm_client.usage.to_dict()
            wiki.metadata["llm_model"] = self.config.llm.default_model
            report = self.llm_client.degradation_report()
            wiki.metadata["llm_degradation"] = report
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

        logger.info("KNOX pipeline complete: %s -> %s", self.config.source_dir, self.config.output_dir)
        fire_hook(
            HookEvent.POST_COMPILE,
            wiki=wiki,
            output_dir=str(self.config.output_dir),
            node_count=len(wiki.nodes),
        )
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
        if target_level >= TIER_LEVEL[TIER_RESOLVE]:
            self._run_resolve_tier()
        if target_level >= TIER_LEVEL[TIER_COMPILE]:
            self._run_compile_tier()
        if target_level >= TIER_LEVEL[TIER_OUTPUT]:
            self._run_output_tier()

        self.save_state()

    def iter_sources(self) -> Any:
        """Yield source files to process.

        If ``config.input_files`` is set, yields those directly.
        Otherwise auto-discovers supported files from ``config.source_dir``.
        """
        if self.config.input_files:
            yield from self.config.input_files
            return
        for ext in ("*.pdf", "*.PDF", "*.docx", "*.DOCX", "*.doc", "*.DOC", "*.epub", "*.md", "*.txt", "*.html", "*.pptx"):
            yield from self.config.source_dir.rglob(ext)

    def list_tiers(self) -> list[dict[str, Any]]:
        """Return the available tiers with their current persisted state.

        Used by ``knox compile --list-tiers`` to show progress and by
        tests to assert the tier contract. Each entry contains the
        tier name, level, the current persisted max level per source,
        and a sample of recent sources.
        """
        state = self.state
        distribution = state.tier_distribution()
        sample = list(state.entries.values())[:5]
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
        """Emit a progress event if a callback is registered."""
        if self._progress_callback is not None:
            try:
                self._progress_callback(phase, current, total, detail)
            except Exception:  # noqa: BLE001
                logger.debug("Progress callback error", exc_info=True)

    def _default_state_path(self) -> Path:
        """Resolve the persistent state file path."""
        if self.config.tier_state_path is not None:
            return self.config.tier_state_path
        return self.config.knox_dir() / COMPILE_STATE_FILE

    def _record_tier_state(self, tier: int) -> None:
        """Update current_tier for every known source file."""
        state = self.state
        for source_file in self.iter_sources():
            stem = source_file.stem
            try:
                content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            except OSError:
                continue
            self.state_store.record(state, stem, source_file, content_hash, tier=tier)
        self.state_store.save(state)

    def _run_index_tier(self) -> None:
        """Tier 0: download online sources and record state for local files."""
        if self.downloader and self.config.online_sources:
            for url in self.config.online_sources:
                self.downloader.download(url, self.config.source_dir)

    def _run_resolve_tier(self) -> None:
        """Tier 2: resolve chunks into ResolvedChunks (if a resolver is wired)."""
        # Re-runs the chunk->resolve step; cheap when no resolver is set.
        chunks = self._parse_and_chunk()
        if not self.resolver:
            return
        for chunk in chunks:
            self.resolver.resolve(chunk)

    def _run_compile_tier(self) -> None:
        """Tier 3: compile resolved chunks into a CompiledWiki."""
        chunks = self._parse_and_chunk()
        resolved = [self.resolver.resolve(chunk) for chunk in chunks] if self.resolver else list(chunks)
        if not self.compiler:
            return
        wiki = self.compiler.compile(resolved)
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
        return self.config.knox_dir() / "compiled_wiki.json"

    def _save_compiled_wiki(self, wiki: CompiledWiki) -> None:

        path = self._compiled_wiki_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [
                {
                    "id": n.id,
                    "title": n.title,
                    "content": n.content,
                    "node_type": n.node_type,
                    "source_chunks": n.source_chunks or [],
                    "metadata": n.metadata,
                }
                for n in wiki.nodes
            ],
            "edges": wiki.edges,
            "toc": wiki.toc,
            "quality_score": wiki.quality_score,
            "iteration": wiki.iteration,
            "metadata": wiki.metadata,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _load_compiled_wiki(self) -> CompiledWiki | None:
        path = self._compiled_wiki_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug("Compiled wiki snapshot unreadable — treating as missing", exc_info=True)
            return None
        nodes = [WikiNode(**n) for n in data.get("nodes", [])]
        return CompiledWiki(
            nodes=nodes,
            edges=data.get("edges", []),
            toc=data.get("toc", []),
            quality_score=data.get("quality_score", 0.0),
            iteration=data.get("iteration", 0),
            metadata=data.get("metadata", {}),
        )

    def _parse_and_chunk(self) -> list[Any]:  # noqa: PLR0915 — nested _process_one plus chunk cache logic
        """Parse documents and chunk them, with optional caching and parallelism.

        When tier state is available, sources that already match the
        persisted hash and tier are skipped to keep the per-tier
        execution incremental.
        """
        source_files = list(self.iter_sources())
        if not source_files:
            return []

        self._emit("parse", 0, len(source_files), "starting")

        workers = max(1, self.config.workers)
        use_cache = not self.config.no_cache

        state = self.state

        def _process_one(source_file: Path) -> list[Any]:
            """Parse a single file (cache-aware) and chunk it."""
            stem = source_file.stem
            content_hash = ""

            # 1. Load or parse the document.
            if use_cache and not self.config.force:
                content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

            parsed = self._load_cache(source_file) if use_cache else None
            if parsed is None:
                if use_cache and not self.config.force:
                    cached_entry = state.entries.get(stem)
                    if cached_entry and cached_entry.content_hash == content_hash and cached_entry.current_tier >= 1:
                        parsed = self._load_cache(source_file)
                if parsed is None:
                    if not self.parser:
                        return []
                    try:
                        parsed = self.parser.parse(source_file)
                        _ = len(parsed.elements)
                    except Exception:
                        logger.exception("Failed to parse %s", source_file)
                        return []
                    if use_cache:
                        self._save_cache(source_file, parsed)
                        if not content_hash:
                            content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

            if self.config.privacy_filter:
                self._apply_privacy_filter(parsed.elements)

            if not self.chunker:
                return []

            # 2a. Chunk cache lookup (Tier 1).
            if use_cache and content_hash:
                chunk_deps = {
                    "source": content_hash,
                    "chunk_effective_max": str(getattr(self.chunker, "_effective_max", 0)),
                }
                cached = self._cache_store.get("chunk", chunk_deps)
                if cached is not None:
                    if content_hash:
                        self.state_store.record(state, stem, source_file, content_hash, tier=1)
                    return cached

            # 2b. Chunk and cache.
            chunks = self.chunker.chunk(parsed)

            if content_hash:
                self.state_store.record(state, stem, source_file, content_hash, tier=1)
                chunk_deps = {
                    "source": content_hash,
                    "chunk_effective_max": str(getattr(self.chunker, "_effective_max", 0)),
                }
                self._cache_store.set("chunk", chunk_deps, chunks)
            return chunks

        all_chunks: list[Any] = []
        if workers > 1 and len(source_files) > 1:
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_one, sf): sf for sf in source_files}
                for future in as_completed(futures):
                    sf = futures[future]
                    try:
                        all_chunks.extend(future.result())
                    except Exception:
                        logger.exception("Worker failed for %s", sf)
                    finally:
                        completed += 1
                        self._emit("parse", completed, len(source_files), sf.name)
        else:
            for idx, sf in enumerate(source_files):
                all_chunks.extend(_process_one(sf))
                self._emit("parse", idx + 1, len(source_files), sf.name)

        self._emit("parse", len(source_files), len(source_files), "done")
        return all_chunks

    def _cache_dir(self) -> Path:
        """Resolve the cache directory."""
        if self.config.cache_dir:
            return self.config.cache_dir
        return self.config.knox_dir() / CACHE_SUBDIR

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
                    reading_order=el.get("reading_order"),
                    heading_level=el.get("heading_level"),
                    metadata=el.get("metadata", {}),
                )
                for el in data.get("elements", [])
            ]
            return ParsedDocument(
                source_path=Path(data["source_path"]),
                elements=elements,
                title=data.get("title", ""),
                author=data.get("author", ""),
                language=data.get("language", "en"),
                page_count=data.get("page_count", 0),
                metadata=data.get("metadata", {}),
            )
        except Exception:
            logger.debug("Cache read failed for %s — ignoring", cache_path, exc_info=True)
            return None

    def _save_cache(self, source_file: Path, doc: ParsedDocument) -> None:
        """Write a ParsedDocument to cache."""
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self._cache_key(source_file)}.json"
        try:
            data = {
                "source_path": str(doc.source_path),
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
                        "reading_order": el.reading_order,
                        "heading_level": el.heading_level,
                        "metadata": el.metadata,
                    }
                    for el in doc.elements
                ],
            }
            cache_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception:
            logger.debug("Cache write failed for %s — ignoring", cache_path, exc_info=True)

    def _resolve_chunks(self, chunks: list[Any]) -> list[Any]:
        """Resolve chunks with per-chunk cache (Tier 2)."""
        if not self.resolver:
            return list(chunks)

        resolved: list[Any] = []
        vlm_model = self.config.llm.vlm_model or "none"
        use_cache = not self.config.no_cache
        total = len(chunks)

        for idx, chunk in enumerate(chunks):
            if use_cache and chunk.text:
                deps = {
                    "chunk_text": hashlib.sha256(chunk.text.encode()).hexdigest(),
                    "vlm_model": vlm_model,
                }
                cached = self._cache_store.get("resolve", deps)
                if cached is not None:
                    resolved.append(cached)
                    self._emit("resolve", idx + 1, total, "")
                    continue

            rc = self.resolver.resolve(chunk)

            if use_cache and chunk.text:
                deps = {
                    "chunk_text": hashlib.sha256(chunk.text.encode()).hexdigest(),
                    "vlm_model": vlm_model,
                }
                self._cache_store.set("resolve", deps, rc)
            resolved.append(rc)
            self._emit("resolve", idx + 1, total, "")

        return resolved

    @staticmethod
    def _apply_privacy_filter(elements: list[Any]) -> None:
        """Apply privacy filter in-place to document elements."""
        for el in elements:
            if isinstance(el.content, str):
                el.content, _ = filter_sensitive(el.content)
            elif isinstance(el.content, list):
                el.content = [filter_sensitive(item)[0] if isinstance(item, str) else item for item in el.content]

    @staticmethod
    def _cache_key(source_file: Path) -> str:
        """SHA-256 of file contents for cache key."""
        digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        return f"{source_file.stem}_{digest[:12]}"


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
        fallback_chain=list(config.llm.fallback_chain),
        max_retries_per_model=config.llm.fallback_max_retries,
        timeout=config.llm.timeout,
        _fallback_max_input=config.llm.fallback_max_input,
        _fallback_max_output=config.llm.fallback_max_output,
        _num_ctx_min=config.llm.num_ctx_min,
        _num_ctx_max=config.llm.num_ctx_max,
        _context_window_cap=config.llm.context_window_cap,
    )


def build_pipeline(
    config: PipelineConfig,
) -> WikiPipeline:
    """Shared factory used by both CLI and MCP.

    Constructs a WikiPipeline with default components:
    - DoclingParser (lazy-fallback)
    - HybridChunker
    - Compiler selected by ``config.compiler``:
      - ``"graph"``: TwoPhaseCompiler wrapping GraphCompiler (default)
      - ``"wicer"``: WiCERCompiler (compile-evaluate-refine loop)
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

    parser = None
    try:
        parser = DoclingParser(image_mode=config.image_mode)
    except Exception as exc:
        logger.warning("DoclingParser not available: %s", exc, exc_info=True)

    # T0 (index) and T1 (chunk) run without an LLM. Only build the
    # LLM-dependent components when the [start, stop] tier range reaches
    # T2 (resolve) or beyond. This lets `knox ingest` (stop_at_tier=CHUNK)
    needs_llm = TIER_LEVEL[config.stop_at_tier] >= TIER_LEVEL[Tier.RESOLVE]

    resolver: BaseResolver | None = None
    base_compiler: BaseCompiler | None = None
    chunker_context_window = None
    if needs_llm:
        llm_client = build_llm_client(config)
        chunker_context_window = llm_client.context_window
        resolver = _build_resolver_chain(config, llm_client)

        if config.compiler == Compiler.WICER:
            base_compiler = WiCERCompiler(
                config=config,
                base_compiler=GraphCompiler(
                    model=config.llm.default_model,
                    min_content_length=config.budget.min_content_length,
                    min_concept_sources=config.budget.min_concept_sources,
                    max_concept_pages=config.budget.max_concept_pages,
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
        parser=parser,
        chunker=HybridChunker(
            context_window=chunker_context_window,
            max_tokens=config.budget.chunk_size_max if chunker_context_window is None else None,
            chunk_size_min=config.budget.chunk_size_min,
            chunk_size_max=config.budget.chunk_size_max,
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
    )


def _build_resolver_chain(config: PipelineConfig, llm_client: LLMClient) -> BaseResolver:
    """Construct the default resolver chain (M7).

    When relation extraction is enabled, an :class:`LLMResolver` runs first
    to extract typed relations with confidence. When a VLM model is
    configured, a :class:`VLMEnricher` is appended to enrich image chunks
    without discarding earlier resolver output.
    """
    resolvers: list[BaseResolver] = []

    if config.resolve_relations:
        if config.llm.default_model:
            resolvers.append(
                LLMResolver(
                    client=llm_client,
                    confidence_threshold=config.relation_confidence_threshold,
                )
            )
            logger.debug(
                "LLMResolver added to resolver chain (threshold=%s)",
                config.relation_confidence_threshold,
            )
        else:
            logger.warning("Relation extraction requested but no default LLM model is configured")

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
            )
            vlm_client.usage = llm_client.usage
            resolvers.append(VLMEnricher(client=vlm_client))
            logger.debug("VLMEnricher added to resolver chain (model=%s)", config.llm.vlm_model)
        except Exception as exc:
            logger.debug("VLMEnricher not available: %s", exc, exc_info=True)

    if not resolvers:
        logger.info("No resolvers configured — chunks will pass through without resolution")

    return ResolverChain(resolvers)
