"""Pipeline orchestrator for the KNOX 5-tier architecture."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from knox.chunking.base import BaseChunker
from knox.chunking.hybrid import HybridChunker
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.core.models import CompiledWiki, DocumentElement, ElementType, ParsedDocument
from knox.core.tier_state import TierState, TierStateStore
from knox.llm.client import LLMClient
from knox.output.base import BaseOutputWriter
from knox.preprocessing.downloaders.base import BaseDownloader
from knox.preprocessing.parsers.base import BaseParser
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.resolution.base import BaseResolver
from knox.trust import filter_sensitive, filter_wiki

logger = logging.getLogger(__name__)


# Tier identifiers for the Q20 tiered compilation feature. They are
# cumulative: tier 3 runs tiers 0+1+2+3, tier 1 runs tiers 0+1.
TIER_INDEX: ClassVar[str] = "index"
TIER_CHUNK: ClassVar[str] = "chunk"
TIER_RESOLVE: ClassVar[str] = "resolve"
TIER_COMPILE: ClassVar[str] = "compile"
TIER_OUTPUT: ClassVar[str] = "output"

# Ordered list used for `--list-tiers` and tier-int validation.
ALL_TIERS: ClassVar[tuple[str, ...]] = (TIER_INDEX, TIER_CHUNK, TIER_RESOLVE, TIER_COMPILE, TIER_OUTPUT)

# Tier name -> level number used for skip decisions (higher = runs more).
TIER_LEVEL: ClassVar[dict[str, int]] = {
    TIER_INDEX: 0,
    TIER_CHUNK: 1,
    TIER_RESOLVE: 2,
    TIER_COMPILE: 3,
    TIER_OUTPUT: 4,
}

# Mode short-cuts: a mode expands to a set of tier levels that should run.
MODE_LEVELS: ClassVar[dict[str, set[int]]] = {
    "full": {0, 1, 2, 3, 4},
    "index-only": {0, 1},
    "resolve-only": {2, 3, 4},
    "output-only": {4},
}


@dataclass
class PipelineConfig:
    """Configuration for the KNOX pipeline."""

    source_dir: Path
    output_dir: Path
    image_mode: str = "file"  # "file" or "base64"
    chunk_size: int = 512
    compile_target: str = "markdown"  # "markdown", "json", "typst"
    llm_model: str = "llama3.2"
    llm_provider: str = "ollama"
    vlm_model: str = "phi-3-vision"
    llm_fallback_chain: list[str] = field(default_factory=list)
    """Ordered list of fallback models tried when the primary model fails.

    Each entry is a litellm-style ``"provider/model"`` string, e.g.
    ``["openai/gpt-4o", "anthropic/claude-3-haiku-20240307"]``.
    Empty (default) means single-model mode — backwards compatible
    with the pre-Q12 behaviour.
    """
    llm_fallback_max_retries_per_model: int = 2
    """Transient-error retries before advancing to the next model.

    ``2`` means up to three attempts per model (1 + 2 retries).
    """
    llm_context_budget: int = 100000
    """Max characters of page content sent to the LLM in a single lint call.

    When page content exceeds this budget a warning is logged and the
    content is truncated to this many characters. Users can adjust via
    the ``--llm-context-budget`` CLI flag or the
    ``KNOX_LLM_CONTEXT_BUDGET`` env var.

    Default: 100 000 (roughly 25 000 tokens for most models).
    """
    llm_default_model_override: str | None = None
    """Explicit default model in litellm format (``"provider/model"``).

    When set, this is used directly as the primary model, bypassing
    ``llm_provider`` and ``llm_model``. Intended for the
    ``--llm-default-model`` CLI flag and ``KNOX_LLM_DEFAULT_MODEL``
    env var. When ``None`` (default), the model is constructed from
    ``llm_provider`` and ``llm_model`` as before.
    """
    online_sources: list[str] = field(default_factory=list)
    input_files: list[Path] | None = None  # explicit file list; None = auto-discover from source_dir
    cache_dir: Path | None = None  # None = .knox/cache inside output_dir
    no_cache: bool = False  # force re-parse even if cached
    workers: int = 1  # parallel parse+chunk workers
    mode: str = "full"  # full | index-only | resolve-only | output-only
    force: bool = False  # bypass persistent tier state and re-process everything
    tier_state_path: Path | None = None  # None = .knox/compile_state.json inside output_dir
    ignore_patterns: list[str] | None = None  # glob patterns to exclude
    respect_gitignore: bool = True  # respect .gitignore in source dir
    flavor: str | None = None  # Markdown flavor (obsidian, gfm, mkdocs, etc.); None = auto-detect
    frontmatter_defaults: dict[str, Any] | None = None  # default frontmatter fields for all pages (Q18)
    page_kind: str | None = None  # Page kind for N9 schema-driven compilation (concept, entity, comparison, overview, custom); None = auto-detect
    rumdl_config: Path | None = None  # path to custom .rumdl.toml config


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
    ) -> None:
        self.config = config
        self.downloader = downloader
        self.parser = parser
        self.chunker = chunker
        self.resolver = resolver
        self.compiler = compiler
        self.output_writer = output_writer
        self.state_store = state_store or TierStateStore(self._default_state_path())

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

    def active_levels(self) -> set[int]:
        """Return the set of tier levels enabled for the current mode."""
        return MODE_LEVELS.get(self.config.mode, MODE_LEVELS["full"])

    def run(self) -> CompiledWiki:
        """Execute the full pipeline from sources to compiled wiki."""
        logger.info("Starting KNOX pipeline (mode=%s): %s", self.config.mode, self.config.source_dir)

        levels = self.active_levels()
        all_chunks: list[Any] = []
        resolved: list[Any] = []
        wiki = CompiledWiki(nodes=[], edges=[], toc=[], metadata={})

        if 0 in levels:
            self.run_tier(TIER_INDEX)
        if 1 in levels:
            all_chunks = self._parse_and_chunk()
        if 2 in levels:
            if all_chunks:
                resolved = [self.resolver.resolve(chunk) for chunk in all_chunks] if self.resolver else list(all_chunks)
            else:
                resolved = self._load_resolved_from_disk() or []
        if 3 in levels:
            if self.compiler:
                wiki = self.compiler.compile(resolved)
            if self.config.privacy_filter:
                wiki, _ = filter_wiki(wiki)
        else:
            wiki = self._load_compiled_wiki() or wiki
        if 4 in levels and self.output_writer:
            if self.config.privacy_filter:
                wiki, _ = filter_wiki(wiki)
            self.output_writer.write(wiki, self.config.output_dir)

        self.save_state()
        logger.info("KNOX pipeline complete: %s -> %s", self.config.source_dir, self.config.output_dir)
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

        target_level = TIER_LEVEL[tier]

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

    def _default_state_path(self) -> Path:
        """Resolve the persistent state file path."""
        if self.config.tier_state_path is not None:
            return self.config.tier_state_path
        return self.config.output_dir / ".knox" / "compile_state.json"

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
        return self.config.output_dir / ".knox" / "compiled_wiki.json"

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
            logger.debug("Compiled wiki snapshot unreadable — treating as missing")
            return None
        from knox.core.models import WikiNode  # local import to avoid module-level cycle

        nodes = [WikiNode(**n) for n in data.get("nodes", [])]
        return CompiledWiki(
            nodes=nodes,
            edges=data.get("edges", []),
            toc=data.get("toc", []),
            quality_score=data.get("quality_score", 0.0),
            iteration=data.get("iteration", 0),
            metadata=data.get("metadata", {}),
        )

    def _load_resolved_from_disk(self) -> list[Any]:
        """Best-effort load of previously resolved chunks (not yet implemented).

        Returns an empty list — the resolver tier is idempotent and re-runs
        cheaply when no snapshot is present.
        """
        return []

    def _parse_and_chunk(self) -> list[Any]:
        """Parse documents and chunk them, with optional caching and parallelism.

        When tier state is available, sources that already match the
        persisted hash and tier are skipped to keep the per-tier
        execution incremental.
        """
        source_files = list(self.iter_sources())
        if not source_files:
            return []

        workers = max(1, self.config.workers)
        use_cache = not self.config.no_cache

        state = self.state

        def _process_one(source_file: Path) -> list[Any]:
            """Parse a single file (cache-aware) and chunk it."""
            stem = source_file.stem
            content_hash = ""
            if use_cache and not self.config.force:
                content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
                cached_entry = state.entries.get(stem)
                if cached_entry and cached_entry.content_hash == content_hash and cached_entry.current_tier >= 1:
                    parsed = self._load_cache(source_file)
                    if parsed is not None:
                        if not self.chunker:
                            return []
                        if self.config.privacy_filter:
                            for el in parsed.elements:
                                if isinstance(el.content, str):
                                    el.content, _ = filter_sensitive(el.content)
                                elif isinstance(el.content, list):
                                    el.content = [filter_sensitive(item)[0] if isinstance(item, str) else item for item in el.content]
                        return self.chunker.chunk(parsed)

            parsed = self._load_cache(source_file) if use_cache else None
            if parsed is None:
                if not self.parser:
                    return []
                try:
                    parsed = self.parser.parse(source_file)
                    parsed.elements  # ensure fully materialized
                except Exception:
                    logger.exception("Failed to parse %s", source_file)
                    return []
                if use_cache:
                    self._save_cache(source_file, parsed)
                    if not content_hash:
                        content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

            if self.config.privacy_filter:
                for el in parsed.elements:
                    if isinstance(el.content, str):
                        el.content, _ = filter_sensitive(el.content)
                    elif isinstance(el.content, list):
                        el.content = [filter_sensitive(item)[0] if isinstance(item, str) else item for item in el.content]

            if content_hash:
                self.state_store.record(state, stem, source_file, content_hash, tier=1)
            if not self.chunker:
                return []
            return self.chunker.chunk(parsed)

        all_chunks: list[Any] = []
        if workers > 1 and len(source_files) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_one, sf): sf for sf in source_files}
                for future in as_completed(futures):
                    try:
                        all_chunks.extend(future.result())
                    except Exception:
                        sf = futures[future]
                        logger.exception("Worker failed for %s", sf)
        else:
            for sf in source_files:
                all_chunks.extend(_process_one(sf))

        return all_chunks

    def _cache_dir(self) -> Path:
        """Resolve the cache directory."""
        if self.config.cache_dir:
            return self.config.cache_dir
        return self.config.output_dir / ".knox" / "cache"

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
            logger.debug("Cache read failed for %s — ignoring", cache_path)
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
            logger.debug("Cache write failed for %s — ignoring", cache_path)

    @staticmethod
    def _cache_key(source_file: Path) -> str:
        """SHA-256 of file contents for cache key."""
        digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        return f"{source_file.stem}_{digest[:12]}"


_TIER_DESCRIPTIONS: dict[str, str] = {
    TIER_INDEX: "Parse documents and update tier state.",
    TIER_CHUNK: "Chunk parsed documents.",
    TIER_RESOLVE: "Resolve references and entities (LLM-dependent).",
    TIER_COMPILE: "Compile resolved chunks into a knowledge graph.",
    TIER_OUTPUT: "Render the compiled wiki to the output directory.",
}


def build_llm_client(config: PipelineConfig) -> LLMClient:
    """Construct an :class:`LLMClient` from a :class:`PipelineConfig`.

    When ``config.llm_default_model_override`` is set it is used
    directly as the primary model. Otherwise the model is constructed
    from ``llm_provider`` and ``llm_model``. The fallback chain is
    taken verbatim from ``config.llm_fallback_chain`` and the retry
    budget from ``config.llm_fallback_max_retries_per_model``.
    """
    default_model = config.llm_default_model_override or f"{config.llm_provider}/{config.llm_model}"
    return LLMClient(
        default_model=default_model,
        fallback_chain=list(config.llm_fallback_chain),
        max_retries_per_model=config.llm_fallback_max_retries_per_model,
    )


def build_pipeline(
    config: PipelineConfig,
    *,
    two_phase: bool = True,
    prompt_budget: int = 200_000,
) -> WikiPipeline:
    """Shared factory used by both CLI and MCP.

    Constructs a WikiPipeline with default components:
    - DoclingParser (lazy-fallback)
    - HybridChunker
    - TwoPhaseCompiler (when ``two_phase`` is True) wrapping GraphCompiler
    - Output writer selected by ``config.compile_target``

    Parameters
    ----------
    config:
        Pipeline configuration.
    two_phase:
        When True (default), wrap the GraphCompiler in a TwoPhaseCompiler
        that performs extraction + cross-source merging before generation.
    prompt_budget:
        Per-concept char budget (N10 fair-share truncation).
    """
    writer_cls = BaseOutputWriter.get(config.compile_target)
    if writer_cls is None:
        available = ", ".join(BaseOutputWriter.available_formats())
        msg = f"Unknown output format {config.compile_target!r}. Available: {available}"
        raise ValueError(msg)

    parser = None
    try:
        parser = DoclingParser(image_mode=config.image_mode)
    except Exception as exc:
        logger.warning("DoclingParser not available: %s", exc)

    base_compiler: BaseCompiler = GraphCompiler()
    if two_phase:
        from knox.compilation.two_phase import TwoPhaseCompiler  # local import: avoid cycle

        resolved_model = config.llm_default_model_override or f"{config.llm_provider}/{config.llm_model}"
        base_compiler = TwoPhaseCompiler(
            base=GraphCompiler(),
            prompt_budget=prompt_budget,
            model=resolved_model,
        )

    return WikiPipeline(
        config=config,
        parser=parser,
        chunker=HybridChunker(chunk_size=config.chunk_size),
        compiler=base_compiler,
        output_writer=writer_cls(
            flavor=config.flavor,
            frontmatter_defaults=config.frontmatter_defaults or {},
            rumdl_config=config.rumdl_config,
        ),
    )
