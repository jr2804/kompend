"""Pipeline constants, tier definitions, and configuration.

This module is the single source of truth for:

- :class:`Tier` — the 5 pipeline tier identifiers
- :data:`ALL_TIERS` — ordered tuple of every tier
- :data:`TIER_LEVEL` — tier name → numeric level (used for skip/ordering decisions)
- :data:`TIER_DESCRIPTIONS` — human-readable description per tier (with inputs/outputs)
- :data:`PipelineConfig` — dataclass holding the full pipeline configuration

Both the CLI (``knox.cli.commands``) and the orchestrator (``knox.core.pipeline``)
file is part of the public configuration contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from knox.core.config import (
    BudgetConfig,
    EmbeddingConfig,
    ImageMode,
    LLMConfig,
    LlmDataFormat,
    OutputConfig,
    OutputFormat,
    PdfConfig,
    ResolvedConfig,
)
from knox.core.config.keys import (
    COMPILE_BUDGET_CHUNK_SIZE_MAX,
    COMPILE_BUDGET_CHUNK_SIZE_MIN,
    COMPILE_BUDGET_CONCEPTS_PER_BATCH,
    COMPILE_BUDGET_EXTRACT_OUTPUT_MAX,
    COMPILE_BUDGET_GEN_OUTPUT_MAX,
    COMPILE_BUDGET_LINT_OUTPUT_MAX,
    COMPILE_BUDGET_MAX_CONCEPT_PAGES,
    COMPILE_BUDGET_MIN_CONCEPT_SOURCES,
    COMPILE_BUDGET_MIN_CONTENT_LENGTH,
    COMPILE_BUDGET_OVERVIEW_OUTPUT_MAX,
    COMPILE_BUDGET_SUMMARY_CHUNK_THRESHOLD,
    COMPILE_BUDGET_SUMMARY_MAX_CONTENT_CHARS,
    COMPILE_BUDGET_SUMMARY_SKIP_SINGLE_CHILD,
    COMPILE_BUDGET_SUMMARY_TARGET_CHARS,
    COMPILE_BUDGET_VERIFY_MIN_NGRAM,
    COMPILE_BUDGET_VERIFY_OUTPUT_MAX,
    COMPILE_BUDGET_WICER_EVAL_OUTPUT_MAX,
    COMPILE_BUDGET_WICER_MAX_ITERATIONS,
    COMPILE_BUDGET_WICER_PROBE_OUTPUT_MAX,
    COMPILE_CACHE_DIR,
    COMPILE_COMPILER,
    COMPILE_EMBEDDING_MODEL,
    COMPILE_FORCE,
    COMPILE_FORMAT,
    COMPILE_IMAGE_MODE,
    COMPILE_LLM_CONTEXT_WINDOW_CAP,
    COMPILE_LLM_DATA_FORMAT,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_FALLBACK_MAX_INPUT,
    COMPILE_LLM_FALLBACK_MAX_OUTPUT,
    COMPILE_LLM_FALLBACK_MAX_RETRIES,
    COMPILE_LLM_NUM_CTX_MAX,
    COMPILE_LLM_NUM_CTX_MIN,
    COMPILE_LLM_ROLE_MODELS,
    COMPILE_LLM_TIMEOUT,
    COMPILE_LLM_VLM_MODEL,
    COMPILE_MEMORY_DECAY_THRESHOLD,
    COMPILE_MEMORY_ENABLED,
    COMPILE_MEMORY_HALF_LIFE_DAYS,
    COMPILE_NO_CACHE,
    COMPILE_OUTPUT_FLAVOR,
    COMPILE_OUTPUT_FRONTMATTER,
    COMPILE_OUTPUT_RUMDL_CONFIG,
    COMPILE_PDF_MAX_PAGES_BEFORE_CHUNKING,
    COMPILE_PDF_PAGE_CHUNK_OVERLAP,
    COMPILE_PDF_PAGE_CHUNK_SIZE,
    COMPILE_RESPECT_GITIGNORE,
    COMPILE_START_AT_TIER,
    COMPILE_STOP_AT_TIER,
    COMPILE_TIER_STATE_PATH,
    COMPILE_VLM_MIN_IMAGE_BYTES,
    COMPILE_VLM_MIN_IMAGE_DIMENSION,
    COMPILE_WORKERS,
    INGEST_PARSER,
    PATHS_WIKI_DIR,
    TRUST_PRIVACY_FILTER,
)
from knox.core.paths import VaultPaths


class Tier(StrEnum):
    """Identifiers for the KNOX 5-tier compilation pipeline.

    Tiers are cumulative: running a higher tier first ensures the
    state of every lower tier is up to date.
    """

    INDEX = "index"
    CHUNK = "chunk"
    RESOLVE = "resolve"
    COMPILE = "compile"
    OUTPUT = "output"


class Compiler(StrEnum):
    """Compiler backend selection.

    - ``GRAPH``: GraphCompiler (heading-based grouping, default)
    - ``WICER``: WiCERCompiler (iterative compile-evaluate-refine loop)
    """

    GRAPH = "graph"
    WICER = "wicer"


# Ordered tuple used for `--list-tiers` and tier-name validation.
ALL_TIERS: tuple[Tier, ...] = (
    Tier.INDEX,
    Tier.CHUNK,
    Tier.RESOLVE,
    Tier.COMPILE,
    Tier.OUTPUT,
)

# Tier name → level number used for skip decisions (higher = runs more).
TIER_LEVEL: dict[Tier, int] = {
    Tier.INDEX: 0,
    Tier.CHUNK: 1,
    Tier.RESOLVE: 2,
    Tier.COMPILE: 3,
    Tier.OUTPUT: 4,
}

# Tier name → human-readable description (with inputs and outputs).
# Kept short enough to fit in a CLI table; full prose lives in docs/.
TIER_DESCRIPTIONS: dict[Tier, str] = {
    Tier.INDEX: ("Tier 0: index sources — record local files and download online URLs. Inputs: source_dir, online_sources. Outputs: tier_state entries."),
    Tier.CHUNK: ("Tier 1: parse documents and split them into chunks. Inputs: indexed source files. Outputs: list of chunks (cached on disk)."),
    Tier.RESOLVE: ("Tier 2: resolve chunks — extract entities/links (LLM-dependent). Inputs: chunks. Outputs: resolved chunks with entities/relations."),
    Tier.COMPILE: ("Tier 3: compile resolved chunks into a knowledge graph (CompiledWiki). Inputs: resolved chunks. Outputs: CompiledWiki snapshot."),
    Tier.OUTPUT: ("Tier 4: render the compiled wiki to the output directory. Inputs: CompiledWiki snapshot. Outputs: rendered files (markdown/json/typst)."),
}


@dataclass
class PipelineConfig:
    """Configuration for the KNOX pipeline.

    This dataclass is the single shape consumed by :class:`WikiPipeline` and
    its factory :func:`build_pipeline`. Defaults are intentionally conservative
    so the pipeline can run on a developer laptop without any configuration.

    Directory layout::

        base_dir/
        ├── .knox/         ← internal state (cache, tier state, vectors, etc.)
        ├── sources/        ← source_dir: user-owned inputs
        ├── wiki/           ← output_dir: generated pages
        ├── index.md        ← entry point (TOC)
        └── overview.md      ← LLM-generated overview (opt-in)

    When ``base_dir`` is not set explicitly, it defaults to ``output_dir.parent``.
    All ``.knox`` paths resolve relative to ``base_dir``.
    """

    source_dir: Path
    output_dir: Path
    llm: LLMConfig
    embedding: EmbeddingConfig
    base_dir: Path | None = None  # None → output_dir.parent; .knox lives here
    image_mode: ImageMode = ImageMode.FILE
    vlm_min_image_bytes: int = 2048
    vlm_min_image_dimension: int = 64
    online_sources: list[str] = field(default_factory=list)
    input_files: list[Path] | None = None  # explicit file list; None = auto-discover from source_dir
    cache_dir: Path | None = None  # None = base_dir/.knox/cache
    no_cache: bool = False  # force re-parse even if cached
    workers: int = 1  # parallel parse+chunk workers
    start_at_tier: Tier = Tier.INDEX  # pipeline entry point (T0 by default)
    stop_at_tier: Tier = Tier.OUTPUT  # pipeline exit point (T4 by default; T1 for ingest)
    force: bool = False  # bypass persistent tier state and re-process everything
    tier_state_file: Path | None = None  # None = base_dir/.knox/compile_state.json
    ignore_patterns: list[str] | None = None  # glob patterns to exclude
    respect_gitignore: bool = True  # respect .gitignore in source dir
    compiler: Compiler = Compiler.GRAPH  # C4: compiler backend selection
    privacy_filter: bool = True  # redact PII/secrets from parsed content
    memory_enabled: bool = True  # Track node confidence/tiers/supersession/decay
    memory_half_life_days: float = 90.0  # Retention half-life in days
    memory_decay_threshold: float = 0.3  # Retention below this = stale
    notes_file: Path | None = None  # User notes file path (None = no notes)
    images_scale: float = 200.0 / 72.0  # Docling image resolution multiplier (200 DPI base)
    parser: str = "docling"  # ingestion backend name (see PARSER_BACKENDS)

    def resolved_base_dir(self) -> Path:
        """Return the effective base directory."""
        return self.base_dir if self.base_dir is not None else self.output_dir.parent

    def knox_dir(self) -> Path:
        """Return the ``.knox`` directory (always inside base_dir)."""
        return self.paths.knox_dir

    @property
    def paths(self) -> VaultPaths:
        """Return a :class:`VaultPaths` for this pipeline's base directory."""
        return VaultPaths(
            self.resolved_base_dir(),
            sources_dir_name=self.source_dir.name,
            output_dir_name=self.output_dir.name,
        )

    # Sub-configs (additive for backward compatibility)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pdf: PdfConfig = field(default_factory=PdfConfig)

    @classmethod
    def from_resolved(cls, cfg: ResolvedConfig, *, source_dir: Path, output_dir: Path, input_files: list[Path] | None = None) -> PipelineConfig:
        """Build PipelineConfig from a fully resolved config dict."""
        llm_fallback_chain = cfg.get(COMPILE_LLM_FALLBACK_CHAIN, [])
        llm_fallback_max_retries = cfg.get(COMPILE_LLM_FALLBACK_MAX_RETRIES, 2)
        cfg.get(COMPILE_LLM_DEFAULT_MODEL)
        role_models_raw = cfg.get(COMPILE_LLM_ROLE_MODELS, {})
        role_models = {k: str(v).strip() for k, v in role_models_raw.items() if str(v).strip()} if isinstance(role_models_raw, dict) else {}

        frontmatter_defaults = dict(cfg.get(COMPILE_OUTPUT_FRONTMATTER, {}))
        rumdl_config_file = cfg.get(COMPILE_OUTPUT_RUMDL_CONFIG)
        if isinstance(rumdl_config_file, str):
            rumdl_config_file = Path(rumdl_config_file)

        llm_config = LLMConfig(
            default_model=cfg.get(COMPILE_LLM_DEFAULT_MODEL),
            vlm_model=cfg.get(COMPILE_LLM_VLM_MODEL),
            fallback_chain=llm_fallback_chain,
            fallback_max_retries=llm_fallback_max_retries,
            timeout=cfg.get(COMPILE_LLM_TIMEOUT, 120.0),
            fallback_max_input=cfg.get(COMPILE_LLM_FALLBACK_MAX_INPUT, 8192),
            fallback_max_output=cfg.get(COMPILE_LLM_FALLBACK_MAX_OUTPUT, 4096),
            num_ctx_min=cfg.get(COMPILE_LLM_NUM_CTX_MIN, 2048),
            num_ctx_max=cfg.get(COMPILE_LLM_NUM_CTX_MAX, 131072),
            context_window_cap=cfg.get(COMPILE_LLM_CONTEXT_WINDOW_CAP),
            role_models=role_models,
            data_format=LlmDataFormat(cfg.get(COMPILE_LLM_DATA_FORMAT, "json")),
        )

        budget_config = BudgetConfig(
            concepts_per_batch=cfg.get(COMPILE_BUDGET_CONCEPTS_PER_BATCH, BudgetConfig().concepts_per_batch),
            verify_min_ngram=cfg.get(COMPILE_BUDGET_VERIFY_MIN_NGRAM, BudgetConfig().verify_min_ngram),
            wicer_max_iterations=cfg.get(COMPILE_BUDGET_WICER_MAX_ITERATIONS, BudgetConfig().wicer_max_iterations),
            min_content_length=cfg.get(COMPILE_BUDGET_MIN_CONTENT_LENGTH, BudgetConfig().min_content_length),
            min_concept_sources=cfg.get(COMPILE_BUDGET_MIN_CONCEPT_SOURCES, BudgetConfig().min_concept_sources),
            max_concept_pages=cfg.get(COMPILE_BUDGET_MAX_CONCEPT_PAGES, BudgetConfig().max_concept_pages),
            chunk_size_min=cfg.get(COMPILE_BUDGET_CHUNK_SIZE_MIN, BudgetConfig().chunk_size_min),
            chunk_size_max=cfg.get(COMPILE_BUDGET_CHUNK_SIZE_MAX, BudgetConfig().chunk_size_max),
            extract_output_max=cfg.get(COMPILE_BUDGET_EXTRACT_OUTPUT_MAX, BudgetConfig().extract_output_max),
            gen_output_max=cfg.get(COMPILE_BUDGET_GEN_OUTPUT_MAX, BudgetConfig().gen_output_max),
            overview_output_max=cfg.get(COMPILE_BUDGET_OVERVIEW_OUTPUT_MAX, BudgetConfig().overview_output_max),
            verify_output_max=cfg.get(COMPILE_BUDGET_VERIFY_OUTPUT_MAX, BudgetConfig().verify_output_max),
            lint_output_max=cfg.get(COMPILE_BUDGET_LINT_OUTPUT_MAX, BudgetConfig().lint_output_max),
            wicer_probe_output_max=cfg.get(COMPILE_BUDGET_WICER_PROBE_OUTPUT_MAX, BudgetConfig().wicer_probe_output_max),
            wicer_eval_output_max=cfg.get(COMPILE_BUDGET_WICER_EVAL_OUTPUT_MAX, BudgetConfig().wicer_eval_output_max),
            summary_max_content_chars=cfg.get(COMPILE_BUDGET_SUMMARY_MAX_CONTENT_CHARS, BudgetConfig().summary_max_content_chars),
            summary_target_chars=cfg.get(COMPILE_BUDGET_SUMMARY_TARGET_CHARS, BudgetConfig().summary_target_chars),
            summary_skip_single_child=cfg.get(COMPILE_BUDGET_SUMMARY_SKIP_SINGLE_CHILD, BudgetConfig().summary_skip_single_child),
            summary_chunk_threshold=cfg.get(COMPILE_BUDGET_SUMMARY_CHUNK_THRESHOLD, BudgetConfig().summary_chunk_threshold),
        )

        output_config = OutputConfig(
            format=cfg.get(COMPILE_FORMAT, OutputConfig().format),
            flavor=cfg.get(COMPILE_OUTPUT_FLAVOR, OutputConfig().flavor),
            frontmatter=frontmatter_defaults,
            rumdl_config=rumdl_config_file,
        )

        embedding_config = EmbeddingConfig(
            model=cfg.get(COMPILE_EMBEDDING_MODEL),
        )

        pdf_config = PdfConfig(
            max_pages_before_chunking=cfg.get(COMPILE_PDF_MAX_PAGES_BEFORE_CHUNKING, PdfConfig().max_pages_before_chunking),
            page_chunk_size=cfg.get(COMPILE_PDF_PAGE_CHUNK_SIZE, PdfConfig().page_chunk_size),
            page_chunk_overlap=cfg.get(COMPILE_PDF_PAGE_CHUNK_OVERLAP, PdfConfig().page_chunk_overlap),
        )

        # Resolve notes file path relative to wiki_dir
        notes_file_name = cfg.get("notes.file", "NOTES.md")
        wiki_dir = cfg.get(PATHS_WIKI_DIR)
        notes_file: Path | None = None
        if notes_file_name:
            base_dir = Path(wiki_dir) if wiki_dir else output_dir.parent
            notes_file = base_dir / notes_file_name

        return cls(
            source_dir=source_dir,
            output_dir=output_dir,
            image_mode=ImageMode(cfg.get(COMPILE_IMAGE_MODE, ImageMode.FILE)),
            vlm_min_image_bytes=cfg.get(COMPILE_VLM_MIN_IMAGE_BYTES, 2048),
            vlm_min_image_dimension=cfg.get(COMPILE_VLM_MIN_IMAGE_DIMENSION, 64),
            online_sources=[],
            input_files=input_files,
            cache_dir=cfg.get(COMPILE_CACHE_DIR),
            no_cache=cfg.get(COMPILE_NO_CACHE, False),
            workers=cfg.get(COMPILE_WORKERS, 1),
            start_at_tier=Tier(cfg.get(COMPILE_START_AT_TIER, Tier.INDEX)),
            stop_at_tier=Tier(cfg.get(COMPILE_STOP_AT_TIER, Tier.OUTPUT)),
            force=cfg.get(COMPILE_FORCE, False),
            tier_state_file=cfg.get(COMPILE_TIER_STATE_PATH),
            ignore_patterns=list(cfg.get("ignore.patterns", [])),
            respect_gitignore=cfg.get(COMPILE_RESPECT_GITIGNORE, True),
            compiler=cfg.get(COMPILE_COMPILER, Compiler.GRAPH),
            privacy_filter=cfg.get(TRUST_PRIVACY_FILTER, True),
            parser=cfg.get(INGEST_PARSER, "docling"),
            memory_enabled=cfg.get(COMPILE_MEMORY_ENABLED, True),
            memory_half_life_days=cfg.get(COMPILE_MEMORY_HALF_LIFE_DAYS, 90.0),
            memory_decay_threshold=cfg.get(COMPILE_MEMORY_DECAY_THRESHOLD, 0.3),
            notes_file=notes_file,
            llm=llm_config,
            budget=budget_config,
            output=output_config,
            embedding=embedding_config,
            pdf=pdf_config,
        )


__all__ = [
    "ALL_TIERS",
    "TIER_DESCRIPTIONS",
    "TIER_LEVEL",
    "Compiler",
    "ImageMode",
    "OutputFormat",
    "PipelineConfig",
    "Tier",
]
