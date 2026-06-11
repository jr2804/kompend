"""Pipeline constants, tier definitions, and configuration.

This module is the single source of truth for:

- :class:`Tier` — the 5 pipeline tier identifiers
- :class:`Mode` — high-level pipeline run modes (short-cuts over tier levels)
- :data:`ALL_TIERS` — ordered tuple of every tier
- :data:`TIER_LEVEL` — tier name → numeric level (used for skip/ordering decisions)
- :data:`MODE_LEVELS` — mode name → set of tier levels to run
- :data:`TIER_DESCRIPTIONS` — human-readable description per tier (with inputs/outputs)
- :data:`PipelineConfig` — dataclass holding the full pipeline configuration

Both the CLI (``knox.cli.commands``) and the orchestrator (``knox.core.pipeline``)
import these definitions from here. Keep the surface small and stable: this
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
    OutputConfig,
    OutputFormat,
    ResolvedConfig,
)


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

    - ``GRAPH``: TwoPhaseCompiler (two-phase extract-then-generate, default)
    - ``WICER``: WiCERCompiler (iterative compile-evaluate-refine loop)
    """

    GRAPH = "graph"
    WICER = "wicer"


class Mode(StrEnum):
    """High-level pipeline run modes.

    A mode expands to the set of tier levels that should run for
    that mode. See :data:`MODE_LEVELS` for the mapping.
    """

    FULL = "full"
    INDEX_ONLY = "index-only"
    RESOLVE_ONLY = "resolve-only"
    OUTPUT_ONLY = "output-only"


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

# Mode → set of tier levels to run.
MODE_LEVELS: dict[Mode, set[int]] = {
    Mode.FULL: {0, 1, 2, 3, 4},
    Mode.INDEX_ONLY: {0, 1},
    Mode.RESOLVE_ONLY: {2, 3, 4},
    Mode.OUTPUT_ONLY: {4},
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
    """

    source_dir: Path
    output_dir: Path
    llm: LLMConfig
    embedding: EmbeddingConfig
    image_mode: ImageMode = ImageMode.FILE
    chunk_size: int = 512
    online_sources: list[str] = field(default_factory=list)
    input_files: list[Path] | None = None  # explicit file list; None = auto-discover from source_dir
    cache_dir: Path | None = None  # None = .knox/cache inside output_dir
    no_cache: bool = False  # force re-parse even if cached
    workers: int = 1  # parallel parse+chunk workers
    mode: Mode = Mode.FULL
    force: bool = False  # bypass persistent tier state and re-process everything
    tier_state_path: Path | None = None  # None = .knox/compile_state.json inside output_dir
    ignore_patterns: list[str] | None = None  # glob patterns to exclude
    respect_gitignore: bool = True  # respect .gitignore in source dir
    page_kind: str | None = None  # Page kind for N9 schema-driven compilation (concept, entity, comparison, overview, custom); None = auto-detect
    compiler: Compiler = Compiler.GRAPH  # C4: compiler backend selection
    privacy_filter: bool = True  # redact PII/secrets from parsed content

    # Sub-configs (additive for backward compatibility)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_resolved(cls, cfg: ResolvedConfig, *, source_dir: Path, output_dir: Path, input_files: list[Path] | None = None) -> PipelineConfig:
        """Build PipelineConfig from a fully resolved config dict."""
        # Read old flat keys to populate flat fields
        llm_fallback_chain = cfg.get("compile.llm.fallback_chain", [])
        llm_fallback_max_retries = cfg.get("compile.llm.fallback_max_retries", 2)
        cfg.get("compile.llm.default_model")
        role_models_raw = cfg.get("compile.llm.role_models", {})
        role_models = {k: str(v).strip() for k, v in role_models_raw.items() if str(v).strip()} if isinstance(role_models_raw, dict) else {}

        frontmatter_defaults = dict(cfg.get("compile.output.frontmatter", {}))
        rumdl_config = cfg.get("compile.output.rumdl_config")
        if isinstance(rumdl_config, str):
            rumdl_config = Path(rumdl_config)

        # Read sub-configs
        llm_config = LLMConfig(
            default_model=cfg.get("compile.llm.default_model"),
            vlm_model=cfg.get("compile.llm.vlm_model"),
            fallback_chain=llm_fallback_chain,
            fallback_max_retries=llm_fallback_max_retries,
            context_budget=cfg.get("compile.llm.context_budget", 100_000),
            timeout=cfg.get("compile.llm.timeout", 120.0),
            num_ctx=cfg.get("compile.llm.num_ctx"),
            role_models=role_models,
        )

        budget_config = BudgetConfig(
            prompt=cfg.get("compile.prompt_budget", BudgetConfig().prompt),
            extract=cfg.get("compile.budget.extract", BudgetConfig().extract),
            generate_source_segment=cfg.get("compile.budget.generate_source_segment", BudgetConfig().generate_source_segment),
            concepts_per_batch=cfg.get("compile.budget.concepts_per_batch", BudgetConfig().concepts_per_batch),
            overview=cfg.get("compile.budget.overview", BudgetConfig().overview),
            verify=cfg.get("compile.budget.verify", BudgetConfig().verify),
            verify_min_ngram=cfg.get("compile.budget.min_ngram", BudgetConfig().verify_min_ngram),
            lint_page=cfg.get("compile.budget.lint_page", BudgetConfig().lint_page),
            wicer_probe=cfg.get("compile.budget.wicer_probe", BudgetConfig().wicer_probe),
            wicer_eval=cfg.get("compile.budget.wicer_eval", BudgetConfig().wicer_eval),
            wicer_max_iterations=cfg.get("compile.budget.wicer_max_iterations", BudgetConfig().wicer_max_iterations),
        )

        output_config = OutputConfig(
            format=cfg.get("compile.format", OutputConfig().format),
            flavor=cfg.get("compile.output.flavor", OutputConfig().flavor),
            frontmatter=frontmatter_defaults,
            rumdl_config=rumdl_config,
        )

        embedding_config = EmbeddingConfig(
            model=cfg.get("compile.embedding.model"),
            dim=cfg.get("compile.embedding.dim", 384),
        )

        return cls(
            source_dir=source_dir,
            output_dir=output_dir,
            image_mode=ImageMode(cfg.get("compile.image_mode", ImageMode.FILE)),
            chunk_size=cfg.get("compile.chunk_size", 512),
            online_sources=[],
            input_files=input_files,
            cache_dir=cfg.get("compile.cache_dir"),
            no_cache=cfg.get("compile.no_cache", False),
            workers=cfg.get("compile.workers", 1),
            mode=Mode(cfg.get("compile.mode", Mode.FULL)),
            force=cfg.get("compile.force", False),
            tier_state_path=cfg.get("compile.tier_state_path"),
            ignore_patterns=list(cfg.get("ignore.patterns", [])),
            respect_gitignore=cfg.get("compile.respect_gitignore", True),
            page_kind=cfg.get("compile.page_kind"),
            compiler=cfg.get("compile.compiler", Compiler.GRAPH),
            privacy_filter=cfg.get("trust.privacy_filter", True),
            llm=llm_config,
            budget=budget_config,
            output=output_config,
            embedding=embedding_config,
        )


__all__ = [
    "ALL_TIERS",
    "MODE_LEVELS",
    "TIER_DESCRIPTIONS",
    "TIER_LEVEL",
    "Compiler",
    "ImageMode",
    "Mode",
    "OutputFormat",
    "PipelineConfig",
    "Tier",
]
