"""Pipeline constants, tier definitions, and configuration.

This module is the single source of truth for:

- :class:`Tier` — the 5 pipeline tier identifiers
- :class:`TierInfo` — metadata per tier (level, description)
- :meth:`Tier.all_tiers` — ordered tuple of every tier's metadata
- :data:`PipelineConfig` — dataclass holding the full pipeline configuration

Both the CLI (``knox.cli.commands``) and the orchestrator (``knox.core.pipeline``)
file is part of the public configuration contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any

from knox.core.config import (
    BudgetConfig,
    DisplayConfig,
    EmbeddingConfig,
    ImageMode,
    LLMConfig,
    OutputConfig,
    OutputFormat,
    PdfConfig,
    ResolvedConfig,
)
from knox.core.config.keys import (
    COMPILE_CACHE_DIR,
    COMPILE_COMPILER,
    COMPILE_FORCE,
    COMPILE_IMAGE_MODE,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_MEMORY_DECAY_THRESHOLD,
    COMPILE_MEMORY_ENABLED,
    COMPILE_MEMORY_HALF_LIFE_DAYS,
    COMPILE_NO_CACHE,
    COMPILE_RESPECT_GITIGNORE,
    COMPILE_SKILLS,
    COMPILE_SOURCE_TAGS,
    COMPILE_START_AT_TIER,
    COMPILE_STOP_AT_TIER,
    COMPILE_TIER_STATE_FILE,
    COMPILE_VLM_MIN_IMAGE_BYTES,
    COMPILE_VLM_MIN_IMAGE_DIMENSION,
    COMPILE_WORKERS,
    INGEST_PARSER,
    PATHS_WIKI_DIR,
    TRUST_PRIVACY_FILTER,
)
from knox.core.paths import VaultPaths

__all__ = [
    "Compiler",
    "DisplayConfig",
    "ImageMode",
    "OutputFormat",
    "PipelineConfig",
    "Tier",
    "TierInfo",
]


class Compiler(StrEnum):
    """Compiler backend selection.

    - ``GRAPH``: GraphCompiler (heading-based grouping, default)
    - ``WICER``: WiCERCompiler (iterative compile-evaluate-refine loop)
    """

    GRAPH = "graph"
    WICER = "wicer"


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

    @classmethod
    @cache
    def all_tiers(cls) -> tuple[TierInfo, ...]:
        """Return metadata for every tier in execution order."""
        return (
            TierInfo(
                Tier.INDEX,
                0,
                "Tier 0: index sources — record local files and download online URLs. Inputs: source_dir, online_sources. Outputs: tier_state entries.",
            ),
            TierInfo(
                Tier.CHUNK,
                1,
                "Tier 1: parse documents and split them into chunks. Inputs: indexed source files. Outputs: list of chunks (cached on disk).",
            ),
            TierInfo(
                Tier.RESOLVE,
                2,
                "Tier 2: resolve chunks — extract entities/links (LLM-dependent). Inputs: chunks. Outputs: resolved chunks with entities/relations.",
            ),
            TierInfo(
                Tier.COMPILE,
                3,
                "Tier 3: compile resolved chunks into a knowledge graph (CompiledWiki). Inputs: resolved chunks. Outputs: CompiledWiki snapshot.",
            ),
            TierInfo(
                Tier.OUTPUT,
                4,
                "Tier 4: render the compiled wiki to the output directory. Inputs: CompiledWiki snapshot. Outputs: rendered files (markdown/json/typst).",
            ),
        )

    @classmethod
    def level_of(cls, tier: Tier) -> int:
        """Return the numeric level of a tier."""
        return cls._info_by_tier()[tier].level

    @classmethod
    def description_of(cls, tier: Tier) -> str:
        """Return the human-readable description of a tier."""
        return cls._info_by_tier()[tier].description

    @classmethod
    @cache
    def _info_by_tier(cls) -> dict[Tier, TierInfo]:
        """Return a mapping from tier to its metadata for O(1) lookup."""
        return {ti.tier: ti for ti in cls.all_tiers()}


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
    skills: dict[str, str] = field(default_factory=dict)  # user-defined skill instructions (PydanticAI)
    memory_half_life_days: float = 90.0  # Retention half-life in days
    memory_decay_threshold: float = 0.3  # Retention below this = stale
    notes_file: Path | None = None  # User notes file path (None = no notes)
    images_scale: float = 200.0 / 72.0  # Docling image resolution multiplier (200 DPI base)
    parser: str = "liteparse"  # ingestion backend name (see PARSER_BACKENDS)
    source_tags: list[str] = field(default_factory=list)  # CLI-provided source tags
    providers: dict[str, dict] = field(default_factory=dict)  # user-defined providers from [providers] section

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

    @classmethod
    def from_resolved(cls, cfg: ResolvedConfig, *, source_dir: Path, output_dir: Path, input_files: list[Path] | None = None) -> PipelineConfig:
        """Build PipelineConfig from a fully resolved config dict."""

        def _kwargs_for(cls: Any, section: str) -> dict[str, Any]:
            """Build constructor kwargs for a pydantic model from a config section.

            Only includes keys present (non-None) in the resolved config.
            Pydantic uses field defaults for absent keys and handles type
            coercion (str-to-enum, str-to-Path) during construction.
            """
            out: dict[str, Any] = {}
            for name in cls.model_fields:  # type: ignore[union-attr]
                val = cfg.get(f"{section}.{name}")
                if val is not None:
                    out[name] = val
            return out

        default_model = cfg.get(COMPILE_LLM_DEFAULT_MODEL)
        if not default_model or "${" in str(default_model):
            raise ValueError(
                f"{COMPILE_LLM_DEFAULT_MODEL} is not set — "
                f"set ${{KNOX_LLM_DEFAULT_MODEL}} in the environment, "
                f"add it to the TOML config file, or pass "
                f"--set {COMPILE_LLM_DEFAULT_MODEL}=... on the command line."
            )
        llm_kwargs = _kwargs_for(LLMConfig, "compile.llm")
        # Sanitize role_models values (pydantic handles enum/Path coercion)
        role_models_raw = llm_kwargs.pop("role_models", {})
        if isinstance(role_models_raw, dict):
            llm_kwargs["role_models"] = {k: str(v).strip() for k, v in role_models_raw.items() if str(v).strip()}
        else:
            llm_kwargs["role_models"] = {}
        llm_config = LLMConfig(**llm_kwargs)

        budget_config = BudgetConfig(**_kwargs_for(BudgetConfig, "compile.budget"))

        output_config = OutputConfig(**_kwargs_for(OutputConfig, "compile.output"))
        embedding_config = EmbeddingConfig(**_kwargs_for(EmbeddingConfig, "compile.embedding"))
        pdf_config = PdfConfig(**_kwargs_for(PdfConfig, "compile.pdf"))
        display_config = DisplayConfig(**_kwargs_for(DisplayConfig, "display"))

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
            tier_state_file=cfg.get(COMPILE_TIER_STATE_FILE),
            ignore_patterns=list(cfg.get("ignore.patterns", [])),
            respect_gitignore=cfg.get(COMPILE_RESPECT_GITIGNORE, True),
            compiler=cfg.get(COMPILE_COMPILER, Compiler.GRAPH),
            privacy_filter=cfg.get(TRUST_PRIVACY_FILTER, True),
            parser=cfg.get(INGEST_PARSER, "liteparse"),
            memory_enabled=cfg.get(COMPILE_MEMORY_ENABLED, True),
            memory_half_life_days=cfg.get(COMPILE_MEMORY_HALF_LIFE_DAYS, 90.0),
            memory_decay_threshold=cfg.get(COMPILE_MEMORY_DECAY_THRESHOLD, 0.3),
            notes_file=notes_file,
            llm=llm_config,
            budget=budget_config,
            output=output_config,
            embedding=embedding_config,
            pdf=pdf_config,
            display=display_config,
            skills={name: skill.get("instructions", "") for name, skill in (cfg.get(COMPILE_SKILLS) or {}).items()},
            source_tags=cfg.get(COMPILE_SOURCE_TAGS, []),
            providers=cfg.get("providers", {}),
        )

    # Sub-configs (additive for backward compatibility)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pdf: PdfConfig = field(default_factory=PdfConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


@dataclass(frozen=True)
class TierInfo:
    """Metadata for a single pipeline tier."""

    tier: Tier
    level: int
    description: str
