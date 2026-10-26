"""Pipeline constants, tier definitions, and configuration.

This module re-exports the tier/compiler enums (defined in
:mod:`kompend.core.config`, next to the config fields that select them)
and defines :data:`PipelineConfig`, the dataclass consumed by
:class:`WikiPipeline` and its factory :func:`build_pipeline`.

Both the CLI (``kompend.cli.commands``) and the orchestrator (``kompend.core.pipeline``)
file is part of the public configuration contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kompend.core.config import (
    BudgetConfig,
    Compiler,
    DisplayConfig,
    EmbeddingConfig,
    ImageMode,
    LLMConfig,
    OutputConfig,
    OutputFormat,
    ParsecraftJudge,
    ParsecraftPreference,
    ResolvedConfig,
    Tier,
    TierInfo,
)
from kompend.core.config._configs import DecisionConfig, SearchConfig
from kompend.core.paths import VaultPaths
from kompend.preprocessing.parsers._registry import ParserBuildOptions

__all__ = [
    "Compiler",
    "DisplayConfig",
    "ImageMode",
    "OutputFormat",
    "PipelineConfig",
    "Tier",
    "TierInfo",
]

# Compiler/Tier/TierInfo live in kompend.core.config (next to the config
# fields that select them); they are re-exported here for the pipeline
# module's long-standing import surface.


@dataclass
class PipelineConfig:
    """Configuration for the kompend pipeline.

    This dataclass is the single shape consumed by :class:`WikiPipeline` and
    its factory :func:`build_pipeline`. Defaults are intentionally conservative
    so the pipeline can run on a developer laptop without any configuration.

    Directory layout::

        base_dir/
        ├── .kompend/         ← internal state (cache, tier state, vectors, etc.)
        ├── sources/        ← source_dir: user-owned inputs
        ├── wiki/           ← output_dir: generated pages
        ├── index.md        ← entry point (TOC)
        └── overview.md      ← LLM-generated overview (opt-in)

    When ``base_dir`` is not set explicitly, it defaults to ``output_dir.parent``.
    All ``.kompend`` paths resolve relative to ``base_dir``.
    """

    source_dir: Path
    output_dir: Path
    llm: LLMConfig
    embedding: EmbeddingConfig
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    base_dir: Path | None = None  # None → output_dir.parent; .kompend lives here
    image_mode: ImageMode = ImageMode.FILE
    vlm_min_image_bytes: int = 2048
    vlm_min_image_dimension: int = 64
    online_sources: list[str] = field(default_factory=list)
    input_files: list[Path] | None = None  # explicit file list; None = auto-discover from source_dir
    cache_dir: Path | None = None  # None = base_dir/.kompend/cache
    no_cache: bool = False  # force re-parse even if cached
    workers: int = 1  # parallel parse+chunk workers
    start_at_tier: Tier = Tier.INDEX  # pipeline entry point (T0 by default)
    stop_at_tier: Tier = Tier.OUTPUT  # pipeline exit point (T4 by default; T1 for ingest)
    force: bool = False  # bypass persistent tier state and re-process everything
    cache_ignore_model: bool = False  # cache keys drop the model (compile.cache_ignore_model)
    tier_state_file: Path | None = None  # None = base_dir/.kompend/compile_state.json
    ignore_patterns: list[str] | None = None  # glob patterns to exclude
    respect_gitignore: bool = True  # respect .gitignore in source dir
    recursive: bool = True  # scan sources_dir subdirectories (paths.recursive)
    compiler: Compiler = Compiler.GRAPH  # C4: compiler backend selection
    privacy_filter: bool = True  # redact PII/secrets from parsed content
    memory_enabled: bool = True  # Track node confidence/tiers/supersession/decay
    skills: dict[str, str] = field(default_factory=dict)  # user-defined skill instructions (PydanticAI)
    extraction_categories: dict[str, str] = field(default_factory=dict)  # compile.extraction.categories (extra page kinds)
    memory_half_life_days: float = 90.0  # Retention half-life in days
    memory_decay_threshold: float = 0.3  # Retention below this = stale
    notes_file: Path | None = None  # User notes file path (None = no notes)
    images_scale: float = 200.0 / 72.0  # Image resolution multiplier (200 DPI base)
    parsecraft_judge: ParsecraftJudge = ParsecraftJudge.NONE  # judge tier for the parsecraft backend
    parsecraft_judge_model: str = ""  # provider-specific model token (e.g. parable-tinyjev for ollama)
    parsecraft_preference: ParsecraftPreference = ParsecraftPreference.BALANCED  # judge ranking axis
    #: Parser-construction inputs, built by :meth:`ParserBuildOptions.from_config`
    #: in :meth:`from_resolved`. ``build_pipeline`` reads this rather than
    #: reassembling the fields, so the pipeline, the CLI and the MCP server all
    #: hand the router an identical option set — they share one parse cache
    #: whose key does not cover these fields.
    parser_options: ParserBuildOptions = field(default_factory=ParserBuildOptions)
    source_tags: list[str] = field(default_factory=list)  # CLI-provided source tags
    search: SearchConfig = field(default_factory=SearchConfig)  # search config (contextual enrichment gate lives here)
    attachments_ingest_adopted: bool = False  # attachments.ingest_adopted: global gate — manifest flag AND this gate decides ingestion
    providers: dict[str, dict] = field(default_factory=dict)  # user-defined providers from [providers] section

    def resolved_base_dir(self) -> Path:
        """Return the effective vault root as an absolute, resolved path.

        Downstream code must never see a CWD-relative vault base: a relative
        ``base_dir``/``output_dir`` would otherwise be re-resolved against
        whatever process CWD happens to be active (e.g. an MCP server, a test
        runner) and scatter ``.kompend/`` state into the wrong directory.
        """
        base = self.base_dir if self.base_dir is not None else self.output_dir.parent
        return base.expanduser().resolve()

    def kompend_dir(self) -> Path:
        """Return the ``.kompend`` directory (always inside base_dir)."""
        return self.paths.kompend_dir

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
            for name in cls.model_fields:
                val = cfg.get(f"{section}.{name}")
                if val is not None:
                    out[name] = val
            return out

        kompend = cfg.kompend
        default_model = kompend.compile.llm.default_model
        if not default_model or "${" in str(default_model):
            raise ValueError(
                "compile.llm.default_model is not set — "
                "set ${KOMP_LLM_DEFAULT_MODEL} in the environment, "
                "add it to the TOML config file, or pass "
                "--set compile.llm.default_model=... on the command line."
            )
        llm_kwargs = _kwargs_for(LLMConfig, "compile.llm")
        llm_config = LLMConfig(**llm_kwargs)

        budget_config = BudgetConfig(**_kwargs_for(BudgetConfig, "compile.budget"))

        output_config = OutputConfig(**_kwargs_for(OutputConfig, "compile.output"))
        embedding_config = EmbeddingConfig(**_kwargs_for(EmbeddingConfig, "compile.embedding"))
        display_config = DisplayConfig(**_kwargs_for(DisplayConfig, "display"))

        # Resolve notes file path relative to wiki_dir
        notes_file_name = kompend.notes.file
        wiki_dir = kompend.paths.wiki_dir
        notes_file: Path | None = None
        if notes_file_name:
            base_dir = Path(wiki_dir) if wiki_dir else output_dir.parent
            notes_file = base_dir / notes_file_name

        providers_raw = cfg.get("providers", {})
        # Sections reach this point as raw config JSON; keep only the mappings
        # (a non-mapping section is a config error the loader already reports)
        # so the field's declared shape is what callers see.
        providers: dict[str, dict[str, Any]] = (
            {name: dict(section) for name, section in providers_raw.items() if isinstance(section, dict)} if isinstance(providers_raw, dict) else {}
        )

        pipeline_config = cls(
            source_dir=source_dir,
            output_dir=output_dir,
            image_mode=ImageMode(kompend.compile.image_mode),
            vlm_min_image_bytes=kompend.compile.vlm_min_image_bytes,
            vlm_min_image_dimension=kompend.compile.vlm_min_image_dimension,
            online_sources=[],
            input_files=input_files,
            cache_dir=Path(kompend.paths.cache_dir) if kompend.paths.cache_dir else None,
            no_cache=kompend.compile.no_cache,
            workers=kompend.compile.workers,
            start_at_tier=Tier(kompend.compile.start_at_tier),
            stop_at_tier=Tier(kompend.compile.stop_at_tier),
            force=kompend.compile.force,
            cache_ignore_model=kompend.compile.cache_ignore_model,
            tier_state_file=Path(kompend.compile.tier_state_file) if kompend.compile.tier_state_file else None,
            ignore_patterns=list(kompend.ignore.patterns),
            respect_gitignore=kompend.compile.respect_gitignore,
            recursive=kompend.paths.recursive,
            compiler=Compiler(kompend.compile.compiler),
            privacy_filter=kompend.trust.privacy_filter,
            parsecraft_judge=kompend.ingest.parsecraft_judge,
            parsecraft_judge_model=kompend.ingest.parsecraft_judge_model,
            parsecraft_preference=kompend.ingest.parsecraft_preference,
            memory_enabled=kompend.compile.memory_enabled,
            memory_half_life_days=kompend.compile.memory_half_life_days,
            memory_decay_threshold=kompend.compile.memory_decay_threshold,
            notes_file=notes_file,
            llm=llm_config,
            decision=kompend.decision,
            budget=budget_config,
            search=SearchConfig(**_kwargs_for(SearchConfig, "search")),
            output=output_config,
            embedding=embedding_config,
            display=display_config,
            skills={name: skill.instructions for name, skill in kompend.compile.skills.items()},
            extraction_categories=dict(kompend.compile.extraction.categories),
            source_tags=list(kompend.compile.source_tags),
            attachments_ingest_adopted=bool(kompend.attachments.ingest_adopted),
            providers=providers,
        )
        # Assigned after construction because ``paths`` needs a bound instance;
        # re-deriving the vault layout here would duplicate it.
        pipeline_config.parser_options = ParserBuildOptions.from_config(cfg, images_dir=pipeline_config.paths.images_dir)
        return pipeline_config

    # Sub-configs (additive for backward compatibility)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
