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
from typing import Any, Literal


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
    ``--llm-default-model`` CLI flag and ``KNOX_LLM_DEFAULT_MODEL`` env var.
    When ``None`` (default), the model is constructed from ``llm_provider``
    and ``llm_model`` as before.
    """
    online_sources: list[str] = field(default_factory=list)
    input_files: list[Path] | None = None  # explicit file list; None = auto-discover from source_dir
    cache_dir: Path | None = None  # None = .knox/cache inside output_dir
    no_cache: bool = False  # force re-parse even if cached
    workers: int = 1  # parallel parse+chunk workers
    mode: str = Mode.FULL  # full | index-only | resolve-only | output-only
    force: bool = False  # bypass persistent tier state and re-process everything
    tier_state_path: Path | None = None  # None = .knox/compile_state.json inside output_dir
    ignore_patterns: list[str] | None = None  # glob patterns to exclude
    respect_gitignore: bool = True  # respect .gitignore in source dir
    flavor: str | None = None  # Markdown flavor (obsidian, gfm, mkdocs, etc.); None = auto-detect
    frontmatter_defaults: dict[str, Any] | None = None  # default frontmatter fields for all pages (Q18)
    page_kind: str | None = None  # Page kind for N9 schema-driven compilation (concept, entity, comparison, overview, custom); None = auto-detect
    rumdl_config: Path | None = None  # path to custom .rumdl.toml config
    compiler: Literal["graph", "wicer"] = "graph"  # C4: compiler backend selection
    privacy_filter: bool = True  # redact PII/secrets from parsed content


__all__ = [
    "ALL_TIERS",
    "MODE_LEVELS",
    "TIER_DESCRIPTIONS",
    "TIER_LEVEL",
    "Mode",
    "PipelineConfig",
    "Tier",
]
