"""CLI arguments, options, and defaults for KNOX.

All Typer ``Argument`` and ``Option`` declarations live here as
``Annotated`` types. Command files import the types and apply them as
plain annotations — this keeps B008 happy and centralises the surface
area that users see in ``--help``.
"""

from __future__ import annotations

from typing import Annotated

import typer

# Application name used for environment variable prefixes.
APP_NAME_UPPERCASE = "KNOX"

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_SOURCES_DIR: str = "sources"
DEFAULT_OUTPUT_DIR: str = "wiki"
DEFAULT_COMPILE_FORMAT: str = "markdown"
DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_IMAGE_MODE: str = "file"
DEFAULT_COMPILE_MODE: str = "full"
DEFAULT_PROMPT_BUDGET: int = 200_000
DEFAULT_LLM_FALLBACK_MAX_RETRIES: int = 2


# ── Shared options (used by multiple commands) ──────────────────────────

Recursive = Annotated[
    bool,
    typer.Option(
        "--recursive/--no-recursive",
        "-r/-R",
        help="Search source directories recursively.",
        envvar=f"{APP_NAME_UPPERCASE}_RECURSIVE",
    ),
]

SourcesDir = Annotated[
    str | None,
    typer.Option(
        "--sources-dir",
        "-s",
        help="Input sources directory. Files not already here are copied in. "
        f"Default: '{DEFAULT_SOURCES_DIR}' in the current working directory. "
        f"Env: {APP_NAME_UPPERCASE}_SOURCES_DIR.",
    ),
]

OutputDir = Annotated[
    str | None,
    typer.Option(
        "--output-dir",
        "-o",
        help=f"Output directory for compiled wiki. Default: '{DEFAULT_OUTPUT_DIR}' in the current working directory. Env: {APP_NAME_UPPERCASE}_OUTPUT_DIR.",
    ),
]


# ── `parse` command surface ─────────────────────────────────────────────

ParseFiles = Annotated[
    list[str],
    typer.Argument(
        help="Files, glob patterns, or directories to parse.",
    ),
]

PrintElements = Annotated[
    bool,
    typer.Option(
        "--elements",
        "-e",
        help="Print extracted elements.",
    ),
]


# ── `compile` command surface ───────────────────────────────────────────

CompileFiles = Annotated[
    list[str],
    typer.Argument(
        help="Files, glob patterns, or directories to compile. If only directories are given, all supported files inside are used.",
    ),
]

OnlineUrls = Annotated[
    list[str] | None,
    typer.Option(
        "--url",
        help="URLs to download before processing (repeatable).",
    ),
]

TargetFormat = Annotated[
    str,
    typer.Option(
        "--format",
        "-f",
        help="Output format: markdown, json, typst.",
    ),
]

ChunkSize = Annotated[
    int,
    typer.Option(
        "--chunk-size",
        help="Chunk size in tokens.",
        min=1,
    ),
]

ImageMode = Annotated[
    str,
    typer.Option(
        "--image-mode",
        help="Image handling: file, base64.",
    ),
]


# ── Shared flags (used by multiple commands) ────────────────────────────

Verbose = Annotated[
    bool,
    typer.Option(
        "--verbose",
        help="Show detailed output.",
    ),
]

JsonOutput = Annotated[
    bool,
    typer.Option(
        "--json",
        help="Output in JSON format for machine readability.",
    ),
]

Ignore = Annotated[
    list[str] | None,
    typer.Option(
        "--ignore",
        help="Ignore pattern (glob-style, repeatable).",
    ),
]

Progress = Annotated[
    bool | None,
    typer.Option(
        "--progress/--no-progress",
        help="Show rich progress bars (auto-detected for TTY).",
    ),
]

IndexStore = Annotated[
    bool,
    typer.Option(
        "--index",
        help="Index compiled wiki in zvec for semantic search.",
    ),
]

EmbedModel = Annotated[
    str | None,
    typer.Option(
        "--embed-model",
        help="Embedding model for zvec indexing.",
    ),
]

# ── Q20 tiered compilation ──────────────────────────────────────────────

CompileMode = Annotated[
    str,
    typer.Option(
        "--mode",
        help="Compilation mode: full | index-only | resolve-only | output-only. Tiers below the selected mode are skipped (cached state is reused).",
    ),
]

NoCacheFlag = Annotated[
    bool,
    typer.Option(
        "--no-cache",
        help="Bypass the parse cache and re-parse every source.",
    ),
]

Workers = Annotated[
    int,
    typer.Option(
        "--workers",
        "-j",
        help="Number of parallel parse workers (1 = sequential).",
        min=1,
    ),
]

ForceTier = Annotated[
    bool,
    typer.Option(
        "--force",
        help="Re-process every source regardless of persisted tier state.",
    ),
]

ListTiers = Annotated[
    bool,
    typer.Option(
        "--list-tiers",
        help="Show the available pipeline tiers and their current state, then exit.",
    ),
]

# ── N3 two-phase compile ────────────────────────────────────────────────

TwoPhase = Annotated[
    bool,
    typer.Option(
        "--two-phase/--single-pass",
        help="Use two-phase compile (extract all, then generate). Default: two-phase.",
    ),
]

# ── N10 per-concept budget ──────────────────────────────────────────────

PromptBudget = Annotated[
    int,
    typer.Option(
        "--prompt-budget",
        help="Maximum total chars of source content per concept (N10 fair-share).",
        min=1,
    ),
]

# ── N8 review queue ─────────────────────────────────────────────────────

ReviewMode = Annotated[
    bool,
    typer.Option(
        "--review",
        help="Send generated pages to the review queue instead of the output dir.",
    ),
]

BypassReview = Annotated[
    bool,
    typer.Option(
        "--bypass-review",
        help="Skip review entirely (overrides --review).",
    ),
]

# ── N4 provenance / Q21 trust / verification ────────────────────────────

Provenance = Annotated[
    bool,
    typer.Option(
        "--provenance",
        help="Emit claim-level ^[source:42-58] citations in generated pages.",
    ),
]

Private = Annotated[
    bool,
    typer.Option(
        "--private",
        help="Mark generated pages as private (excluded from default search).",
    ),
]

NoPrivacy = Annotated[
    bool,
    typer.Option(
        "--no-privacy",
        help="Disable privacy redaction (secrets/PII patterns kept as-is).",
    ),
]

Verify = Annotated[
    bool,
    typer.Option(
        "--verify",
        help="Run grounding verification on each claim (opt-in, costly).",
    ),
]

IncludePrivate = Annotated[
    bool,
    typer.Option(
        "--include-private",
        help="Include private pages in search/queries.",
    ),
]


# ── Output flavor (N13: multi-flavor Markdown via rumdl) ────────────────

OutputFlavor = Annotated[
    str | None,
    typer.Option(
        "--flavor",
        help=(
            "Markdown output flavor: obsidian, gfm, mkdocs, mdx, quarto, myst, "
            "standard, pandoc, kramdown, azure_devops. "
            "Auto-detected from output dir when not set."
        ),
    ),
]


# ── Frontmatter (Q18: custom frontmatter fields) ────────────────────────

Frontmatter = Annotated[
    list[str] | None,
    typer.Option(
        "--frontmatter",
        help=("Frontmatter field in key=value format (repeatable). E.g. --frontmatter language=en --frontmatter domain=ml"),
    ),
]

SynthesizeFlag = Annotated[
    bool,
    typer.Option(
        "--synthesize",
        help="Also generate an LLM-synthesized overview.md page.",
    ),
]


# ── LLM configuration (Q12: failover chain) ─────────────────────────────

LlmDefaultModel = Annotated[
    str | None,
    typer.Option(
        "--llm-default-model",
        help=(
            "Primary model in litellm format ('provider/model'), "
            "e.g. 'openai/gpt-4o' or 'ollama/llama3.2'. "
            "Overrides the config-file model/provider when set. "
            f"Env: {APP_NAME_UPPERCASE}_LLM_DEFAULT_MODEL."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_LLM_DEFAULT_MODEL",
    ),
]

LlmFallbackChain = Annotated[
    list[str] | None,
    typer.Option(
        "--llm-fallback-chain",
        help=(
            "Ordered fallback model(s) tried when the primary LLM fails "
            "(repeatable). Each value is a litellm-style 'provider/model' "
            "string, e.g. 'openai/gpt-4o'. Empty means single-model mode. "
            f"Env: {APP_NAME_UPPERCASE}_LLM_FALLBACK_CHAIN (comma-separated)."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_LLM_FALLBACK_CHAIN",
    ),
]

LlmFallbackMaxRetries = Annotated[
    int,
    typer.Option(
        "--llm-fallback-max-retries",
        help=("Transient-error retries before advancing to the next model in the failover chain. 2 means up to three attempts per model."),
        min=0,
        envvar=f"{APP_NAME_UPPERCASE}_LLM_FALLBACK_MAX_RETRIES",
    ),
]

LlmRoleModel = Annotated[
    list[str] | None,
    typer.Option(
        "--llm-role-model",
        help=(
            "Override primary model for a pipeline role (repeatable). "
            "Format: 'role=provider/model', e.g. "
            "'resolve=openai/gpt-4o-mini' "
            "'compile=anthropic/claude-3.5-sonnet'. "
            "Roles: resolve, compile, enrich, query."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_LLM_ROLE_MODEL",
    ),
]

LlmContextBudget = Annotated[
    int | None,
    typer.Option(
        "--llm-context-budget",
        help=(
            "Max characters of page content sent to LLM in a single lint call. "
            "When content exceeds this, a warning is logged and content is truncated. "
            "Default: 100000 (roughly 25000 tokens). "
            f"Env: {APP_NAME_UPPERCASE}_LLM_CONTEXT_BUDGET."
        ),
        min=1000,
        envvar=f"{APP_NAME_UPPERCASE}_LLM_CONTEXT_BUDGET",
    ),
]

# ── N9 schema-driven compilation ────────────────────────────────────────

KindOption = Annotated[
    str | None,
    typer.Option(
        "--kind",
        help=(
            "Page kind for schema-driven compilation. "
            "Built-in kinds: concept, entity, comparison, overview, custom. "
            "When set, pages are compiled and validated against the kind's schema. "
            f"Env: {APP_NAME_UPPERCASE}_KIND."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_KIND",
    ),
]

CompilerChoice = Annotated[
    str,
    typer.Option(
        "--compiler",
        help=(
            "Compiler backend: graph (default, two-phase extract-then-generate) "
            "or wicer (iterative compile-evaluate-refine loop). "
            f"Env: {APP_NAME_UPPERCASE}_COMPILER."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_COMPILER",
    ),
]
DEFAULT_COMPILER: str = "graph"


# ── N1 lint command ────────────────────────────────────────────────────

LintFix = Annotated[
    bool,
    typer.Option(
        "--fix",
        help="Auto-fix structural issues where possible.",
    ),
]

LintLlm = Annotated[
    bool,
    typer.Option(
        "--llm",
        help="Enable LLM-powered lint rules (slow, expensive).",
    ),
]

LintRule = Annotated[
    list[str] | None,
    typer.Option(
        "--rule",
        help="Run only specific rule(s) (repeatable).",
    ),
]

LintThreshold = Annotated[
    int | None,
    typer.Option(
        "--threshold",
        help="Exit with error if health score is below this value (0-100).",
        min=0,
        max=100,
    ),
]

LintOutputDir = Annotated[
    str | None,
    typer.Option(
        "--output-dir",
        "-o",
        help="Compiled wiki directory to lint. Default: 'wiki' in CWD.",
    ),
]

LintSourcesDir = Annotated[
    str | None,
    typer.Option(
        "--sources-dir",
        "-s",
        help="Sources directory (for file-based checks). Default: 'sources' in CWD.",
    ),
]

# ── N1 compile --lint integration ───────────────────────────────────────

CompileLint = Annotated[
    bool,
    typer.Option(
        "--lint",
        help="Run lint after compilation and report quality issues.",
    ),
]

# ── Q15 config file path ────────────────────────────────────────────────

ConfigPath = Annotated[
    str | None,
    typer.Option(
        "--config",
        "-c",
        help=(
            f"Path to configuration file. Default: auto-discover './knox.toml' (walking up) then ~/.config/knox/config.toml. Env: {APP_NAME_UPPERCASE}_CONFIG."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_CONFIG",
    ),
]

# ── Q17 query command ──────────────────────────────────────────────────

QueryText = Annotated[
    str,
    typer.Argument(
        help="Natural language question to ask against the compiled wiki.",
    ),
]

QueryTopK = Annotated[
    int,
    typer.Option(
        "--top-k",
        help="Number of context chunks to retrieve (default: 5).",
        min=1,
        max=50,
    ),
]

QueryRaw = Annotated[
    bool,
    typer.Option(
        "--raw",
        help="Return raw context chunks without LLM synthesis.",
    ),
]

QueryFormat = Annotated[
    str,
    typer.Option(
        "--format",
        help="Output format: text (default) or json.",
    ),
]
