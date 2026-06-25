"""CLI arguments for other commands (lint, serve, organize, capture, etc.)."""

from __future__ import annotations

from typing import Annotated

import typer

from knox.core.pipeline_defs import Compiler

from .common import APP_NAME_UPPERCASE

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

# ── N5 web serve command ─────────────────────────────────────────────────

ServeHost = Annotated[
    str,
    typer.Option(
        "--host",
        help="Host to bind the web server to. Default: 127.0.0.1 (localhost only).",
    ),
]

ServePort = Annotated[
    int,
    typer.Option(
        "--port",
        "-p",
        help="Port to bind the web server to.",
        min=1024,
        max=65535,
    ),
]

ServeOpen = Annotated[
    bool,
    typer.Option(
        "--open",
        help="Open the web UI in the default browser automatically.",
    ),
]

# ── `organize` command surface ──────────────────────────────────────────

OrganizeMaxDepth = Annotated[
    int | None,
    typer.Option(
        "--max-depth",
        help="Maximum category tree depth (1-4). Env: KNOX_ORGANIZE_MAX_DEPTH.",
        min=1,
        max=4,
    ),
]

OrganizeMinSubcategories = Annotated[
    int | None,
    typer.Option(
        "--min-subcategories",
        help="Minimum sub-categories required to justify a new level. Env: KNOX_ORGANIZE_MIN_SUBCATEGORIES.",
        min=2,
    ),
]

OrganizeDryRun = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Show the proposed category tree without moving files or rewriting wiki links.",
    ),
]

OrganizeCollapseSingle = Annotated[
    bool | None,
    typer.Option(
        "--collapse-single-file/--no-collapse-single-file",
        help="Collapse single-file subcategories into the parent directory.",
    ),
]

# ── Q22 capture/learn/ingest commands ────────────────────────────────────

CaptureText = Annotated[
    str,
    typer.Argument(
        help="Text to extract knowledge items from.",
    ),
]

LearnText = Annotated[
    str,
    typer.Argument(
        help="Fact or snippet to store directly as a source file.",
    ),
]

LearnTags = Annotated[
    list[str] | None,
    typer.Option(
        "--tag",
        help="Tags to attach to the learned fact (repeatable).",
    ),
]

IngestTarget = Annotated[
    str,
    typer.Argument(
        help="URL or file path to ingest into the sources directory.",
    ),
]

# ── Q22 capture/learn/ingest commands (budget and LLM) ────────────────────

PromptBudget = Annotated[
    int,
    typer.Option(
        "--prompt-budget",
        help="Maximum total chars of source content per concept (N10 fair-share).",
        min=1,
    ),
]

BudgetGenSegment = Annotated[
    int | None,
    typer.Option(
        "--budget-gen-segment",
        help="Generation output max tokens per concept.",
        rich_help_panel="Advanced",
    ),
]

BudgetConcepts = Annotated[
    int | None,
    typer.Option(
        "--budget-concepts",
        help="Number of concepts per generation batch.",
        rich_help_panel="Advanced",
    ),
]

LlmTimeout = Annotated[
    float | None,
    typer.Option(
        "--llm-timeout",
        help="LLM call timeout in seconds.",
        rich_help_panel="Advanced",
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

NoUserTracking = Annotated[
    bool,
    typer.Option(
        "--no-user-tracking",
        help="Do not record the OS user (ingested_by / memory_by) in metadata. Env: KNOX_USER_TRACKING=false.",
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
    Compiler,
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

VlmModel = Annotated[
    str | None,
    typer.Option(
        "--vlm-model",
        help=(
            "Vision model in litellm format ('provider/model') used to caption "
            "IMAGE chunks during resolution. When set, VLMEnricher is added to "
            "the resolver chain. Example: 'openai/gpt-4o'. "
            f"Env: {APP_NAME_UPPERCASE}_VLM_MODEL."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_VLM_MODEL",
    ),
]

MemoryEnabled = Annotated[
    bool,
    typer.Option(
        "--memory/--no-memory",
        help=(
            "Track node lifecycle (confidence, tiers, supersession, retention/decay) "
            "across compiles. Disabled skips the memory update pass. "
            f"Env: {APP_NAME_UPPERCASE}_MEMORY_ENABLED."
        ),
        envvar=f"{APP_NAME_UPPERCASE}_MEMORY_ENABLED",
    ),
]

MemoryHalfLifeDays = Annotated[
    float,
    typer.Option(
        "--memory-half-life-days",
        help=(
            "Retention half-life in days for the memory decay curve. "
            "Nodes whose retention drops below the decay threshold are flagged as stale. "
            f"Env: {APP_NAME_UPPERCASE}_MEMORY_HALF_LIFE_DAYS."
        ),
        min=1.0,
        envvar=f"{APP_NAME_UPPERCASE}_MEMORY_HALF_LIFE_DAYS",
    ),
]

MemoryDecayThreshold = Annotated[
    float,
    typer.Option(
        "--memory-decay-threshold",
        help=(f"Retention value (0.0-1.0) below which a node is considered stale. Env: {APP_NAME_UPPERCASE}_MEMORY_DECAY_THRESHOLD."),
        min=0.0,
        max=1.0,
        envvar=f"{APP_NAME_UPPERCASE}_MEMORY_DECAY_THRESHOLD",
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
