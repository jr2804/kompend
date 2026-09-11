"""CLI arguments for other commands (lint, serve, organize, capture, etc.)."""

from __future__ import annotations

from typing import Annotated

import typer

from knox.core.config import ExportMode
from knox.core.config.keys import EnvVar

# ── N1 lint command ────────────────────────────────────────────────────

LintFix = Annotated[
    bool,
    typer.Option(
        "--fix/--no-fix",
        help="Auto-fix issues where possible (default: on).",
    ),
]

LintFixDestructive = Annotated[
    bool,
    typer.Option(
        "--fix-destructive/--no-fix-destructive",
        help=(
            "Also apply destructive fixes (file deletion, e.g. orphan-attachments). Off by default — the post-compile lint and plain --fix never delete files."
        ),
    ),
]

LintLlm = Annotated[
    bool,
    typer.Option(
        "--llm/--no-llm",
        help="Enable LLM-powered lint rules (default). --no-llm runs structural rules only.",
    ),
]

LintRule = Annotated[
    list[str] | None,
    typer.Option(
        "--rule",
        help="Run only specific rule(s) (repeatable).",
    ),
]

LintHistoryLast = Annotated[
    int,
    typer.Option(
        "--last",
        "-n",
        help="Show the last N history entries.",
        min=1,
    ),
]

LintHistoryTrend = Annotated[
    bool,
    typer.Option(
        "--trend",
        help="Show a health-score trend sparkline.",
    ),
]

LintPruneKeep = Annotated[
    int | None,
    typer.Option(
        "--keep-last",
        help="Keep only the most recent N entries, removing the rest.",
        min=1,
    ),
]

LintPruneAgeDays = Annotated[
    int | None,
    typer.Option(
        "--max-age-days",
        help="Remove entries older than N days.",
        min=1,
    ),
]

LintDryRun = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Show what would be remediated without modifying the wiki.",
    ),
]

LintFormatCheck = Annotated[
    bool,
    typer.Option(
        "--check",
        help="Exit with code 1 if any formatting changes are needed (CI).",
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

SourceTags = Annotated[
    list[str] | None,
    typer.Option(
        "--tags",
        help=(
            "Source tags (repeatable; comma-separated within one option is AND, "
            "multiple --tags flags are OR). Blanks stripped from tag names; quote "
            "tags containing spaces to preserve them."
        ),
    ),
]

SourceAttrs = Annotated[
    list[str] | None,
    typer.Option(
        "--attr",
        help=("Source attributes as KEY=VALUE (repeatable). Multiple --attr flags are AND — each sets one key=value pair on the source."),
    ),
]

IngestTarget = Annotated[
    str,
    typer.Argument(
        help="URL or file path to ingest into the sources directory.",
    ),
]

# ── sources export ───────────────────────────────────────────────────────

ExportInline = Annotated[
    bool,
    typer.Option(
        "--inline/--no-inline",
        help="Embed images as data URIs (default: copy to <output stem>_assets/).",
    ),
]

ExportAnnotations = Annotated[
    ExportMode | None,
    typer.Option(
        "--annotations",
        help="Where extracted table/figure annotations go: off, inline, or asset (default: export.annotations config).",
    ),
]

ExportSummary = Annotated[
    ExportMode | None,
    typer.Option(
        "--summary",
        help="Where the document summary goes: off, inline, or asset (default: export.summary config).",
    ),
]

# ── N8 review queue ─────────────────────────────────────────────────────

ReviewMode = Annotated[
    bool | None,
    typer.Option(
        "--review/--no-review",
        help="Send generated pages to the review queue instead of the output dir. Default: off.",
        rich_help_panel="Output",
    ),
]

BypassReview = Annotated[
    bool,
    typer.Option(
        "--bypass-review",
        help="Skip review entirely (overrides --review).",
        rich_help_panel="Output",
    ),
]

# ── N4 provenance / Q21 trust / verification ────────────────────────────

Provenance = Annotated[
    bool | None,
    typer.Option(
        "--provenance/--no-provenance",
        help="Emit claim-level [source:42-58] citations in generated pages. Default: off.",
        rich_help_panel="Output",
    ),
]

Private = Annotated[
    bool | None,
    typer.Option(
        "--private/--no-private",
        help="Mark generated pages as private (excluded from default search). Default: off.",
        rich_help_panel="Output",
    ),
]

NoPrivacy = Annotated[
    bool | None,
    typer.Option(
        "--no-privacy/--privacy",
        help="Disable privacy redaction (secrets/PII patterns kept as-is). Default: filter on.",
        rich_help_panel="Output",
    ),
]

Verify = Annotated[
    bool | None,
    typer.Option(
        "--verify/--no-verify",
        help="Run grounding verification on each claim. Default: on; --no-verify skips.",
        rich_help_panel="Output",
    ),
]

# ── LLM configuration (Q12: failover chain) ─────────────────────────────

LlmDefaultModel = Annotated[
    str | None,
    typer.Option(
        "--llm-default-model",
        help=(
            "Primary model in provider/model format ('provider/model'), "
            "e.g. 'openai/gpt-4o' or 'ollama/llama3.2'. "
            "Overrides the config-file model/provider when set."
        ),
        envvar=EnvVar.LLM_DEFAULT_MODEL,
        rich_help_panel="LLM",
    ),
]

VlmModel = Annotated[
    str | None,
    typer.Option(
        "--vlm-model",
        help=(
            "Vision model in provider/model format ('provider/model') used to caption "
            "IMAGE chunks during resolution. When set, VLMEnricher is added to "
            "the resolver chain. Example: 'openai/gpt-4o'. Default: off."
        ),
        envvar=EnvVar.VLM_MODEL,
        rich_help_panel="LLM",
    ),
]

MemoryEnabled = Annotated[
    bool | None,
    typer.Option(
        "--memory/--no-memory",
        help=("Track node lifecycle (confidence, tiers, supersession, retention/decay) across compiles. Disabled skips the memory update pass. Default: on."),
        envvar=EnvVar.MEMORY_ENABLED,
        rich_help_panel="Pipeline",
    ),
]

LlmFallbackChain = Annotated[
    list[str] | None,
    typer.Option(
        "--llm-fallback-chain",
        help=(
            "Ordered fallback model(s) tried when the primary LLM fails "
            "(repeatable). Each value is a message-dict 'provider/model' "
            "string, e.g. 'openai/gpt-4o'. Empty means single-model mode. "
            "Env var value: comma-separated list."
        ),
        envvar=EnvVar.LLM_FALLBACK_CHAIN,
        rich_help_panel="LLM",
    ),
]

LlmFallbackMaxRetries = Annotated[
    int,
    typer.Option(
        "--llm-fallback-max-retries",
        help=("Transient-error retries before advancing to the next model in the failover chain. 2 means up to three attempts per model."),
        min=0,
        envvar=EnvVar.LLM_FALLBACK_MAX_RETRIES,
        rich_help_panel="LLM",
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
            "Roles: resolve, compile, enrich, query. Default: config llm.role_models."
        ),
        envvar=EnvVar.LLM_ROLE_MODEL,
        rich_help_panel="LLM",
    ),
]
