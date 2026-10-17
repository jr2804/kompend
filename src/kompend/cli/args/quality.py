"""CLI arguments for the quality commands (lint, review, provenance, doctor,
status).
"""

from __future__ import annotations

from typing import Annotated

import typer

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

LintSemantic = Annotated[
    bool,
    typer.Option(
        "--semantic/--no-semantic",
        help="Verify missing topics against source chunks via the embedding model (needs compile.embedding.model).",
    ),
]

LintGapsFix = Annotated[
    bool,
    typer.Option(
        "--fix",
        help="Generate pages for source-covered missing topics (same as lint remediate).",
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

ReviewAction = Annotated[
    str,
    typer.Argument(help="Action: list | approve | reject | lock | unlock"),
]

ReviewSlug = Annotated[
    str | None,
    typer.Argument(help="Page slug (required for approve/reject/lock/unlock)."),
]

ReviewReason = Annotated[
    str | None,
    typer.Option("--reason", help="Rejection reason (for 'reject')."),
]

ReviewAll = Annotated[
    bool,
    typer.Option("--all", help="Approve every pending page."),
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

ProvenanceSlugs = Annotated[
    list[str] | None,
    typer.Argument(
        help="Page slug(s) or title(s) to trace. Omit to list all pages with their slugs."
    ),
]

ProvenanceWithContent = Annotated[
    bool,
    typer.Option("--with-content", help="Show cited content snippets."),
]

# ── log command ─────────────────────────────────────────────────────────

LogLast = Annotated[
    int,
    typer.Option(
        "--last",
        "-n",
        help="Maximum number of command/lint entries to include.",
    ),
]

LogWrite = Annotated[
    bool,
    typer.Option(
        "--write",
        help="Write the report to .kompend/log.md instead of printing it.",
    ),
]
