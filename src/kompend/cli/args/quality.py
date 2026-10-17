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
