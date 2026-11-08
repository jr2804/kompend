"""CLI arguments for the quality commands (lint, review, provenance, doctor,
status).
"""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from kompend.cli.args.common import OUTPUT_GROUP, in_range

# ── N1 lint command ────────────────────────────────────────────────────

LintFix = Annotated[
    bool,
    Parameter(
        name="--fix",
        help="Auto-fix issues where possible (default: on).",
    ),
]

LintFixDestructive = Annotated[
    bool,
    Parameter(
        name="--fix-destructive",
        help=(
            "Also apply destructive fixes (file deletion, e.g. orphan-attachments). Off by default — the post-compile lint and plain --fix never delete files."
        ),
        negative="",
    ),
]

LintVerbose = Annotated[
    bool,
    Parameter(
        name="--verbose-lint",
        help="Print every lint finding (default: a bounded summary; the full report is always written to .kompend/lint-log/).",
        negative="",
    ),
]

LintQuiet = Annotated[
    bool,
    Parameter(
        name="--quiet-lint",
        help="Print only the lint verdict and the path of the full report.",
        negative="",
    ),
]

LintLlm = Annotated[
    bool,
    Parameter(
        name="--llm",
        help="Enable LLM-powered lint rules (default). --no-llm runs structural rules only.",
    ),
]

LintRule = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        name="--rule",
        help="Run only specific rule(s) (repeatable).",
    ),
]

LintHistoryLast = Annotated[
    int,
    Parameter(
        name=("--last", "-n"),
        help="Show the last N history entries.",
        validator=in_range(1),
    ),
]

LintShowRun = Annotated[
    str | None,
    Parameter(
        name="--run",
        help="Report stamp, e.g. 2026-10-24T01-42-11 (default: the newest).",
    ),
]

LintHistoryTrend = Annotated[
    bool,
    Parameter(
        name="--trend",
        help="Show a health-score trend sparkline.",
        negative="",
    ),
]

LintPruneKeep = Annotated[
    int | None,
    Parameter(
        name="--keep-last",
        help="Keep only the most recent N entries, removing the rest.",
        validator=in_range(1),
    ),
]

LintPruneAgeDays = Annotated[
    int | None,
    Parameter(
        name="--max-age-days",
        help="Remove entries older than N days.",
        validator=in_range(1),
    ),
]

LintDryRun = Annotated[
    bool,
    Parameter(
        name="--dry-run",
        help="Show what would be remediated without modifying the wiki.",
        negative="",
    ),
]

LintFormatCheck = Annotated[
    bool,
    Parameter(
        name="--check",
        help="Exit with code 1 if any formatting changes are needed (CI).",
        negative="",
    ),
]

LintSemantic = Annotated[
    bool,
    Parameter(
        name="--semantic",
        help="Verify missing topics against source chunks via the embedding model (needs compile.embedding.model).",
    ),
]

LintGapsFix = Annotated[
    bool,
    Parameter(
        name="--fix",
        help="Generate pages for source-covered missing topics (same as lint remediate).",
        negative="",
    ),
]

# ── N8 review queue ─────────────────────────────────────────────────────

ReviewMode = Annotated[
    bool | None,
    Parameter(
        name="--review",
        help="Send generated pages to the review queue instead of the output dir. Default: off.",
        group=OUTPUT_GROUP,
    ),
]

BypassReview = Annotated[
    bool,
    Parameter(
        name="--bypass-review",
        help="Skip review entirely (overrides --review).",
        group=OUTPUT_GROUP,
        negative="",
    ),
]

ReviewAction = Annotated[
    str,
    Parameter(help="Action: list | approve | reject | lock | unlock"),
]

ReviewSlug = Annotated[
    str | None,
    Parameter(help="Page slug (required for approve/reject/lock/unlock)."),
]

ReviewReason = Annotated[
    str | None,
    Parameter(name="--reason", help="Rejection reason (for 'reject')."),
]

ReviewAll = Annotated[
    bool,
    Parameter(name="--all", help="Approve every pending page.", negative=""),
]


# ── N4 provenance / Q21 trust / verification ────────────────────────────

Provenance = Annotated[
    bool | None,
    Parameter(
        name="--provenance",
        help="Emit claim-level [source:42-58] citations in generated pages. Default: off.",
        group=OUTPUT_GROUP,
    ),
]

Private = Annotated[
    bool | None,
    Parameter(
        name="--private",
        help="Mark generated pages as private (excluded from default search). Default: off.",
        group=OUTPUT_GROUP,
    ),
]

NoPrivacy = Annotated[
    bool | None,
    Parameter(
        name="--privacy",
        help="Disable privacy redaction (secrets/PII patterns kept as-is). Default: filter on.",
        group=OUTPUT_GROUP,
    ),
]

Verify = Annotated[
    bool | None,
    Parameter(
        name="--verify",
        help="Run grounding verification on each claim. Default: on; --no-verify skips.",
        group=OUTPUT_GROUP,
    ),
]

ProvenanceSlugs = Annotated[
    list[str] | None,
    Parameter(negative="", help="Page slug(s) or title(s) to trace. Omit to list all pages with their slugs."),
]

ProvenanceWithContent = Annotated[
    bool,
    Parameter(name="--with-content", help="Show cited content snippets.", negative=""),
]

# ── log command ─────────────────────────────────────────────────────────

LogLast = Annotated[
    int,
    Parameter(
        name=("--last", "-n"),
        help="Maximum number of command/lint entries to include.",
    ),
]

LogWrite = Annotated[
    bool,
    Parameter(
        name="--write",
        help="Write the report to .kompend/log.md instead of printing it.",
        negative="",
    ),
]

# ── explain command ─────────────────────────────────────────────────────

ExplainSource = Annotated[
    str,
    Parameter(
        negative="",
        help="Source file to explain: ledger name (sources/notes/a.md), file name, or an unambiguous fragment.",
    ),
]

ExplainRun = Annotated[
    str | None,
    Parameter(
        name="--run",
        negative="",
        help="Run id to explain; defaults to the latest run that touched the file.",
    ),
]
