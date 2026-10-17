"""CLI arguments for other commands (lint, serve, organize, capture, etc.).

Declarations now live in the domain modules mirroring ``cmds/``; this module
re-exports them so deep imports keep working:

    from kompend.cli.args.other import LintFix, ExportAll

Retire this shim once no caller deep-imports from ``args.other`` (Phase 3).
"""

from __future__ import annotations

# Compile command (LLM failover configuration)
from .compile import (
    LlmDefaultModel,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    VlmModel,
)

# Knowledge commands (attachments, references, capture, pack, organize, agents)
from .knowledge import (
    CaptureText,
    IngestTarget,
    LearnText,
    OrganizeCollapseSingle,
    OrganizeDryRun,
    OrganizeMaxDepth,
    OrganizeMinSubcategories,
    SourceAttrs,
    SourceTags,
)

# Pipeline/tiered compilation (sources export strays)
from .pipeline import (
    ExportAll,
    ExportAnnotations,
    ExportInline,
    ExportSummary,
    MemoryEnabled,
)

# Quality commands (lint, review, provenance, doctor, status)
from .quality import (
    BypassReview,
    LintDryRun,
    LintFix,
    LintFixDestructive,
    LintFormatCheck,
    LintHistoryLast,
    LintHistoryTrend,
    LintLlm,
    LintPruneAgeDays,
    LintPruneKeep,
    LintRule,
    NoPrivacy,
    Private,
    Provenance,
    ReviewMode,
    Verify,
)

# Serving commands (serve, mcp, agent)
from .serving import ServeHost, ServeOpen, ServePort

__all__ = [
    "BypassReview",
    "CaptureText",
    "ExportAll",
    "ExportAnnotations",
    "ExportInline",
    "ExportSummary",
    "IngestTarget",
    "LearnText",
    "LintDryRun",
    "LintFix",
    "LintFixDestructive",
    "LintFormatCheck",
    "LintHistoryLast",
    "LintHistoryTrend",
    "LintLlm",
    "LintPruneAgeDays",
    "LintPruneKeep",
    "LintRule",
    "LlmDefaultModel",
    "LlmFallbackChain",
    "LlmFallbackMaxRetries",
    "MemoryEnabled",
    "NoPrivacy",
    "OrganizeCollapseSingle",
    "OrganizeDryRun",
    "OrganizeMaxDepth",
    "OrganizeMinSubcategories",
    "Private",
    "Provenance",
    "ReviewMode",
    "ServeHost",
    "ServeOpen",
    "ServePort",
    "SourceAttrs",
    "SourceTags",
    "Verify",
    "VlmModel",
]
