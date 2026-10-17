"""kompend CLI argument types organized by command/domain.

This module re-exports all argument types from submodules to maintain
backward compatibility with existing imports. Use specific submodule imports
for better organization:

    from kompend.cli.args.common import ConfigPath, WikiDir
    from kompend.cli.args.query import QueryText, QueryTopK
    from kompend.cli.args.compile import OnlineUrls
"""

from __future__ import annotations

# Common/global arguments
from .common import (
    APP_NAME_UPPERCASE,
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_START_AT_TIER,
    CliSet,
    ConfigPath,
    JsonOutput,
    NoUserTracking,
    Progress,
    Recursive,
    Verbose,
    Version,
    WikiDir,
)

# Compile command
from .compile import (
    CompileLint,
    CompileLintAll,
    EmbedModel,
    Frontmatter,
    Ignore,
    IndexStore,
    LlmDefaultModel,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    OnlineUrls,
    SynthesizeFlag,
    UrlFile,
    VlmModel,
)

# Config command
from .config import ConfigGlobalOpt, ConfigKeyArg, ConfigValueArg

# Init command
from .init import InitBare, InitConfig, InitForce, InitGit, InitKeepCache, InitTarget

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

# Memory command
from .memory import (
    MemoryAnswer,
    MemoryFeedback,
    MemoryLimit,
    MemoryQuery,
    MemoryTopK,
)
from .parse import ParseFiles, PrintElements

# Pipeline/tiered compilation
from .pipeline import (
    ExportAll,
    ExportAnnotations,
    ExportInline,
    ExportSummary,
    ForceTier,
    InPlaceFlag,
    JsonProgress,
    ListTiers,
    MemoryEnabled,
    NoCacheFlag,
    StartAtTier,
    Workers,
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

# Query command
from .query import (
    QueryAdaptive,
    QueryFormatOption,
    QueryNoClassify,
    QueryRaw,
    QueryRemember,
    QuerySearchMode,
    QueryText,
    QueryTopK,
    QueryVerbosity,
    SearchMode,
)

# Serving commands (serve, mcp, agent)
from .serving import ServeHost, ServeOpen, ServePort

__all__ = [
    # Common
    "APP_NAME_UPPERCASE",
    "DEFAULT_LLM_FALLBACK_MAX_RETRIES",
    "DEFAULT_START_AT_TIER",
    # Other
    "BypassReview",
    # Knowledge
    "CaptureText",
    "CliSet",
    # Compile
    "CompileLint",
    "CompileLintAll",
    # Config
    "ConfigGlobalOpt",
    "ConfigKeyArg",
    "ConfigPath",
    "ConfigValueArg",
    "EmbedModel",
    # Pipeline
    "ExportAll",
    "ExportAnnotations",
    "ExportInline",
    "ExportSummary",
    "ForceTier",
    "Frontmatter",
    "Ignore",
    "InPlaceFlag",
    "IndexStore",
    "IngestTarget",
    # Init
    "InitBare",
    "InitConfig",
    "InitForce",
    "InitGit",
    "InitKeepCache",
    "InitTarget",
    "JsonOutput",
    "JsonProgress",
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
    "ListTiers",
    "LlmDefaultModel",
    "LlmFallbackChain",
    "LlmFallbackMaxRetries",
    # Memory
    "MemoryAnswer",
    "MemoryEnabled",
    "MemoryFeedback",
    "MemoryLimit",
    "MemoryQuery",
    "MemoryTopK",
    "NoCacheFlag",
    "NoPrivacy",
    "NoUserTracking",
    "OnlineUrls",
    "OrganizeCollapseSingle",
    "OrganizeDryRun",
    "OrganizeMaxDepth",
    "OrganizeMinSubcategories",
    "ParseFiles",
    "PrintElements",
    "Private",
    "Progress",
    "Provenance",
    # Query
    "QueryAdaptive",
    "QueryFormatOption",
    "QueryNoClassify",
    "QueryRaw",
    "QueryRemember",
    "QuerySearchMode",
    "QueryText",
    "QueryTopK",
    "QueryVerbosity",
    "Recursive",
    # Serving
    "ReviewMode",
    "SearchMode",
    "ServeHost",
    "ServeOpen",
    "ServePort",
    "SourceAttrs",
    "SourceTags",
    "StartAtTier",
    "SynthesizeFlag",
    "UrlFile",
    "Verbose",
    "Verify",
    "Version",
    "VlmModel",
    "WikiDir",
    "Workers",
]
