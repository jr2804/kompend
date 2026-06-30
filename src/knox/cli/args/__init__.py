"""KNOX CLI argument types organized by command/domain.

This module re-exports all argument types from submodules to maintain
backward compatibility with existing imports. Use specific submodule imports
for better organization:

    from knox.cli.args.common import ConfigPath, WikiDir
    from knox.cli.args.query import QueryText, QueryTopK
    from knox.cli.args.compile import TargetFormat, OnlineUrls
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
    Progress,
    Recursive,
    Verbose,
    WikiDir,
)

# Compile command
from .compile import (
    CompileFiles,
    CompileLint,
    EmbedModel,
    Frontmatter,
    Ignore,
    IndexStore,
    OnlineUrls,
    SynthesizeFlag,
    UrlFile,
)

# Config command
from .config import ConfigGlobalOpt, ConfigKeyArg, ConfigValueArg

# Init command
from .init import InitBare, InitConfig, InitForce, InitGit, InitKeepCache, InitTarget

# Memory command
from .memory import (
    MemoryAnswer,
    MemoryFeedback,
    MemoryLimit,
    MemoryQuery,
    MemoryTopK,
)

# Other commands (lint, serve, organize, capture, learn, ingest, llm, etc.)
from .other import (
    BypassReview,
    CaptureText,
    IncludePrivate,
    IngestTarget,
    LearnTags,
    LearnText,
    LintDryRun,
    LintFix,
    LintHistoryLast,
    LintHistoryTrend,
    LintLlm,
    LintPruneAgeDays,
    LintPruneKeep,
    LintRule,
    LlmContextBudget,
    LlmDefaultModel,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    LlmRoleModel,
    MemoryEnabled,
    NoPrivacy,
    NoUserTracking,
    OrganizeCollapseSingle,
    OrganizeDryRun,
    OrganizeMaxDepth,
    OrganizeMinSubcategories,
    Private,
    Provenance,
    ReviewMode,
    ServeHost,
    ServeOpen,
    ServePort,
    Verify,
    VlmModel,
)

# Parse command
from .parse import ParseFiles, PrintElements

# Pipeline/tiered compilation
from .pipeline import (
    ForceTier,
    InPlaceFlag,
    ListTiers,
    NoCacheFlag,
    StartAtTier,
    Workers,
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
    SearchMode,
)

__all__ = [
    # Common
    "APP_NAME_UPPERCASE",
    "DEFAULT_LLM_FALLBACK_MAX_RETRIES",
    "DEFAULT_START_AT_TIER",
    # Other
    "BypassReview",
    "CaptureText",
    "CliSet",
    # Compile
    "CompileFiles",
    "CompileLint",
    # Config
    "ConfigGlobalOpt",
    "ConfigKeyArg",
    "ConfigPath",
    "ConfigValueArg",
    "EmbedModel",
    # Pipeline
    "ForceTier",
    "Frontmatter",
    "Ignore",
    "InPlaceFlag",
    "IncludePrivate",
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
    "LearnTags",
    "LearnText",
    "LintDryRun",
    "LintFix",
    "LintHistoryLast",
    "LintHistoryTrend",
    "LintLlm",
    "LintPruneAgeDays",
    "LintPruneKeep",
    "LintRule",
    "ListTiers",
    "LlmContextBudget",
    "LlmDefaultModel",
    "LlmFallbackChain",
    "LlmFallbackMaxRetries",
    "LlmRoleModel",
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
    # Parse
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
    "Recursive",
    "ReviewMode",
    "SearchMode",
    "ServeHost",
    "ServeOpen",
    "ServePort",
    "StartAtTier",
    "SynthesizeFlag",
    "UrlFile",
    "Verbose",
    "Verify",
    "VlmModel",
    "WikiDir",
    "Workers",
]
