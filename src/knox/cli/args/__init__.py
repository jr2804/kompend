"""KNOX CLI argument types organized by command/domain.

This module re-exports all argument types from submodules to maintain
backward compatibility with existing imports. Use specific submodule imports
for better organization:

    from knox.cli.args.common import ConfigPath, WikiDir
    from knox.cli.args.query import QueryText, QueryTopK
    from knox.cli.args.compile import OnlineUrls
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
    Parser,
    Progress,
    Recursive,
    Verbose,
    WikiDir,
)

# Compile command
from .compile import (
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
    IngestTarget,
    LearnText,
    LintDryRun,
    LintFix,
    LintFormatCheck,
    LintHistoryLast,
    LintHistoryTrend,
    LintLlm,
    LintPruneAgeDays,
    LintPruneKeep,
    LintRule,
    LlmDefaultModel,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    LlmRoleModel,
    MemoryEnabled,
    NoPrivacy,
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
    SourceAttrs,
    SourceTags,
    Verify,
    VlmModel,
)
from .parse import ParseFiles, PrintElements

# Pipeline/tiered compilation
from .pipeline import (
    ForceTier,
    InPlaceFlag,
    JsonProgress,
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
    QueryVerbosity,
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
    "OnlineUrls",
    "OrganizeCollapseSingle",
    "OrganizeDryRun",
    "OrganizeMaxDepth",
    "OrganizeMinSubcategories",
    "ParseFiles",
    "Parser",
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
    "VlmModel",
    "WikiDir",
    "Workers",
]
