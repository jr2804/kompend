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
    DEFAULT_COMPILE_FORMAT,
    DEFAULT_COMPILER,
    DEFAULT_IMAGE_MODE,
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_PROMPT_BUDGET,
    DEFAULT_START_AT_TIER,
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
    EmbedBatchSize,
    EmbedModel,
    Frontmatter,
    Ignore,
    ImageMode,
    IndexStore,
    OnlineUrls,
    OutputFlavor,
    SynthesizeFlag,
    TargetFormat,
    UrlFile,
)

# Config command
from .config import ConfigGlobalOpt, ConfigKeyArg, ConfigValueArg

# Init command
from .init import InitBare, InitForce, InitGit, InitKeepCache, InitTarget

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
    BudgetConcepts,
    BudgetGenSegment,
    BypassReview,
    CaptureText,
    CompilerChoice,
    IncludePrivate,
    IngestTarget,
    KindOption,
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
    LlmTimeout,
    MemoryDecayThreshold,
    MemoryEnabled,
    MemoryHalfLifeDays,
    NoPrivacy,
    NoUserTracking,
    OrganizeCollapseSingle,
    OrganizeDryRun,
    OrganizeMaxDepth,
    OrganizeMinSubcategories,
    Private,
    PromptBudget,
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
    QueryFormat,
    QueryNoClassify,
    QueryRaw,
    QueryRemember,
    QuerySearchMode,
    QueryText,
    QueryTopK,
)

__all__ = [
    # Common
    "APP_NAME_UPPERCASE",
    "DEFAULT_COMPILER",
    "DEFAULT_COMPILE_FORMAT",
    "DEFAULT_IMAGE_MODE",
    "DEFAULT_LLM_FALLBACK_MAX_RETRIES",
    "DEFAULT_PROMPT_BUDGET",
    "DEFAULT_START_AT_TIER",
    "BudgetConcepts",
    "BudgetGenSegment",
    # Other
    "BypassReview",
    "CaptureText",
    # Compile
    "CompileFiles",
    "CompileLint",
    "CompilerChoice",
    # Config
    "ConfigGlobalOpt",
    "ConfigKeyArg",
    "ConfigPath",
    "ConfigValueArg",
    "EmbedBatchSize",
    "EmbedModel",
    # Pipeline
    "ForceTier",
    "Frontmatter",
    "Ignore",
    "ImageMode",
    "InPlaceFlag",
    "IncludePrivate",
    "IndexStore",
    "IngestTarget",
    # Init
    "InitBare",
    "InitForce",
    "InitGit",
    "InitKeepCache",
    "InitTarget",
    "JsonOutput",
    "KindOption",
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
    "LlmTimeout",
    # Memory
    "MemoryAnswer",
    "MemoryDecayThreshold",
    "MemoryEnabled",
    "MemoryFeedback",
    "MemoryHalfLifeDays",
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
    "OutputFlavor",
    # Parse
    "ParseFiles",
    "PrintElements",
    "Private",
    "Progress",
    "PromptBudget",
    "Provenance",
    # Query
    "QueryAdaptive",
    "QueryFormat",
    "QueryNoClassify",
    "QueryRaw",
    "QueryRemember",
    "QuerySearchMode",
    "QueryText",
    "QueryTopK",
    "Recursive",
    "ReviewMode",
    "ServeHost",
    "ServeOpen",
    "ServePort",
    "StartAtTier",
    "SynthesizeFlag",
    "TargetFormat",
    "UrlFile",
    "Verbose",
    "Verify",
    "VlmModel",
    "WikiDir",
    "Workers",
]
