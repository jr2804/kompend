"""LLM utilities for KNOX.

Provides context window discovery, error classification, failover chains,
format utilities (JSON/TOON/ISON), and usage tracking.
Business modules must not import litellm directly.
"""

from __future__ import annotations

from knox.llm.classifier import classify_litellm_error
from knox.llm.context_window import ContextSource, ContextWindow, discover_context_window
from knox.llm.errors import (
    AmbiguousError,
    ConfigurationError,
    LLMError,
    PermanentError,
    TransientError,
)
from knox.llm.failover import FailoverChain
from knox.llm.format import salvage_json, strip_reasoning
from knox.llm.usage import UsageTracker

__all__ = [
    "AmbiguousError",
    "ConfigurationError",
    "ContextSource",
    "ContextWindow",
    "FailoverChain",
    "LLMError",
    "PermanentError",
    "TransientError",
    "UsageTracker",
    "classify_litellm_error",
    "discover_context_window",
    "salvage_json",
    "strip_reasoning",
]
