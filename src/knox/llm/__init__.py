"""Centralised LLM client for KNOX.

All LLM calls in the pipeline go through :class:`LLMClient` from
:mod:`knox.llm.client`. The client wraps litellm with:

- A typed error hierarchy (:mod:`knox.llm.errors`)
- Configurable fallback chains (:mod:`knox.llm.failover`)
- An exception classifier (:mod:`knox.llm.classifier`)
- Per-call observability (model_used, attempts, latency)

This module is the single source of truth for LLM semantics in KNOX.
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
