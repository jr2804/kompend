"""LLM utilities for KNOX.

Provides context window discovery, format utilities (JSON/TOON/ISON),
model construction, and usage tracking. Business modules build models
via :func:`knox.llm.models.build_model` — never import litellm directly.
"""

from __future__ import annotations

from knox.llm.context_window import ContextSource, ContextWindow, discover_context_window
from knox.llm.format import salvage_json, strip_reasoning
from knox.llm.models import build_model
from knox.llm.usage import UsageTracker

__all__ = [
    "ContextSource",
    "ContextWindow",
    "UsageTracker",
    "build_model",
    "discover_context_window",
    "salvage_json",
    "strip_reasoning",
]
