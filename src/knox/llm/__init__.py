"""LLM utilities for KNOX.

Provides context window discovery, format utilities (JSON/TOON/ISON),
model construction, and usage tracking. Business modules build models
via :func:`knox.llm.models.build_model` — never construct pydantic-ai models directly.
"""

from __future__ import annotations

from knox.llm.context_window import ContextSource, ContextWindow, discover_context_window
from knox.llm.format import salvage_json, strip_reasoning
from knox.llm.models import build_model
from knox.llm.postprocess import (
    DebugSaver,
    ResponsePipeline,
    ResponsePostprocessor,
    SurrogateScrubber,
    ThinkingPartRepair,
    ToolCallRepair,
    UsageRecorder,
    default_pipeline,
    get_registered_processor,
    list_registered_processors,
    pipeline_from_registry,
    register_postprocessor,
)
from knox.llm.usage import UsageTracker

__all__ = [
    "ContextSource",
    "ContextWindow",
    "DebugSaver",
    "ResponsePipeline",
    "ResponsePostprocessor",
    "SurrogateScrubber",
    "ThinkingPartRepair",
    "ToolCallRepair",
    "UsageRecorder",
    "UsageTracker",
    "build_model",
    "default_pipeline",
    "discover_context_window",
    "get_registered_processor",
    "list_registered_processors",
    "pipeline_from_registry",
    "register_postprocessor",
    "salvage_json",
    "strip_reasoning",
]
