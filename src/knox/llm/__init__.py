"""LLM utilities for KNOX.

Provides context window discovery, format utilities (JSON/TOML), model
construction, and usage tracking. Business modules build models via
:func:`knox.llm.models.build_model` — never construct pydantic-ai models
directly.

Heavy submodules (``models``, ``postprocess``, ``context_window``, …) are
imported lazily via module ``__getattr__``, so importing anything under
``knox.llm.*`` — e.g. the lightweight :mod:`knox.llm.user_providers` used by
config resolution — does not pull in pydantic-ai and the provider SDKs on
every ``knox`` invocation (notably ``--help``).
"""

from __future__ import annotations

import importlib
from typing import Any

#: Exported name → submodule that defines it, imported on demand. No module
#: ``__all__`` is declared because static checkers cannot verify names resolved
#: through ``__getattr__``; the canonical surface is listed below for reference.
_LAZY: dict[str, str] = {
    "ContextSource": "knox.llm.context_window",
    "ContextWindow": "knox.llm.context_window",
    "discover_context_window": "knox.llm.context_window",
    "DebugSaver": "knox.llm.postprocess",
    "ResponsePipeline": "knox.llm.postprocess",
    "ResponsePostprocessor": "knox.llm.postprocess",
    "SurrogateScrubber": "knox.llm.postprocess",
    "ThinkingPartRepair": "knox.llm.postprocess",
    "ToolCallRepair": "knox.llm.postprocess",
    "UsageRecorder": "knox.llm.postprocess",
    "default_pipeline": "knox.llm.postprocess",
    "get_registered_processor": "knox.llm.postprocess",
    "list_registered_processors": "knox.llm.postprocess",
    "pipeline_from_registry": "knox.llm.postprocess",
    "register_postprocessor": "knox.llm.postprocess",
    "build_model": "knox.llm.models",
    "messages_to_debug_dicts": "knox.llm.debug",
    "save_debug_prompt": "knox.llm.debug",
    "salvage_json": "knox.llm.format",
    "strip_reasoning": "knox.llm.format",
    "UsageTracker": "knox.llm.usage",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr = getattr(importlib.import_module(module), name)
    globals()[name] = attr  # cache for subsequent access
    return attr


def __dir__() -> list[str]:
    return sorted(_LAZY)
