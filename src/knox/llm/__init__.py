"""LLM utilities for KNOX.

Provides context window discovery, format utilities (JSON/TOML), model
construction, and usage tracking. Business modules build models via
:func:`knox.llm.models.build_model` — never construct pydantic-ai models
directly.

Heavy submodules (``models``, ``postprocess``, ``context_window``, …) are
imported lazily via ``lazy_loader``, so importing anything under ``knox.llm.*``
— e.g. the lightweight :mod:`knox.llm.user_providers` used by config
resolution — does not pull in pydantic-ai and the provider SDKs on every
``knox`` invocation (notably ``--help``). Pattern per scientific-python
SPEC-0001; ``EAGER_IMPORT=1`` disables it for debugging.
"""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "context_window": ["ContextSource", "ContextWindow", "discover_context_window"],
        "postprocess": [
            "DebugSaver",
            "ResponsePipeline",
            "ResponsePostprocessor",
            "SurrogateScrubber",
            "ThinkingPartRepair",
            "ToolCallRepair",
            "UsageRecorder",
            "default_pipeline",
            "get_registered_processor",
            "list_registered_processors",
            "pipeline_from_registry",
            "register_postprocessor",
        ],
        "models": ["build_model"],
        "debug": ["messages_to_debug_dicts", "save_debug_prompt"],
        "format": ["salvage_json", "strip_reasoning"],
        "usage": ["UsageTracker"],
    },
)
