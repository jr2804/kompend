"""User-defined LLM provider registry.

Holds the ``[providers.<name>]`` config section in a lightweight module with no
heavy imports (pydantic-ai, SDKs), so the config engine can register providers
without paying ``knox.llm.models``' import cost (which pulls pydantic-ai,
openai, ollama, …) on every ``knox`` invocation — including ``--help``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["_user_providers", "set_user_providers"]

#: Module-level store for user-defined providers. Maps provider_id -> dict with
#: keys: base_url, api_key (optional), api_type (optional), context_window
#: (optional; deprecated alias context_window_cap), models (optional; per-model
#: settings keyed by model name, each with context_window).
_user_providers: dict[str, dict[str, Any]] = {}


def set_user_providers(providers: dict[str, dict[str, Any]]) -> None:
    """Register user-defined providers for ``build_model()`` resolution.

    Called during config resolution from the merged config. Each entry is a
    ``[providers.<name>]`` TOML section as a dict (base_url, api_key,
    api_type, context_window, models).
    """
    _user_providers.clear()
    _user_providers.update(providers)
