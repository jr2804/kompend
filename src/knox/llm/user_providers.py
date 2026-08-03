"""User-defined LLM provider registry.

Holds the ``[providers.<name>]`` config section in a lightweight module with no
heavy imports (pydantic-ai, SDKs), so the config engine can register providers
without paying ``knox.llm.models``' import cost (which pulls pydantic-ai,
openai, ollama, …) on every ``knox`` invocation — including ``--help``.
"""

from __future__ import annotations

from typing import Any

#: Module-level store for user-defined providers. Maps provider_id -> dict with
#: keys: base_url, api_key (optional), context_window_cap (optional).
_user_providers: dict[str, dict[str, Any]] = {}


def set_user_providers(providers: dict[str, dict[str, Any]]) -> None:
    """Register user-defined providers for ``build_model()`` resolution.

    Called during config resolution from the merged config. Each entry is a
    ``{provider_id: {base_url: str, api_key: str | None, context_window_cap: int | None}}``
    dict matching the ``[providers.<name>]`` TOML section.
    """
    _user_providers.clear()
    _user_providers.update(providers)


__all__ = ["_user_providers", "set_user_providers"]
