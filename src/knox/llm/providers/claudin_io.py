"""Custom litellm provider for claudin.io (OpenAI-compatible API).

``claudinio/<model>`` (primary), ``claudin-io/<model>`` (alias) route
both to ``https://api.claudin.io/v1`` (OpenAI-format requests; the SDK
appends ``/chat/completions``).
Model names like ``claudinio`` (balanced) or ``claudius`` (premium).
"""

from __future__ import annotations

from typing import Any

from knox.llm.providers._base import _CustomProvider

_CLAUDIN_IO_BASE_URL = "https://api.claudin.io/v1"
_CLAUDIN_IO_API_KEY_ENV = "CLAUDINIO_API_KEY"


class _ClaudinIOProvider(_CustomProvider):
    """Routes ``claudinio/<model>`` (or ``claudin-io/<model>``)
    through the ``openai`` handler at the claudin.io API endpoint.
    """

    BASE_URL = _CLAUDIN_IO_BASE_URL
    API_KEY_ENV = _CLAUDIN_IO_API_KEY_ENV
    MODEL_PREFIX = "openai"

    @classmethod
    def _delegate_kwargs(
        cls,
        model: str,
        optional_params: dict[str, Any],
        api_key: Any = None,
    ) -> dict[str, Any]:
        params = {k: v for k, v in optional_params.items() if k != "stream"}
        return {
            "model": f"{cls.MODEL_PREFIX}/{cls._short_model(model)}",
            "api_base": cls.BASE_URL,
            "api_key": cls._resolve_api_key(api_key),
            **params,
        }
