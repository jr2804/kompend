"""Custom litellm provider for Ollama Cloud (https://ollama.com).

``ollama-cloud/<model>`` delegates to ``ollama_chat/<model>`` with
the cloud ``api_base`` and ``OLLAMA_API_KEY``.
"""

from __future__ import annotations

from typing import Any

from knox.llm.providers._base import _CustomProvider

_OLLAMA_CLOUD_BASE_URL = "https://ollama.com"
_OLLAMA_CLOUD_API_KEY_ENV = "OLLAMA_API_KEY"


class _OllamaCloudProvider(_CustomProvider):
    """Routes ``ollama-cloud/<model>`` through litellm's existing
    ``ollama_chat`` handler but at the Ollama Cloud API endpoint.
    """

    BASE_URL = _OLLAMA_CLOUD_BASE_URL
    API_KEY_ENV = _OLLAMA_CLOUD_API_KEY_ENV
    MODEL_PREFIX = "ollama_chat"

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
