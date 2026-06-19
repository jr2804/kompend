"""Custom litellm provider registrations for KNOX.

To avoid having to set ``OPENAI_BASE_URL`` or ``OPENAI_API_KEY`` (which
conflicts with real OpenAI usage), this submodule lets you implement your
own custom providers — or simply register "newcomers" like claudin.io.

Even though litellm supports a vast number of providers out of the box,
some very obvious ones are still missing: Ollama Cloud, OpenCode Go/Zen.

Registered providers
--------------------
``claudin-io``
    Anthropic-compatible API at https://api.claudin.io.  Delegates
    to the ``anthropic`` provider but routes to the claudin.io endpoint.

    Model strings follow Anthropic's naming:
    ``claudin-io/claude-3.5-sonnet``, ``claudin-io/claude-3-haiku``, etc.

    Example::

        default_model = "claudin-io/claude-3.5-sonnet"

    Authentication: set ``CLAUDINIO_API_KEY`` in your environment.

``ollama-cloud``
    Ollama's cloud service (https://ollama.com).  Mirrors the local
    ``ollama`` provider but routes every call to the cloud API using
    the ``OLLAMA_API_KEY`` environment variable for authentication.

    Model strings follow the same convention as the local provider:
    ``ollama-cloud/<model-name>``

    Example::

        default_model = "ollama-cloud/deepseek-v4-flash"

    Authentication: set ``OLLAMA_API_KEY`` in your environment.

``opencode-zen``
    OpenCode Zen (https://opencode.ai) — token-based pay-as-you-go
    with open source + frontier models and time-limited free tiers.
    Routes to ``https://opencode.ai/zen/v1/...`` endpoints.

    Model strings: ``opencode-zen/<model-name>``

    Example::

        default_model = "opencode-zen/deepseek-v4-flash-free"

    Authentication: ``OPENCODE_ZEN_API_KEY`` or ``OPENCODE_API_KEY``.

``opencode-go``
    OpenCode Go (https://opencode.ai) — monthly subscription with
    mainly open source models.  Routes to
    ``https://opencode.ai/zen/go/v1/...`` endpoints.

    Model strings: ``opencode-go/<model-name>``

    Example::

        default_model = "opencode-go/deepseek-v4-flash"

    Authentication: ``OPENCODE_GO_API_KEY`` or ``OPENCODE_API_KEY``.

``opencode``
    Invalid/placeholder — raises an error directing the user to
    ``opencode-zen`` or ``opencode-go``.
"""

from __future__ import annotations

import logging

import litellm

from knox.llm.providers.claudin_io import _ClaudinIOProvider
from knox.llm.providers.ollama_cloud import _OllamaCloudProvider
from knox.llm.providers.opencode import (
    _OpenCodeDummyProvider,
    _OpenCodeGoProvider,
    _OpenCodeProvider,
    _OpenCodeZenProvider,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_ClaudinIOProvider",
    "_OllamaCloudProvider",
    "_OpenCodeDummyProvider",
    "_OpenCodeGoProvider",
    "_OpenCodeProvider",
    "_OpenCodeZenProvider",
    "_register_custom_providers",
]


def _register_custom_providers() -> None:
    """Register all Knox custom providers into litellm.

    Idempotent — safe to call multiple times.
    """
    providers: list[tuple[str, type[litellm.CustomLLM]]] = [
        ("claudin-io", _ClaudinIOProvider),
        ("ollama-cloud", _OllamaCloudProvider),
        ("opencode-zen", _OpenCodeZenProvider),
        ("opencode-go", _OpenCodeGoProvider),
        ("opencode", _OpenCodeDummyProvider),
    ]

    if litellm.custom_provider_map is None:
        litellm.custom_provider_map = []

    for name, handler_cls in providers:
        already = [e for e in litellm.custom_provider_map if e.get("provider") == name]
        if not already:
            litellm.custom_provider_map.append({"provider": name, "custom_handler": handler_cls()})
            logger.debug("Registered custom litellm provider: %s", name)


# Auto-register on import.
_register_custom_providers()
