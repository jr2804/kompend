"""Custom litellm provider registrations for KNOX.

To avoid having to set ``OPENAI_BASE_URL`` or ``OPENAI_API_KEY`` (which
conflicts with real OpenAI usage), this submodule lets you implement your
own custom providers — or simply register "newcomers" like claudin.io.

Even though litellm supports a vast number of providers out of the box,
some very obvious ones are still missing: Ollama Cloud, OpenCode Go/Zen.

Registered providers
--------------------
``claudin-io``
    OpenAI-compatible API at https://api.claudin.io/v1/chat/completions.

    Example::

        default_model = "claudin-io/claudinio"

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

from typing import Any

import litellm

from knox.llm.providers.claudin_io import _ClaudinIOProvider
from knox.llm.providers.ollama_cloud import _OllamaCloudProvider
from knox.llm.providers.opencode import (
    _OpenCodeDummyProvider,
    _OpenCodeGoProvider,
    _OpenCodeProvider,
    _OpenCodeZenProvider,
)
from knox.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "_ClaudinIOProvider",
    "_OllamaCloudProvider",
    "_OpenCodeDummyProvider",
    "_OpenCodeGoProvider",
    "_OpenCodeProvider",
    "_OpenCodeZenProvider",
    "_register_custom_providers",
]

_CUSTOM_PROVIDER_PREFIXES: set[str] = set()


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

    # Trigger lazy load of provider_list so we can extend it.
    _ = litellm.provider_list

    for name, handler_cls in providers:
        already = [e for e in litellm.custom_provider_map if e.get("provider") == name]
        if not already:
            litellm.custom_provider_map.append({"provider": name, "custom_handler": handler_cls()})
            litellm._custom_providers.append(name)
            _CUSTOM_PROVIDER_PREFIXES.add(name)
            logger.debug("Registered custom litellm provider: %s", name)


def _patch_get_llm_provider() -> None:
    """Monkey-patch ``get_llm_provider`` to keep custom-provider
    prefixes intact in the returned model name.

    Without this patch, ``get_llm_provider`` strips the prefix (e.g.
    ``opencode-zen/gpt-5.4-nano`` → ``gpt-5.4-nano``) and litellm's
    ``completion()`` then matches the bare model name against
    ``open_ai_chat_completion_models``, routing to the OpenAI handler
    instead of our custom provider.

    With the patch, the full ``prefix/model`` string is returned, so
    the OpenAI model-name check does not falsely match, and the
    ``custom_llm_provider in litellm._custom_providers`` branch is
    reached correctly.

    Patches both ``litellm.utils.get_llm_provider`` (used by
    ``litellm.main.completion``) and ``litellm.get_llm_provider``
    (the public API).
    """
    _original = litellm.utils.get_llm_provider

    def _patched(
        model: str,
        custom_llm_provider: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        litellm_params: Any = None,
    ) -> Any:
        if "/" in model:
            prefix = model.split("/", 1)[0]
            if prefix in _CUSTOM_PROVIDER_PREFIXES:
                return model, prefix, api_key, api_base
        return _original(model, custom_llm_provider, api_base, api_key, litellm_params)

    litellm.utils.get_llm_provider = _patched  # type: ignore[assignment]
    litellm.get_llm_provider = _patched  # type: ignore[assignment]
    litellm.main.get_llm_provider = _patched  # type: ignore[assignment]


# Auto-register on import.
_register_custom_providers()
_patch_get_llm_provider()
