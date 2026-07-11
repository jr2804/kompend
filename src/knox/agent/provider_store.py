"""Thread-safe LLM provider credential store for ACP providers/set.

Stores the LLM endpoint configuration received via ACP ``providers/set``
and reconfigures litellm's env-var-based resolution so the KNOX pipeline
uses the configured endpoint.
"""

from __future__ import annotations

import os
import threading
from copy import deepcopy
from enum import StrEnum


class LlmApiType(StrEnum):
    """Well-known ACP ``apiType`` identifiers.

    The set of provider identifiers accepted by ``providers/set`` and
    advertised via ``providers/list``. Values are the strings clients send
    on the wire; we never re-type them at the call site.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    VERTEX = "vertex"
    BEDROCK = "bedrock"


# ACP apiType → litellm provider prefix. Only entries whose litellm prefix
# differs from the ACP apiType need to be listed explicitly; lookups of any
# other apiType — including unknown custom protocols — fall through to the
# identity mapping (the apiType is its own litellm prefix).
_LITELLM_PREFIX_OVERRIDES: dict[LlmApiType, str] = {
    LlmApiType.VERTEX: "vertex_ai",
}


def _litellm_prefix(api_type: str) -> str:
    """Return the litellm provider prefix for an ACP ``api_type``.

    Unknown ``api_type`` values pass through unchanged, so custom protocols
    Just Work without registering an entry here. Used as the identity
    default for ``LLM_PROVIDER_MAP`` lookups.
    """
    try:
        key = LlmApiType(api_type)
    except ValueError:
        return api_type
    return _LITELLM_PREFIX_OVERRIDES.get(key, key)


class _IdentityDefaultDict(dict[str, str]):
    """``dict`` subclass that returns the requested key as the default value.

    Used to expose the well-known apiType → litellm-prefix mapping with an
    identity fallback for any unknown key. Mirrors the behaviour of a
    ``defaultdict`` whose factory received the missing key, which the
    stdlib does not support directly.
    """

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return key


# Map of well-known apiTypes to their litellm provider prefixes. Used by
# ``providers/list`` to enumerate supported providers. Unrecognised keys
# fall through to the identity mapping via ``_IdentityDefaultDict``.
LLM_PROVIDER_MAP: _IdentityDefaultDict = _IdentityDefaultDict(
    {api_type.value: _litellm_prefix(api_type.value) for api_type in LlmApiType},
)


class ProviderCredentials:
    """Thread-safe store for the LLM endpoint configured via ACP ``providers/set``.

    When a provider is set, litellm's env-var-based resolution is reconfigured
    so the KNOX pipeline uses the configured endpoint. When disabled, the
    fallback chain (env vars → config.toml → DEFAULTS) is restored.
    """

    def __init__(self) -> None:
        self._provider: ProviderConfig | None = None
        self._lock = threading.Lock()
        # Snapshot of env vars before we touched them, for clean restore
        self._saved_env: dict[str, str | None] = {}

    def set(self, provider: ProviderConfig) -> None:
        """Store the provider config and reconfigure litellm.

        Called from the ``providers/set`` handler.
        """
        with self._lock:
            self._provider = provider
            self._reconfigure_litellm()

    def get(self) -> ProviderConfig | None:
        """Return a copy of the current provider config, or *None*."""
        with self._lock:
            return deepcopy(self._provider)

    def disable(self) -> None:
        """Clear the provider config and restore env-var fallback.

        Called from the ``providers/disable`` handler.
        """
        with self._lock:
            self._provider = None
            self._restore_env()

    def _reconfigure_litellm(self) -> None:
        """Apply provider config to litellm's env-var-based resolution."""
        if self._provider is None:
            return

        prefix = _litellm_prefix(self._provider.api_type)

        model = self._provider.model or "default"
        self._set_env("KNOX_LLM_DEFAULT_MODEL", f"{prefix}/{model}")
        self._set_env("KNOX_LLM_API_BASE", self._provider.base_url)

        for key, val in self._provider.headers.items():
            if key.lower() == "authorization":
                self._set_env("KNOX_LLM_API_KEY", val.removeprefix("Bearer ").strip())

    def _set_env(self, key: str, value: str) -> None:
        """Set an env var, saving the previous value for later restore."""
        if key not in self._saved_env:
            self._saved_env[key] = os.environ.get(key)
        os.environ[key] = value

    def _restore_env(self) -> None:
        """Restore env vars to their pre-provider-set state."""
        for key, old_val in self._saved_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val
        self._saved_env.clear()
