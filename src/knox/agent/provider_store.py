"""Thread-safe LLM provider credential store for ACP providers/set.

Stores the LLM endpoint configuration received via ACP ``providers/set``
and reconfigures litellm's env-var-based resolution so the KNOX pipeline
uses the configured endpoint.
"""

from __future__ import annotations

import os
import threading
from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """LLM endpoint configuration from ACP ``providers/set``."""

    api_type: str
    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    model: str | None = None


# ACP apiType → litellm provider prefix
LLM_PROVIDER_MAP: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "azure": "azure",
    "vertex": "vertex_ai",
    "bedrock": "bedrock",
}


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

        prefix = LLM_PROVIDER_MAP.get(self._provider.api_type)
        if prefix is None:
            # Custom protocol — use as-is
            prefix = self._provider.api_type

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
