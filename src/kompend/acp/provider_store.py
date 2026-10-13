"""Thread-safe LLM provider credential store for ACP providers/set.

Stores the LLM endpoint configuration received via ACP ``providers/set``
and reconfigures the pipeline's env-var-based resolution so kompend uses the
configured endpoint.
"""

from __future__ import annotations

import os
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

from aenum import MultiValueEnum

from kompend.core.config.keys import EnvVar


class HttpHeader(StrEnum):
    """Well-known HTTP header names used by the ACP provider config."""

    AUTHORIZATION = "authorization"


class LlmApiType(StrEnum, MultiValueEnum):
    """Well-known ACP ``apiType`` identifiers.

    The set of provider identifiers accepted by ``providers/set`` and
    advertised via ``providers/list``. Values are the strings clients send
    on the wire; we never re-type them at the call site.
    """

    def __init__(self, value: str, *aliases: str) -> None:
        self.prefix = aliases[0] if aliases else value

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    VERTEX = "vertex", "vertex_ai"
    BEDROCK = "bedrock"


@dataclass
class ProviderConfig:
    """LLM endpoint configuration from ACP ``providers/set``."""

    api_type: str
    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    model: str | None = None


class _IdentityDefaultDict(dict[str, str]):
    """``dict`` subclass that returns the requested key as the default value.

    Used to expose the well-known apiType → provider-prefix mapping with an
    identity fallback for any unknown key. Mirrors the behaviour of a
    ``defaultdict`` whose factory received the missing key, which the
    stdlib does not support directly.
    """

    def __missing__(self, key: str) -> str:
        return key


def _provider_prefix(api_type: str) -> str:
    """Return the provider prefix for an ACP ``api_type``.

    Unknown ``api_type`` values pass through unchanged, so custom protocols
    Just Work without registering an entry here. Used as the identity
    default for ``LLM_PROVIDER_MAP`` lookups.
    """
    try:
        return LlmApiType(api_type).prefix
    except ValueError:
        return api_type


# Map of well-known apiTypes to their provider prefixes. Used by
# ``providers/list`` to enumerate supported providers. Unrecognised keys
# fall through to the identity mapping via ``_IdentityDefaultDict``.
LLM_PROVIDER_MAP: _IdentityDefaultDict = _IdentityDefaultDict(
    {api_type.value: _provider_prefix(api_type.value) for api_type in LlmApiType},
)


class ProviderCredentials:
    """Thread-safe store for the LLM endpoint configured via ACP ``providers/set``.

    When a provider is set, the pipeline's env-var-based resolution is reconfigured
    so the kompend pipeline uses the configured endpoint. When disabled, the
    fallback chain (env vars → config.toml → DEFAULTS) is restored.
    """

    def __init__(self) -> None:
        self._provider: ProviderConfig | None = None
        self._lock = threading.Lock()
        # Snapshot of env vars before we touched them, for clean restore
        self._saved_env: dict[str, str | None] = {}

    def set(self, provider: ProviderConfig) -> None:
        """Store the provider config and reconfigure the pipeline.

        Called from the ``providers/set`` handler.
        """
        with self._lock:
            self._provider = provider
            self._reconfigure_env()

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

    def _reconfigure_env(self) -> None:
        """Apply provider config to the pipeline's env-var-based resolution."""
        if self._provider is None:
            return

        prefix = _provider_prefix(self._provider.api_type)

        # Only inject the model when the provider names one — otherwise leave the
        # pipeline's configured default in place (a fabricated "prefix/default"
        # would be rejected by the LLM client).
        # KOMP_LLM_DEFAULT_MODEL and KOMP_LLM_API_BASE are env keys the LLM client
        # reads; they are injected here per ACP session, not written to kompend
        # config.
        if self._provider.model:
            self._set_env(EnvVar.LLM_DEFAULT_MODEL, f"{prefix}/{self._provider.model}")
        self._set_env(EnvVar.LLM_API_BASE, self._provider.base_url)

        # ponytail: headers stay a plain map — ACP sends arbitrary header names
        # (tests use e.g. ``x-api-key``) and only Authorization is consumed; a
        # dataclass with one known field plus an unknown-header bag would be more
        # code, not less. Header names are case-insensitive, hence ``.lower()``.
        for key, val in self._provider.headers.items():
            if key.lower() == HttpHeader.AUTHORIZATION:
                self._set_env(EnvVar.LLM_API_KEY, val.removeprefix("Bearer ").strip())

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
