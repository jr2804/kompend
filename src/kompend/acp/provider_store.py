"""Thread-safe store for the LLM endpoint an ACP client injects.

The editor sends its endpoint through the ACP ``providers/set`` method; the
routes live in :mod:`kompend.acp.provider_wire`, and model selection honours the
stored endpoint in :mod:`kompend.acp.endpoint_model`. This module owns only the
value and its locking — it deliberately does not touch ``os.environ``: kompend
reads ``compile.llm.default_model`` and the ``[providers.*]`` blocks, so an
env-var injection would be lost. The endpoint reaches the pipeline by being
handed to the model constructor.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from aenum import MultiValueEnum

#: Header carrying the bearer token. Header names are case-insensitive.
_AUTHORIZATION: Final[str] = "authorization"
_BEARER_PREFIX: Final[str] = "bearer "


class LlmApiType(StrEnum, MultiValueEnum):
    """Well-known ACP ``apiType`` identifiers.

    The set of provider protocols ``providers/set`` accepts and ``providers/list``
    advertises. Values are the strings clients send on the wire; we never
    re-type them at the call site.
    """

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

    def api_key(self) -> str | None:
        """Return the bearer token from the ``Authorization`` header, if any.

        The secret travels in that header — the only place the ACP provider
        payload carries it — and is stripped of its scheme before use. Header
        names are case-insensitive.
        """
        for key, value in self.headers.items():
            if key.lower() != _AUTHORIZATION:
                continue
            token = value.strip()
            if token.lower().startswith(_BEARER_PREFIX):
                return token[len(_BEARER_PREFIX) :].strip()
            return token
        return None


class ProviderCredentials:
    """Thread-safe store for the LLM endpoint configured via ACP ``providers/set``.

    One endpoint is stored at a time: a later ``providers/set`` replaces it, and
    ``providers/disable`` clears it so model selection falls back to the
    workspace config.
    """

    def __init__(self) -> None:
        self._provider: ProviderConfig | None = None
        self._lock = threading.Lock()

    def set(self, provider: ProviderConfig) -> None:
        """Store the provider config.

        Called from the ``providers/set`` handler.
        """
        with self._lock:
            self._provider = provider

    def get(self) -> ProviderConfig | None:
        """Return a copy of the current provider config, or *None*."""
        with self._lock:
            return deepcopy(self._provider)

    def disable(self) -> None:
        """Clear the provider config.

        Called from the ``providers/disable`` handler.
        """
        with self._lock:
            self._provider = None


_STORE = ProviderCredentials()


def credentials_store() -> ProviderCredentials:
    """Return the process-wide store shared by the routes and model selection."""
    return _STORE


__all__ = ["LlmApiType", "ProviderConfig", "ProviderCredentials", "credentials_store"]
