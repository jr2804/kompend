"""Serve the ACP ``providers/*`` methods over kompend's own router.

The SDK's ``build_agent_router`` registers no ``providers/*`` route and its
``MessageRouter`` forwards only ``_``-prefixed method names to ``ext_method``,
so an agent subclass cannot serve them. Everything used here is public:
``build_agent_router`` returns a ``MessageRouter`` whose ``route_request``
takes the SDK's own schema models, and ``acp.Connection`` accepts any
``MethodHandler``. :func:`serve_agent` assembles the same router the SDK would
build, adds the provider routes, and drives a ``Connection`` itself.

The provider surface is one endpoint, the editor's: ``providers/list`` reports
its current non-secret routing config, ``providers/set`` stores it, and
``providers/disable`` drops it so the config model serves again. Model
selection consults the same store (see :mod:`kompend.acp.model_selection`), so
an override lands on the next prompt without touching ``os.environ``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Final

from acp import AGENT_METHODS, RequestError, schema, stdio_streams
from acp.agent.router import build_agent_router
from acp.connection import Connection
from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.interfaces import Agent
from acp.router import MessageRouter
from acp.utils import normalize_result

from kompend.acp.connection_client import ConnectionClient
from kompend.acp.provider_store import LlmApiType, ProviderConfig, ProviderCredentials
from kompend.logging import get_logger

#: The single provider id kompend serves: the editor's endpoint override.
PROVIDER_ID: Final[str] = "main"

logger = get_logger(__name__)


@dataclass
class ProviderWire:
    """Serve ``providers/list|set|disable`` from a :class:`ProviderCredentials` store.

    Args:
        credentials: The store the model selector consults.
    """

    credentials: ProviderCredentials

    async def providers_list(self, **kwargs: Any) -> schema.ListProvidersResponse:
        """Report the served provider and its current routing config, if any."""
        del kwargs
        current = self.credentials.get()
        return schema.ListProvidersResponse(
            providers=[
                schema.ProviderInfo(
                    id=PROVIDER_ID,
                    supported=[api_type.value for api_type in LlmApiType],
                    required=False,
                    current=(schema.ProviderCurrentConfig(api_type=current.api_type, base_url=current.base_url) if current is not None else None),
                ),
            ],
        )

    async def providers_set(
        self,
        *,
        id: str,
        api_type: str,
        base_url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> schema.SetProviderResponse:
        """Store the editor's endpoint so the next prompt uses it.

        Raises:
            RequestError: an unknown provider id, or a request without a base URL.
        """
        del kwargs
        if id != PROVIDER_ID:
            raise RequestError.invalid_params({"id": id, "served": [PROVIDER_ID]})
        if not base_url:
            raise RequestError.invalid_params({"baseUrl": base_url})
        self.credentials.set(
            ProviderConfig(api_type=str(api_type), base_url=base_url, headers=dict(headers or {})),
        )
        return schema.SetProviderResponse()

    async def providers_disable(self, *, id: str, **kwargs: Any) -> schema.DisableProviderResponse:
        """Drop the override; model selection falls back to ``compile.llm.default_model``."""
        del kwargs
        if id != PROVIDER_ID:
            raise RequestError.invalid_params({"id": id, "served": [PROVIDER_ID]})
        self.credentials.disable()
        return schema.DisableProviderResponse()


async def serve_agent(
    agent: Agent,
    routes: ProviderWire,
    *,
    use_unstable_protocol: bool = True,
) -> None:
    """Serve *agent* over stdio with the provider routes registered.

    Mirrors ``acp.run_agent`` — which cannot be reused because it builds the
    router internally and offers no hook — using only public API. The agent gets
    a :class:`~kompend.acp.connection_client.ConnectionClient` on connect, the
    same client surface the SDK's own connection provides.
    """
    # ``stdio_streams`` returns ``(reader, writer)``; ``run_agent`` renames them
    # from the client's side, and ``Connection`` wants ``(handler, writer, reader)``.
    output_stream, input_stream = await stdio_streams(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
    router = build_provider_router(agent, routes, use_unstable_protocol=use_unstable_protocol)
    conn = Connection(router, input_stream, output_stream, listening=False)
    client = ConnectionClient(conn)
    on_connect = getattr(agent, "on_connect", None)
    if callable(on_connect):
        on_connect(client)
    try:
        await conn.main_loop()
    finally:
        await asyncio.shield(conn.close())


def build_provider_router(
    agent: Agent,
    routes: ProviderWire,
    *,
    use_unstable_protocol: bool = True,
) -> MessageRouter:
    """Register the SDK's agent routes plus ``providers/*`` on one router.

    Args:
        agent: The ACP agent implementation (the harness adapter).
        routes: The provider handlers.
        use_unstable_protocol: Forwarded to ``build_agent_router``; the harness
            always enables it so ``session/close`` is served.
    """
    router = build_agent_router(agent, use_unstable_protocol=use_unstable_protocol)
    router.route_request(
        AGENT_METHODS["providers_list"],
        schema.ListProvidersRequest,
        routes,
        "providers_list",
        adapt_result=normalize_result,
    )
    router.route_request(
        AGENT_METHODS["providers_set"],
        schema.SetProviderRequest,
        routes,
        "providers_set",
        adapt_result=normalize_result,
    )
    router.route_request(
        AGENT_METHODS["providers_disable"],
        schema.DisableProviderRequest,
        routes,
        "providers_disable",
        adapt_result=normalize_result,
    )
    return router


__all__ = ["PROVIDER_ID", "ProviderWire", "build_provider_router", "serve_agent"]
