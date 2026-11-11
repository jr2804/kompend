"""Serve kompend over ACP with the ``providers/*`` methods wired in.

``run_acp_stdio_sync`` builds the adapter and the router internally, so it
offers no hook for the provider routes. This module builds the same adapter
itself (the harness class is public), advertises the ``providers`` capability on
``initialize``, and hands the pair to :func:`kompend.acp.provider_wire.serve_agent`.

Advertising the capability is part of the contract: a spec-following client only
calls ``providers/*`` when ``initialize`` says the agent serves them, and the
advertisement is unconditional because the routes are always registered here.
"""

from __future__ import annotations

import asyncio

from acp import schema
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.acp import PydanticAIACPAgent

from kompend.acp import kompend_session_config
from kompend.acp.provider_store import credentials_store
from kompend.acp.provider_wire import ProviderWire, serve_agent

#: Name and version kompend advertises to the ACP client.
AGENT_NAME = "kompend"
AGENT_VERSION = "0.1.0"


class KompendACPAgent(PydanticAIACPAgent[None, str]):
    """The harness ACP adapter, advertising the ``providers`` capability.

    The harness adapter serves the protocol and the sessions; kompend adds only
    the provider-configuration advertisement, because the ``providers/*`` routes
    are registered on the router in :func:`serve`.
    """

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: schema.ClientCapabilities | None = None,
        client_info: schema.Implementation | None = None,
        **kwargs: object,
    ) -> schema.InitializeResponse:
        """Negotiate the protocol and advertise provider configuration support."""
        response = await super().initialize(
            protocol_version,
            client_capabilities=client_capabilities,
            client_info=client_info,
            **kwargs,
        )
        capabilities = response.agent_capabilities or schema.AgentCapabilities()
        advertised = capabilities.model_copy(update={"providers": schema.ProvidersCapabilities()})
        return response.model_copy(update={"agent_capabilities": advertised})


def serve_sync(agent: Agent[None, str]) -> None:
    """Synchronous entry point: run :func:`serve` on a fresh event loop."""
    asyncio.run(serve(agent))


async def serve(agent: Agent[None, str]) -> None:
    """Serve *agent* over stdio until the client disconnects."""
    await serve_agent(
        build_acp_agent(agent),
        ProviderWire(credentials_store()),
    )


def build_acp_agent(agent: Agent[None, str]) -> KompendACPAgent:
    """Build the adapter around *agent* with kompend's session factory.

    Args:
        agent: The pydantic-ai agent to expose over ACP.
    """
    return KompendACPAgent(
        agent,
        session_config=kompend_session_config,
        name=AGENT_NAME,
        version=AGENT_VERSION,
    )


__all__ = ["AGENT_NAME", "AGENT_VERSION", "KompendACPAgent", "build_acp_agent", "serve", "serve_sync"]
