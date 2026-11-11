"""ACP agent — kompend as an ACP-compatible agent for credential-free LLM injection.

Heavy submodules (``harness_adapter`` → pydantic-ai + the MCP server chain) are
imported lazily via ``lazy_loader``, so importing anything under ``kompend.acp.*``
— e.g. the lightweight :mod:`kompend.acp.definitions` enum used by the CLI — does
not pull in pydantic-ai, the MCP server, and the agent/pipeline stack on every
``kompend`` invocation (notably ``--help``). Pattern per scientific-python
SPEC-0001; ``EAGER_IMPORT=1`` disables it for debugging.
"""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["definitions", "registry", "provider_store", "model_selection", "provider_wire", "endpoint_model", "acp_agent"],
    submod_attrs={
        "definitions": ["AgentDefinition", "AgentTask"],
        "registry": ["AgentRegistry"],
        "provider_store": ["LlmApiType", "ProviderConfig", "ProviderCredentials", "credentials_store"],
        "model_selection": ["NO_MODEL_MESSAGE", "ConfigModelSelector"],
        "endpoint_model": ["EndpointOverrideSelector"],
        "provider_wire": ["PROVIDER_ID", "ProviderWire", "build_provider_router", "serve_agent"],
        "harness_adapter": ["build_kompend_agent", "kompend_session_config"],
        "acp_agent": ["AGENT_NAME", "AGENT_VERSION", "KompendACPAgent", "build_acp_agent", "serve", "serve_sync"],
    },
)
