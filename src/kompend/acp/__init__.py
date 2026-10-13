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
    submodules=["definitions", "registry", "provider_store"],
    submod_attrs={
        "definitions": ["AgentDefinition", "AgentTask"],
        "registry": ["AgentRegistry"],
        "provider_store": ["LlmApiType", "ProviderConfig", "ProviderCredentials"],
        "harness_adapter": ["build_kompend_agent", "kompend_session_config"],
    },
)
