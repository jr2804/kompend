"""ACP Agent implementation for KNOX.

Implements the ``acp.Agent`` protocol to handle ACP lifecycle, provider
configuration, and slash commands.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from acp import Agent, PromptResponse, text_block
from acp.schema import (
    ClientCapabilities,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    NewSessionResponse,
    ProvidersCapabilities,
)

from knox.agent.provider_store import LLM_PROVIDER_MAP, ProviderConfig, ProviderCredentials
from knox.agent.slash_commands import HELP_TEXT, SlashCommand, parse_slash_command


class KnoxAgent(Agent):
    """ACP agent that wraps the KNOX pipeline.

    Handles:
    - ACP lifecycle (initialize, new_session, prompt)
    - Provider configuration (providers/list, providers/set, providers/disable)
    - Slash commands (/knox compile, /knox query, etc.)
    """

    def __init__(self) -> None:
        self._providers = ProviderCredentials()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._conn: Any = None  # Set by on_connect

    def on_connect(self, conn: Any) -> None:
        """Called when the ACP connection is established."""
        self._conn = conn

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Advertise provider configuration support."""
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(name="knox", version="0.1.0"),
            agent_capabilities=ProvidersCapabilities(providers={}),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        """Create a new session."""
        session_id = f"sess_{len(self._sessions) + 1}"
        self._sessions[session_id] = {"cwd": cwd}
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> PromptResponse:
        """Handle a prompt turn — parse slash commands and invoke KNOX."""
        # Extract text from the prompt
        text = ""
        for block in prompt:
            if hasattr(block, "text"):
                text = block.text
                break

        cmd = parse_slash_command(text)
        if cmd is None:
            return PromptResponse(
                stop_reason="end_turn",
                messages=[text_block("Unknown command. Use ``/knox help`` for available commands.")],
            )

        return await self._dispatch_slash(cmd)

    async def _dispatch_slash(self, cmd: SlashCommand) -> PromptResponse:
        """Dispatch a parsed slash command to the appropriate handler."""
        if cmd.name == "help":
            return PromptResponse(
                stop_reason="end_turn",
                messages=[text_block(HELP_TEXT)],
            )

        if cmd.name == "status":
            return await self._handle_status()

        if cmd.name in ("compile", "query", "capture"):
            return await self._run_knox_cli(cmd)

        return PromptResponse(
            stop_reason="end_turn",
            messages=[text_block(f"Unknown command: ``{cmd.name}``. Use ``/knox help``.")],
        )

    async def _handle_status(self) -> PromptResponse:
        """Show vault health and cache stats."""
        return PromptResponse(
            stop_reason="end_turn",
            messages=[text_block("KNOX agent is running. Use ``/knox compile`` to compile a wiki.")],
        )

    async def _run_knox_cli(self, cmd: SlashCommand) -> PromptResponse:
        """Run a KNOX CLI command in a subprocess and return the output."""
        knox_args = [sys.executable, "-m", "knox", cmd.name, *cmd.args]
        for key, val in cmd.kwargs.items():
            knox_args.append(f"--{key}")
            if val:
                knox_args.append(val)

        proc = await asyncio.create_subprocess_exec(
            *knox_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        output = stdout.decode("utf-8", errors="replace")
        if stderr:
            output += f"\n\nStderr:\n{stderr.decode('utf-8', errors='replace')}"

        return PromptResponse(
            stop_reason="end_turn",
            messages=[text_block(output or "(no output)")],
        )

    # ── Provider configuration ──────────────────────────────────────────

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods — including providers/*."""
        if method == "providers/list":
            return await self._handle_providers_list()
        if method == "providers/set":
            return await self._handle_providers_set(params)
        if method == "providers/disable":
            return await self._handle_providers_disable(params)
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications."""
        pass

    async def _handle_providers_list(self) -> dict[str, Any]:
        """Return the current provider configuration."""
        current = self._providers.get()
        providers: list[dict[str, Any]] = [
            {
                "id": "main",
                "supported": list(LLM_PROVIDER_MAP.keys()),
                "required": True,
                "current": {
                    "apiType": current.api_type,
                    "baseUrl": current.base_url,
                }
                if current
                else None,
            }
        ]
        return {"providers": providers}

    async def _handle_providers_set(self, params: dict[str, Any]) -> dict[str, Any]:
        """Configure the LLM provider."""
        config = ProviderConfig(
            api_type=params.get("apiType", params.get("api_type", "openai")),
            base_url=params.get("baseUrl", params.get("base_url", "")),
            headers=params.get("headers", {}),
            model=params.get("model"),
        )
        self._providers.set(config)
        return {}

    async def _handle_providers_disable(self, params: dict[str, Any]) -> dict[str, Any]:
        """Disable the LLM provider."""
        self._providers.disable()
        return {}

    # ── Unused lifecycle methods ────────────────────────────────────────

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        return None

    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        return ListSessionsResponse(sessions=[])

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> Any:
        return None

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> Any:
        return None

    async def authenticate(
        self, method_id: str, **kwargs: Any
    ) -> Any:
        return None

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return None

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return None

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> Any:
        self._sessions.pop(session_id, None)
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        pass
