"""ACP Agent implementation for KNOX.

Implements the ``acp.Agent`` protocol to handle ACP lifecycle, provider
configuration, and slash commands.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from acp import Agent, PromptResponse
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    NewSessionResponse,
    ProvidersCapabilities,
    TextContentBlock,
)

from knox.acp._constants import (
    CONTENT_TYPE_TEXT,
    MSG_ID_PREFIX,
    PROGRESS_CURRENT,
    PROGRESS_EVENT,
    PROGRESS_MESSAGE,
    PROGRESS_PHASE,
    PROGRESS_TOTAL,
    PROGRESS_TYPE_DONE,
    PROGRESS_TYPE_ERROR,
    PROGRESS_TYPE_PROGRESS,
    SESSION_ID_PREFIX,
    SESSION_KEY_CWD,
    SESSION_UPDATE_AVAILABLE_COMMANDS,
    SESSION_UPDATE_MESSAGE_CHUNK,
    STOP_REASON_END_TURN,
)
from knox.acp.conversational import ConversationalAgent
from knox.acp.definitions import AgentDefinition, AgentTask
from knox.acp.provider_store import LLM_PROVIDER_MAP, ProviderConfig, ProviderCredentials
from knox.acp.registry import AgentRegistry
from knox.acp.slash_commands import HELP_TEXT, KNOX_COMMANDS, SlashCommand, parse_slash_command
from knox.core.config import resolve_config
from knox.core.config._core import set_config_file_value
from knox.core.config.keys import (
    AGENTS_CAPTURE,
    AGENTS_COMPILE,
    AGENTS_ORGANIZE,
    AGENTS_QUERY,
    COMPILE_LLM_DEFAULT_MODEL,
    PATHS_WIKI_DIR,
)
from knox.core.paths import KNOX_CONFIG_FILE
from knox.logging import get_logger

logger = get_logger(__name__)

_SLASH_TASK_MAP: dict[str, AgentTask] = {
    "compile": AgentTask.COMPILE,
    "query": AgentTask.QUERY,
    "capture": AgentTask.CAPTURE,
}

_AGENT_CONFIG_KEYS: dict[AgentTask, str] = {
    AgentTask.COMPILE: AGENTS_COMPILE,
    AgentTask.QUERY: AGENTS_QUERY,
    AgentTask.CAPTURE: AGENTS_CAPTURE,
    AgentTask.ORGANIZE: AGENTS_ORGANIZE,
}


class KnoxAgent(Agent):
    """ACP agent that wraps the KNOX pipeline.

    Handles:
    - ACP lifecycle (initialize, new_session, prompt)
    - Provider configuration (providers/list, providers/set, providers/disable)
    - Slash commands (/knox compile, /knox query, etc.)
    """

    def __init__(self, agent_file: str | None = None) -> None:
        self._providers = ProviderCredentials()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._conn: Any = None  # Set by on_connect
        self._agent_file = agent_file  # Explicit agent TOML override

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
            protocolVersion=protocol_version,
            agentInfo=Implementation(name="knox", version="0.1.0"),
            agentCapabilities=AgentCapabilities(providers=ProvidersCapabilities()),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        """Create a new session and advertise available slash commands."""
        session_id = f"{SESSION_ID_PREFIX}{len(self._sessions) + 1}"
        self._sessions[session_id] = {"cwd": cwd}
        await self._advertise_commands(session_id)
        return NewSessionResponse(sessionId=session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> PromptResponse:
        """Handle a prompt turn — route to conversational agent or subprocess.

        When an LLM is configured (via providers/set or config.toml), all
        prompts go through the :class:`ConversationalAgent` tool-calling loop.
        When no LLM is available, ``/knox`` slash commands fall back to
        subprocess dispatch (degraded mode), and free-text prompts are rejected.
        """
        text = ""
        for block in prompt:
            if hasattr(block, "text"):
                text = block.text
                break

        # Try conversational agent path (requires LLM)
        llm_client = self._make_llm_client(session_id)
        if llm_client is not None:
            return await self._run_conversational(text, session_id, llm_client)

        # Degraded mode: no LLM — fall back to subprocess dispatch for slash commands
        cmd = parse_slash_command(text)
        if cmd is None:
            await self._send_text(session_id, "No LLM configured. Use ``/knox`` slash commands, or configure an LLM via providers/set or config.toml.")
            return PromptResponse(stopReason=STOP_REASON_END_TURN)
        return await self._dispatch_slash(cmd, session_id)

    # ── Provider configuration ──────────────────────────────────────────

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods — providers/* and agents/*."""
        if method == "providers/list":
            return await self._handle_providers_list()
        if method == "providers/set":
            return await self._handle_providers_set(params)
        if method == "providers/disable":
            return await self._handle_providers_disable(params)
        if method == "agents/list":
            return self._handle_agents_list(params)
        if method == "agents/use":
            return self._handle_agents_use(params)
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications."""
        pass

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

    async def list_sessions(self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any) -> ListSessionsResponse:
        return ListSessionsResponse(sessions=[])

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> Any:
        return None

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> Any:
        return None

    async def authenticate(self, method_id: str, **kwargs: Any) -> Any:
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

    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        self._sessions.pop(session_id, None)
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        pass

    def _handle_agents_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """List available agents grouped by task."""
        cwd = params.get("cwd", ".")
        registry = AgentRegistry(base_dir=cwd)
        grouped = registry.list_by_task()
        return {"agents": {task.value: [{"name": a.name, "description": a.description} for a in agents] for task, agents in grouped.items() if agents}}

    def _handle_agents_use(self, params: dict[str, Any]) -> dict[str, Any]:
        """Set the active agent for a task."""
        task_str = params.get("task", "")
        name = params.get("name", "")
        cwd = params.get("cwd", ".")

        try:
            task = AgentTask(task_str)
        except ValueError:
            return {"status": "error", "message": f"Unknown task: {task_str}"}

        registry = AgentRegistry(base_dir=cwd)
        agent = registry.get(name)
        if agent is None:
            return {"status": "error", "message": f"Agent not found: {name}"}

        config_key = _AGENT_CONFIG_KEYS.get(task)
        if config_key is None:
            return {"status": "error", "message": f"No config key for task: {task.value}"}
        config_path = Path(cwd) / KNOX_CONFIG_FILE
        set_config_file_value(config_path, config_key, name)
        return {"status": "ok", "task": task.value, "agent": name}

    def _make_llm_client(self, session_id: str) -> str | None:
        """Get the model name from config, or *None* if no model is set."""
        try:
            session = self._sessions.get(session_id, {})
            cwd = session.get("cwd", ".")
            cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: cwd})
            return cfg.get(COMPILE_LLM_DEFAULT_MODEL) or None
        except Exception:
            logger.debug("Model lookup failed — degraded mode", exc_info=True)
            return None

    async def _run_conversational(
        self,
        text: str,
        session_id: str,
        model: str,
    ) -> PromptResponse:
        """Run the conversational agent loop."""
        session = self._sessions.get(session_id, {})
        cwd = session.get(SESSION_KEY_CWD, ".")

        # Infer task from slash command, or default to generic
        cmd = parse_slash_command(text)
        task = _SLASH_TASK_MAP.get(cmd.name, AgentTask.GENERIC) if cmd else AgentTask.GENERIC
        user_message = " ".join(cmd.args) if cmd and cmd.args else text

        registry = AgentRegistry(base_dir=cwd)
        definition = AgentDefinition.from_toml(self._agent_file) if self._agent_file else registry.active_for(task, {})

        agent = ConversationalAgent(definition, model, base_dir=cwd)
        return await agent.run(user_message, session_id, self._conn)

    async def _advertise_commands(self, session_id: str) -> None:
        """Send available_commands_update with the current slash commands."""
        if self._conn is None:
            return
        commands = [AvailableCommand(name=f"/knox {c.name}", description=c.description) for c in KNOX_COMMANDS]
        update = AvailableCommandsUpdate(
            availableCommands=commands,
            sessionUpdate=SESSION_UPDATE_AVAILABLE_COMMANDS,
        )
        await self._conn.session_update(session_id=session_id, update=update)

    async def _dispatch_slash(self, cmd: SlashCommand, session_id: str) -> PromptResponse:
        """Dispatch a parsed slash command to the appropriate handler."""
        if cmd.name == "help":
            await self._send_text(session_id, HELP_TEXT)
            return PromptResponse(stopReason=STOP_REASON_END_TURN)

        if cmd.name == "status":
            return await self._handle_status(session_id)

        if cmd.name in ("compile", "query", "capture"):
            return await self._run_knox_cli(cmd, session_id)

        await self._send_text(session_id, f"Unknown command: ``{cmd.name}``. Use ``/knox help``.")
        return PromptResponse(stopReason=STOP_REASON_END_TURN)

    async def _handle_status(self, session_id: str) -> PromptResponse:
        """Show vault health and cache stats."""
        await self._send_text(session_id, "KNOX agent is running. Use ``/knox compile`` to compile a wiki.")
        return PromptResponse(stopReason=STOP_REASON_END_TURN)

    async def _run_knox_cli(self, cmd: SlashCommand, session_id: str) -> PromptResponse:
        """Run a KNOX CLI command in a subprocess and return the output.

        For ``compile``, streams progress as ``session/update`` notifications.
        """
        use_json_progress = cmd.name == "compile"
        knox_args = [sys.executable, "-m", "knox", cmd.name, *cmd.args]
        if use_json_progress:
            knox_args.append("--json-progress")
        for key, val in cmd.kwargs.items():
            knox_args.append(f"--{key}")
            if val:
                knox_args.append(val)

        proc = await asyncio.create_subprocess_exec(
            *knox_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if not use_json_progress:
            stdout, stderr = await proc.communicate()
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += f"\n\nStderr:\n{stderr.decode('utf-8', errors='replace')}"
            await self._send_text(session_id, output or "(no output)")
            return PromptResponse(stopReason=STOP_REASON_END_TURN)

        # Stream compile progress line by line.
        final_message = ""
        message_id = f"{MSG_ID_PREFIX}{int(time.time() * 1000)}"
        stdout = proc.stdout
        if stdout is None:
            await self._send_text(session_id, "(no stdout from subprocess)")
            return PromptResponse(stopReason=STOP_REASON_END_TURN)
        while True:
            line = await stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue
            try:
                event = json.loads(decoded)
            except json.JSONDecodeError:
                continue
            evt_type = event.get(PROGRESS_EVENT)
            if evt_type == PROGRESS_TYPE_PROGRESS:
                await self._send_progress_chunk(session_id, message_id, event)
            elif evt_type == PROGRESS_TYPE_DONE:
                final_message = event.get(PROGRESS_MESSAGE, "")
            elif evt_type == PROGRESS_TYPE_ERROR:
                final_message = event.get(PROGRESS_MESSAGE, "Compilation failed")

        await proc.wait()
        stderr_data = await proc.stderr.read() if proc.stderr else b""
        stderr_text = stderr_data.decode("utf-8", errors="replace").strip()

        output_parts = [final_message] if final_message else []
        if stderr_text:
            output_parts.append(f"Stderr:\n{stderr_text}")
        output = "\n\n".join(output_parts) if output_parts else "(no output)"

        await self._send_text(session_id, output)
        return PromptResponse(stopReason=STOP_REASON_END_TURN)

    async def _send_text(self, session_id: str, text: str) -> None:
        """Send a text notification as an AgentMessageChunk."""
        if self._conn is None:
            return
        try:
            message_id = f"{MSG_ID_PREFIX}{int(time.time() * 1000)}"
            chunk = AgentMessageChunk(
                content=TextContentBlock(type=CONTENT_TYPE_TEXT, text=text),
                messageId=message_id,
                sessionUpdate=SESSION_UPDATE_MESSAGE_CHUNK,
            )
            await self._conn.session_update(session_id=session_id, update=chunk)
        except Exception:
            logger.debug("Failed to send text notification", exc_info=True)

    async def _send_progress_chunk(self, session_id: str, message_id: str, event: dict[str, Any]) -> None:
        """Forward a progress event as an ``AgentMessageChunk`` notification."""
        if self._conn is None:
            return
        phase = event.get(PROGRESS_PHASE, "")
        current = event.get(PROGRESS_CURRENT, 0)
        total = event.get(PROGRESS_TOTAL, 0)
        detail = event.get(PROGRESS_MESSAGE, "")
        pct = f" {int(current * 100 / total)}%" if total else ""
        text = f"[{phase}] {detail}{pct}".strip()
        chunk = AgentMessageChunk(
            content=TextContentBlock(type=CONTENT_TYPE_TEXT, text=text),
            messageId=message_id,
            sessionUpdate=SESSION_UPDATE_MESSAGE_CHUNK,
        )
        await self._conn.session_update(session_id=session_id, update=chunk)

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
