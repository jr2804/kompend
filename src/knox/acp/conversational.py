"""Conversational agent with a PydanticAI-driven tool-calling loop.

Driven by an :class:`~knox.acp.definitions.AgentDefinition`, this agent
uses a PydanticAI :class:`~pydantic_ai.Agent` with a single generic tool
that dispatches to :class:`~knox.acp.tools.ToolDispatcher`.  PydanticAI
handles the tool-calling loop, retry budgeting, and message history.

The system prompt is built from the agent definition's system prompt plus
a rendered tool manifest (name, description, parameter schema) so the LLM
knows what tools are available and how to call them.
"""

from __future__ import annotations

import json
import time
from typing import Any

from acp import PromptResponse
from acp.schema import AgentMessageChunk, TextContentBlock
from pydantic_ai import Agent

from knox.acp._constants import (
    CONTENT_TYPE_TEXT,
    MSG_ID_PREFIX,
    SESSION_UPDATE_MESSAGE_CHUNK,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
)
from knox.acp.definitions import AgentDefinition
from knox.acp.tools import ToolDispatcher
from knox.agents.knox_model import KnoxModel
from knox.llm.models import build_model
from knox.llm.usage import UsageTracker
from knox.logging import get_logger

logger = get_logger(__name__)


def _build_tool_manifest(definition: AgentDefinition) -> str:
    """Render a human-readable tool manifest for the system prompt.

    Lists each enabled tool with its name, description, and parameter
    schema so the LLM knows how to call it via the generic dispatch tool.
    """
    schemas = ToolDispatcher.schemas(definition.tools.enabled)
    lines: list[str] = ["## Available Tools", ""]
    for s in schemas:
        func = s.get("function", s)
        name = func.get("name", "?")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"### {name}")
        lines.append(f"Description: {desc}")
        props = params.get("properties", {})
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                required = " (required)" if pname in params.get("required", []) else ""
                lines.append(f"  - {pname}: {ptype}{required} — {pdesc}")
        lines.append("")
    return "\n".join(lines)


class ConversationalAgent:
    """Tool-calling conversational loop driven by a PydanticAI Agent."""

    def __init__(
        self,
        definition: AgentDefinition,
        model: str,
        tools: ToolDispatcher,
        timeout: float = 120.0,
    ) -> None:
        self._definition = definition
        self._model = model
        self._tools = tools
        self._timeout = timeout
        self._usage_tracker = UsageTracker()
        # Notification state (set per run() call)
        self._notify_conn: Any = None
        self._notify_session_id: str = ""
        self._notify_message_id: str = ""

    async def run(
        self,
        user_message: str,
        session_id: str,
        conn: Any | None = None,
    ) -> PromptResponse:
        """Run the conversational loop for a single user prompt.

        Args:
            user_message: The user's text prompt.
            session_id: ACP session ID (for progress notifications).
            conn: ACP connection for sending progress chunks (*None* = silent).

        Returns:
            :class:`PromptResponse` with the final answer.
        """
        definition = self._definition
        self._notify_conn = conn
        self._notify_session_id = session_id
        self._notify_message_id = f"{MSG_ID_PREFIX}{int(time.time() * 1000)}"

        # Build the system prompt with tool manifest.
        system_prompt = definition.prompt.system
        tool_manifest = _build_tool_manifest(definition)
        full_system = f"{system_prompt}\n\n{tool_manifest}"

        # Build the PydanticAI Agent lazily (per-run so tools can vary).
        agent = self._build_agent()

        try:
            result = await agent.run(
                user_message,
                instructions=full_system,
                usage_limits=None,
            )
            content = result.output or "(no response)"
            await self._notify(conn, session_id, self._notify_message_id, content)
            return PromptResponse(
                stopReason=STOP_REASON_END_TURN,
            )
        except Exception as exc:
            logger.warning(
                "Conversational agent '%s' failed: %s",
                definition.name,
                exc,
            )
            error_msg = f"I encountered an error: {exc}"
            await self._notify(conn, session_id, self._notify_message_id, error_msg)
            return PromptResponse(
                stopReason=STOP_REASON_MAX_TOKENS,
            )

    def _build_agent(self) -> Any:
        """Build a PydanticAI Agent with a single generic dispatch tool."""
        model = KnoxModel(
            build_model(self._model),
            usage_tracker=self._usage_tracker,
        )
        agent = Agent(
            model,
            system_prompt="You are a helpful assistant with access to tools.",
            retries=2,
        )

        @agent.tool_plain
        async def dispatch_tool(
            tool_name: str,
            tool_args_json: str = "{}",
        ) -> str:
            """Call a tool by name with JSON-encoded arguments.

            Available tools are described in the system prompt under
            '## Available Tools'.  Set *tool_name* to the tool name and
            *tool_args_json* to a JSON object with the tool's parameters.
            """
            args = json.loads(tool_args_json) if tool_args_json.strip() else {}
            logger.info("Agent tool call: %s(%s)", tool_name, args)
            await self._notify(
                self._notify_conn,
                self._notify_session_id,
                self._notify_message_id,
                f"Calling {tool_name}…",
            )
            result = await self._tools.execute(tool_name, args)
            await self._notify(
                self._notify_conn,
                self._notify_session_id,
                self._notify_message_id,
                f"{tool_name} completed",
            )
            return result

        return agent

    async def _notify(  # noqa: PLR6301
        self,
        conn: Any | None,
        session_id: str,
        message_id: str,
        text: str,
    ) -> None:
        """Send a progress chunk as an AgentMessageChunk notification."""
        if conn is None:
            return
        try:
            chunk = AgentMessageChunk(
                content=TextContentBlock(type=CONTENT_TYPE_TEXT, text=text),
                messageId=message_id,
                sessionUpdate=SESSION_UPDATE_MESSAGE_CHUNK,
            )
            await conn.session_update(session_id=session_id, update=chunk)
        except Exception:
            logger.debug("Failed to send progress notification", exc_info=True)
