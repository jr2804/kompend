"""Conversational agent with a tool-calling loop.

Driven by an :class:`~knox.agent.definitions.AgentDefinition`, this agent
uses an LLM to reason about user requests and call available tools
(:class:`~knox.agent.tools.ToolDispatcher`) until a final answer is produced.

The loop:

1. System prompt + user message → LLM (with tools)
2. If LLM returns tool_calls → execute each → feed results back → repeat
3. If LLM returns content (no tool_calls) → final answer

Bounded by ``AgentDefinition.behavior.max_turns``.
"""

from __future__ import annotations

import json
import time
from typing import Any

from acp import PromptResponse
from acp.schema import AgentMessageChunk, TextContentBlock

from knox.agent.definitions import AgentDefinition
from knox.agent.tools import ToolDispatcher
from knox.llm.client import LLMClient
from knox.llm.roles import LLMRole
from knox.logging import get_logger

logger = get_logger(__name__)


class ConversationalAgent:
    """Tool-calling conversational loop driven by an AgentDefinition."""

    def __init__(
        self,
        definition: AgentDefinition,
        llm_client: LLMClient,
        tools: ToolDispatcher,
    ) -> None:
        self._definition = definition
        self._llm = llm_client
        self._tools = tools

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
        tools_schema = ToolDispatcher.schemas(definition.tools.enabled)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": definition.prompt.system},
            {"role": "user", "content": user_message},
        ]

        message_id = f"msg_{int(time.time() * 1000)}"

        for _turn in range(definition.behavior.max_turns):
            result = self._llm.complete_with_tools(
                messages,
                tools=tools_schema,
                role=LLMRole.AGENT,
                temperature=definition.llm.temperature,
                max_tokens=definition.llm.max_tokens,
            )

            # No tool calls → final answer
            if not result.tool_calls:
                await self._notify(conn, session_id, message_id, result.content or "(no response)")
                return PromptResponse(
                    stop_reason="end_turn",
                )

            # Build assistant message with tool calls for the conversation
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in result.tool_calls:
                await self._notify(
                    conn,
                    session_id,
                    message_id,
                    f"Calling {tc.name}…",
                )
                logger.info("Agent tool call: %s(%s)", tc.name, tc.arguments)
                tool_result = await self._tools.execute(tc.name, tc.arguments)
                await self._notify(
                    conn,
                    session_id,
                    message_id,
                    f"{tc.name} completed",
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )

        # Max turns exhausted
        logger.warning(
            "Agent %s reached max_turns=%d without a final answer",
            definition.name,
            definition.behavior.max_turns,
        )
        await self._notify(
            conn,
            session_id,
            message_id,
            "I reached the maximum number of reasoning steps without producing a final answer. Please try rephrasing your request.",
        )
        return PromptResponse(
            stop_reason="max_tokens",
        )

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
                content=TextContentBlock(type="text", text=text),
                message_id=message_id,
                session_update="agent_message_chunk",
            )
            await conn.session_update(session_id=session_id, update=chunk)
        except Exception:
            logger.debug("Failed to send progress notification", exc_info=True)
