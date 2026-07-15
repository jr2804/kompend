"""Conversational agent with a tool-calling loop.

Driven by an :class:`~knox.acp.definitions.AgentDefinition`, this agent
uses an LLM to reason about user requests and call available tools
(:class:`~knox.acp.tools.ToolDispatcher`) until a final answer is produced.

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

import litellm
from acp import PromptResponse
from acp.schema import AgentMessageChunk, TextContentBlock

from knox.acp._constants import (
    CONTENT_TYPE_TEXT,
    MSG_ID_PREFIX,
    SESSION_UPDATE_MESSAGE_CHUNK,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
)
from knox.acp.definitions import AgentDefinition
from knox.acp.tools import ToolDispatcher
from knox.logging import get_logger

logger = get_logger(__name__)


class ConversationalAgent:
    """Tool-calling conversational loop driven by an AgentDefinition."""

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

        message_id = f"{MSG_ID_PREFIX}{int(time.time() * 1000)}"

        for _turn in range(definition.behavior.max_turns):
            response = litellm.completion(
                model=self._model,
                messages=messages,
                tools=tools_schema,
                temperature=definition.llm.temperature,
                max_tokens=definition.llm.max_tokens,
                timeout=self._timeout,
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls = choice.message.tool_calls or []

            # No tool calls → final answer
            if not tool_calls:
                await self._notify(conn, session_id, message_id, content or "(no response)")
                return PromptResponse(
                    stopReason=STOP_REASON_END_TURN,
                )

            # Build assistant message with tool calls for the conversation
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in tool_calls:
                await self._notify(
                    conn,
                    session_id,
                    message_id,
                    f"Calling {tc.name}…",
                )
                logger.info("Agent tool call: %s(%s)", tc.name, tc.arguments)
                tool_result = await self._tools.execute(tc.function.name, json.loads(tc.function.arguments))
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
            stopReason=STOP_REASON_MAX_TOKENS,
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
                content=TextContentBlock(type=CONTENT_TYPE_TEXT, text=text),
                messageId=message_id,
                sessionUpdate=SESSION_UPDATE_MESSAGE_CHUNK,
            )
            await conn.session_update(session_id=session_id, update=chunk)
        except Exception:
            logger.debug("Failed to send progress notification", exc_info=True)
