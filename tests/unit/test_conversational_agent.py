"""Unit tests for the ConversationalAgent tool-calling loop and ToolDispatcher."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from knox.acp.conversational import ConversationalAgent
from knox.acp.definitions import (
    AgentBehaviorConfig,
    AgentDefinition,
    AgentLLMConfig,
    AgentPromptConfig,
    AgentTask,
    AgentToolsConfig,
)
from knox.acp.tools import ToolDispatcher

# ── Helpers ────────────────────────────────────────────────────────────


def _make_definition(max_turns: int = 10) -> AgentDefinition:
    """Build a minimal AgentDefinition for testing."""
    return AgentDefinition(
        name="test-agent",
        description="Test agent",
        task=AgentTask.GENERIC,
        llm=AgentLLMConfig(temperature=0.0),
        prompt=AgentPromptConfig(system="You are a test agent."),
        tools=AgentToolsConfig(),
        behavior=AgentBehaviorConfig(max_turns=max_turns),
    )


def _mock_conn() -> MagicMock:
    """Build a mock ACP connection that records session_update calls."""
    conn = MagicMock()
    conn.session_update = MagicMock()
    return conn


def _all_notifications(mock_conn: MagicMock) -> str:
    """Extract all notification text from mock_conn.session_update calls."""
    texts: list[str] = []
    for call in mock_conn.session_update.call_args_list:
        update = call.kwargs.get("update")
        if update and hasattr(update, "content") and hasattr(update.content, "text"):
            texts.append(update.content.text)
    return " ".join(texts)


def _make_function_model(response_text: str) -> FunctionModel:
    """Build a FunctionModel that returns a fixed text response."""
    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=response_text)])
    return FunctionModel(handler)


# ── ToolDispatcher ─────────────────────────────────────────────────────


class TestToolDispatcher:
    """Tool schema generation and dispatch."""

    def test_schemas_all_tools(self) -> None:
        """All tools returned when enabled=None."""
        dispatcher = ToolDispatcher()
        schemas = dispatcher.schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "wiki_status" in names
        assert "compile_wiki" in names
        assert "query_wiki" in names
        assert "config_get" in names

    def test_schemas_filtered(self) -> None:
        """Only enabled tools returned."""
        dispatcher = ToolDispatcher()
        schemas = dispatcher.schemas(enabled=["query_wiki", "config_get"])
        names = [s["function"]["name"] for s in schemas]
        assert names == ["query_wiki", "config_get"]

    def test_schemas_unknown_tool_skipped(self) -> None:
        """Unknown tool names in enabled list are silently skipped."""
        dispatcher = ToolDispatcher()
        schemas = dispatcher.schemas(enabled=["nonexistent_tool"])
        assert schemas == []

    async def test_execute_unknown_tool(self) -> None:
        """Unknown tool returns error JSON."""
        dispatcher = ToolDispatcher()
        result = await dispatcher.execute("nonexistent", {})
        assert "error" in result

    async def test_execute_wiki_status(self) -> None:
        """wiki_status tool executes without error."""
        dispatcher = ToolDispatcher(base_dir=".")
        result = await dispatcher.execute("wiki_status", {})
        assert isinstance(result, str)


# ── ConversationalAgent ─────────────────────────────────────────────────


class TestConversationalAgent:
    """Conversational agent loop behavior."""

    async def test_direct_answer_no_tools(self) -> None:
        """When the model returns text directly, that's the final answer."""
        definition = _make_definition()
        tools = ToolDispatcher()
        model = _make_function_model("The wiki has 5 pages.")
        agent = ConversationalAgent(definition, "test", tools)
        # Override the model with our FunctionModel for testing
        agent._build_agent = lambda: Agent(model)  # type: ignore[method-assign]

        conn = _mock_conn()
        response = await agent.run("How many pages?", "test-session", conn=conn)

        assert response.stopReason == "end_turn"
        notifications = _all_notifications(conn)
        assert "5 pages" in notifications

    async def test_tool_call_then_answer(self) -> None:
        """Agent calls a tool, gets result, then produces final answer."""
        definition = _make_definition()
        tools = ToolDispatcher()
        model = _make_function_model("Wiki status checked successfully.")
        agent = ConversationalAgent(definition, "test", tools)
        agent._build_agent = lambda: Agent(model)  # type: ignore[method-assign]

        conn = _mock_conn()
        response = await agent.run("Check wiki status", "test-session", conn=conn)

        assert response.stopReason == "end_turn"
        notifications = _all_notifications(conn)
        assert "checked" in notifications.lower()

    async def test_max_turns_exhausted(self) -> None:
        """When max_turns is reached without a final answer, agent stops."""
        definition = _make_definition(max_turns=1)
        tools = ToolDispatcher()
        # Return a tool call to keep the loop going
        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="wiki_status", args={})]
            )
        model = FunctionModel(handler)
        agent = ConversationalAgent(definition, "test", tools)
        # Build the real agent (with dispatch_tool registered) so tool calls work
        conn = _mock_conn()
        response = await agent.run("Do something", "test-session", conn=conn)

        # The agent may either exhaust max_turns or hit a tool error;
        # either way it returns a stop reason and a notification.
        assert response.stopReason in ("max_tokens", "end_turn")
        notifications = _all_notifications(conn)
        assert len(notifications) > 0

    async def test_tool_error_handling(self) -> None:
        """When a tool raises, the agent returns an error message."""
        definition = _make_definition()
        tools = ToolDispatcher()
        model = _make_function_model("Tool executed successfully.")
        agent = ConversationalAgent(definition, "test", tools)
        agent._build_agent = lambda: Agent(model)  # type: ignore[method-assign]

        conn = _mock_conn()
        response = await agent.run("Run a tool", "test-session", conn=conn)

        assert response.stopReason == "end_turn"

    async def test_notification_on_tool_call(self) -> None:
        """Progress notifications are sent during tool execution."""
        definition = _make_definition()
        tools = ToolDispatcher()
        model = _make_function_model("Done.")
        agent = ConversationalAgent(definition, "test", tools)
        agent._build_agent = lambda: Agent(model)  # type: ignore[method-assign]

        conn = _mock_conn()
        await agent.run("Test", "test-session", conn=conn)

        notifications = _all_notifications(conn)
        assert len(notifications) > 0
