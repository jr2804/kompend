"""Unit tests for the ConversationalAgent tool-calling loop and ToolDispatcher."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import litellm
import pytest

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


def _mock_litellm_response(content: str = "", tool_calls: list | None = None) -> MagicMock:
    """Build a mock litellm response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].finish_reason = "stop"
    if tool_calls:
        mock.choices[0].message.tool_calls = tool_calls
    else:
        mock.choices[0].message.tool_calls = []
    mock.usage = MagicMock(prompt_tokens=0, completion_tokens=0)
    return mock


class _MockToolCall:
    """Mock litellm tool call for testing."""

    def __init__(self, id: str, name: str, arguments: str = "{}") -> None:
        class _Function:
            def __init__(self, name: str, arguments: str) -> None:
                self.name = name
                self.arguments = arguments

        self.id = id
        self.function = _Function(name, arguments)
        self.name = name
        self.arguments = arguments


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
        schemas = dispatcher.schemas(enabled=["query_wiki", "nonexistent"])
        names = [s["function"]["name"] for s in schemas]
        assert names == ["query_wiki"]

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self) -> None:
        """Unknown tool returns error dict."""
        dispatcher = ToolDispatcher()
        result = await dispatcher.execute("nonexistent", {})
        data = json.loads(result)
        assert data["status"] == "error"
        assert "Unknown tool" in data["message"]


# ── ConversationalAgent ────────────────────────────────────────────────


class TestConversationalAgent:
    """The tool-calling conversational loop."""

    @pytest.mark.asyncio
    async def test_direct_answer_no_tools(self) -> None:
        """LLM returns content immediately → final answer sent as chunk."""
        definition = _make_definition()
        tools = ToolDispatcher()
        agent = ConversationalAgent(definition, "test-model", tools)

        conn = _mock_conn()
        with patch.object(litellm, "completion", return_value=_mock_litellm_response(content="The wiki has 42 pages.")):
            response = await agent.run("How many pages?", "test-session", conn=conn)
        assert response.stop_reason == "end_turn"
        assert "42 pages" in _all_notifications(conn)

    @pytest.mark.asyncio
    async def test_tool_call_then_answer(self) -> None:
        """LLM calls a tool, gets result, then answers."""
        definition = _make_definition()
        tools = ToolDispatcher()
        conn = _mock_conn()

        with patch.object(tools, "execute", return_value='{"status": "ok", "pages": 42}'):
            agent = ConversationalAgent(definition, "test-model", tools)
            with patch.object(litellm, "completion") as mock_completion:
                mock_completion.side_effect = [
                    _mock_litellm_response(content="", tool_calls=[_MockToolCall(id="call_1", name="wiki_status")]),
                    _mock_litellm_response(content="The wiki is compiled with 42 pages."),
                ]
                response = await agent.run("Status?", "test-session", conn=conn)

        assert response.stop_reason == "end_turn"
        assert "42 pages" in _all_notifications(conn)

    @pytest.mark.asyncio
    async def test_max_turns_exhausted(self) -> None:
        """Agent stops after max_turns with an explanation."""
        definition = _make_definition(max_turns=2)
        tools = ToolDispatcher()
        conn = _mock_conn()

        with patch.object(tools, "execute", return_value='{"status": "ok"}'):
            agent = ConversationalAgent(definition, "test-model", tools)
            with patch.object(
                litellm, "completion", return_value=_mock_litellm_response(content="", tool_calls=[_MockToolCall(id="call_x", name="wiki_status")])
            ):
                response = await agent.run("Keep calling tools", "test-session", conn=conn)

        assert response.stop_reason == "max_tokens"
        assert "maximum" in _all_notifications(conn).lower()

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_turn(self) -> None:
        """LLM requests multiple tools in a single turn."""
        definition = _make_definition()
        tools = ToolDispatcher()
        conn = _mock_conn()
        execute_count = 0

        async def _mock_execute(name: str, args: dict) -> str:  # noqa: ANN001
            nonlocal execute_count
            execute_count += 1
            return f'{{"status": "ok", "tool": "{name}"}}'

        with patch.object(tools, "execute", side_effect=_mock_execute):
            agent = ConversationalAgent(definition, "test-model", tools)
            with patch.object(litellm, "completion") as mock_completion:
                mock_completion.side_effect = [
                    _mock_litellm_response(
                        content="",
                        tool_calls=[
                            _MockToolCall(id="call_1", name="wiki_status"),
                            _MockToolCall(id="call_2", name="config_get", arguments='{"key": "compile.llm.default_model"}'),
                        ],
                    ),
                    _mock_litellm_response(content="Done checking status and config."),
                ]
                response = await agent.run("Check everything", "test-session", conn=conn)

        assert execute_count == 2
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_progress_notification_sent(self) -> None:
        """Progress chunks are sent via the ACP connection."""
        definition = _make_definition()
        tools = ToolDispatcher()
        conn = _mock_conn()

        with patch.object(tools, "execute", return_value='{"status": "ok"}'):
            agent = ConversationalAgent(definition, "test-model", tools)
            with patch.object(litellm, "completion") as mock_completion:
                mock_completion.side_effect = [
                    _mock_litellm_response(content="", tool_calls=[_MockToolCall(id="call_1", name="wiki_status")]),
                    _mock_litellm_response(content="Done."),
                ]
                await agent.run("Status?", "test-session", conn=conn)

        # At least 3 notifications: "Calling…", "completed", final answer
        assert conn.session_update.call_count >= 3

    @pytest.mark.asyncio
    async def test_no_conn_does_not_crash(self) -> None:
        """Agent works without a connection (silent mode)."""
        definition = _make_definition()
        tools = ToolDispatcher()
        agent = ConversationalAgent(definition, "test-model", tools)

        with patch.object(litellm, "completion", return_value=_mock_litellm_response(content="Answer.")):
            response = await agent.run("Question?", "test-session", conn=None)
        assert response.stop_reason == "end_turn"
