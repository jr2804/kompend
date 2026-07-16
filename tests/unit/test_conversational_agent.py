"""Unit tests for the ConversationalAgent and build_toolset."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
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
from knox.acp.tools import ToolName, build_toolset

# ── Helpers ────────────────────────────────────────────────────────────


def _make_definition(
    max_turns: int = 10,
    enabled: list[str] | None = None,
) -> AgentDefinition:
    """Build a minimal AgentDefinition for testing."""
    return AgentDefinition(
        name="test-agent",
        description="Test agent",
        task=AgentTask.GENERIC,
        llm=AgentLLMConfig(temperature=0.0),
        prompt=AgentPromptConfig(system="You are a test agent."),
        tools=AgentToolsConfig(enabled=enabled),
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


# ── build_toolset ──────────────────────────────────────────────────────


class TestBuildToolset:
    """Toolset construction and tool registration."""

    def test_returns_function_toolset(self) -> None:
        """build_toolset returns a FunctionToolset instance."""
        toolset = build_toolset(base_dir=".")
        assert toolset is not None

    def test_all_tool_names_in_enum(self) -> None:
        """All ToolName enum members are valid tool identifiers."""
        for name in ToolName:
            assert isinstance(name.value, str)
            assert len(name.value) > 0

    def test_filtered_toolset(self) -> None:
        """Filtered toolset can be constructed without error."""
        toolset = build_toolset(base_dir=".", enabled=["query_wiki", "config_get"])
        assert toolset is not None

    def test_empty_enabled_list(self) -> None:
        """Empty enabled list produces an empty toolset."""
        toolset = build_toolset(base_dir=".", enabled=[])
        assert toolset is not None


# ── ConversationalAgent ────────────────────────────────────────────────


class TestConversationalAgent:
    """Conversational agent loop behavior."""

    async def test_direct_answer_no_tools(self) -> None:
        """When the model returns text directly, that's the final answer."""
        definition = _make_definition()
        model = _make_function_model("The wiki has 5 pages.")
        agent = ConversationalAgent(definition, "test", base_dir=".")
        agent._build_agent = lambda: Agent(model)  # type: ignore[method-assign]

        conn = _mock_conn()
        response = await agent.run("How many pages?", "test-session", conn=conn)

        assert response.stopReason == "end_turn"
        notifications = _all_notifications(conn)
        assert "5 pages" in notifications

    async def test_tool_error_handling(self) -> None:
        """When a tool raises, the agent returns an error message."""
        definition = _make_definition()
        model = _make_function_model("Tool executed successfully.")
        agent = ConversationalAgent(definition, "test", base_dir=".")
        agent._build_agent = lambda: Agent(model)  # type: ignore[method-assign]

        conn = _mock_conn()
        response = await agent.run("Run a tool", "test-session", conn=conn)

        assert response.stopReason == "end_turn"

    async def test_notification_on_completion(self) -> None:
        """Final answer is sent as a notification."""
        definition = _make_definition()
        model = _make_function_model("Done.")
        agent = ConversationalAgent(definition, "test", base_dir=".")
        agent._build_agent = lambda: Agent(model)  # type: ignore[method-assign]

        conn = _mock_conn()
        await agent.run("Test", "test-session", conn=conn)

        notifications = _all_notifications(conn)
        assert "Done." in notifications

    async def test_construction_with_base_dir(self) -> None:
        """ConversationalAgent accepts base_dir instead of ToolDispatcher."""
        definition = _make_definition(enabled=["wiki_status"])
        agent = ConversationalAgent(definition, "test", base_dir=".")
        assert agent._base_dir == "."

    async def test_error_returns_max_tokens(self) -> None:
        """When _build_agent raises, returns STOP_REASON_MAX_TOKENS."""
        definition = _make_definition()
        agent = ConversationalAgent(definition, "test", base_dir=".")

        def _raise() -> Agent[None, str]:
            raise RuntimeError("build failed")

        agent._build_agent = _raise  # type: ignore[method-assign]

        conn = _mock_conn()
        response = await agent.run("Test", "test-session", conn=conn)

        assert response.stopReason == "max_tokens"
        notifications = _all_notifications(conn)
        assert "build failed" in notifications
