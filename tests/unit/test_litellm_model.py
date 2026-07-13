"""Tests for the LitellmModel FunctionModel bridge.

Verifies that the message converter handles PydanticAI's retry protocol
(RetryPromptPart, ToolReturnPart) — without this, schema-validation
retries are silently dropped and the model repeats the same invalid
output.
"""

from __future__ import annotations

from datetime import datetime

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from knox.agents.litellm_model import _messages_to_litellm


class TestMessagesToLitellm:
    """Verify every PydanticAI part type maps to a litellm message."""

    def test_system_prompt(self) -> None:
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content="You are a bot")]),
        ]
        result = _messages_to_litellm(msgs)
        assert result == [{"role": "system", "content": "You are a bot"}]

    def test_user_prompt(self) -> None:
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ]
        result = _messages_to_litellm(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_text(self) -> None:
        msgs: list[ModelMessage] = [
            ModelResponse(parts=[TextPart(content="Hi back")], model_name="test", timestamp=datetime.now()),
        ]
        result = _messages_to_litellm(msgs)
        assert result == [{"role": "assistant", "content": "Hi back"}]

    def test_retry_prompt_part_is_converted(self) -> None:
        """RetryPromptPart must become a user message — not be dropped.

        This is the regression test for the root cause: without handling
        RetryPromptPart, PydanticAI retries were silently lost and the
        model repeated the same invalid output.
        """
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Generate JSON")]),
            ModelResponse(parts=[TextPart(content='{"bad"')], model_name="test", timestamp=datetime.now()),
            ModelRequest(parts=[RetryPromptPart(content="Invalid JSON near line 1")]),
        ]
        result = _messages_to_litellm(msgs)
        assert len(result) == 3
        assert result[2]["role"] == "user"
        assert "Invalid JSON near line 1" in result[2]["content"]

    def test_retry_prompt_part_with_error_list(self) -> None:
        """RetryPromptPart with a list of ErrorDetails is flattened to readable text."""
        errors = [
            {"msg": "Field required", "loc": ("pages", 0, "title")},
            {"msg": "extra fields not permitted", "loc": ("foo",)},
        ]
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[RetryPromptPart(content=errors)]),
        ]
        result = _messages_to_litellm(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Field required" in result[0]["content"]
        assert "extra fields not permitted" in result[0]["content"]

    def test_tool_return_part_is_converted(self) -> None:
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[ToolReturnPart(tool_name="lookup", content="result data", tool_call_id="tc1")]),
        ]
        result = _messages_to_litellm(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "result data" in result[0]["content"]

    def test_full_retry_conversation_roundtrip(self) -> None:
        """Simulate the full PydanticAI retry flow: prompt → bad response → retry → good response."""
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content="Return JSON"), UserPromptPart(content="Make a page")]),
            ModelResponse(parts=[TextPart(content='{"bad"')], model_name="test", timestamp=datetime.now()),
            ModelRequest(parts=[RetryPromptPart(content="JSON decode error at line 1")]),
            ModelResponse(parts=[TextPart(content='{"pages":[]}')], model_name="test", timestamp=datetime.now()),
        ]
        result = _messages_to_litellm(msgs)
        # System + User + Assistant + RetryFeedback + Assistant = 5 messages
        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"  # retry feedback
        assert "JSON decode error" in result[3]["content"]
