"""Tests for KompendModel — the native-model wrapper with kompend enhancements.

Verifies usage tracking, debug saving, and reasoning-effort passthrough.
Salvage and retry enrichment are intentionally absent from KompendModel:
native models use ToolOutput mode for structured output (tool-call
arguments validated directly), so there is no JSON text to repair.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from kompend.agents.kompend_model import KompendModel
from kompend.core.env import EnvVar
from kompend.llm.usage import UsageTracker


class TestUsageTracking:
    @pytest.mark.asyncio
    async def test_records_usage_when_tracker_set(self) -> None:
        tracker = UsageTracker()
        delegate = TestModel()
        # Make the delegate return a fixed response with known usage.
        delegate.custom_output_text = "hi"
        wrapper = KompendModel(delegate, usage_tracker=tracker)

        # Drive a minimal request.
        params = ModelRequestParameters()
        msgs: list[ModelRequest | ModelResponse] = [ModelRequest(parts=[UserPromptPart(content="ping")])]
        await wrapper.request(msgs, None, params)

        assert tracker.prompt_tokens >= 0
        assert tracker.call_count >= 1  # model call recorded

    @pytest.mark.asyncio
    async def test_no_tracker_no_error(self) -> None:
        """Wrapper works without a usage tracker (just skips tracking)."""
        delegate = TestModel()
        wrapper = KompendModel(delegate, usage_tracker=None)
        params = ModelRequestParameters()
        msgs: list[ModelRequest | ModelResponse] = [ModelRequest(parts=[UserPromptPart(content="ping")])]
        resp = await wrapper.request(msgs, None, params)
        assert resp is not None


class TestReasoningEffort:
    def test_no_effort_returns_settings_unchanged(self) -> None:
        wrapper = KompendModel(TestModel(), reasoning_effort=None)
        settings: ModelSettings = {"timeout": 30.0}
        assert wrapper._merge_reasoning(settings) is settings

    def test_effort_injected_as_thinking(self) -> None:
        wrapper = KompendModel(TestModel(), reasoning_effort="low")
        merged = wrapper._merge_reasoning({"timeout": 30.0})
        assert merged is not None
        assert merged["thinking"] == "low"
        assert merged["timeout"] == 30.0

    def test_none_maps_to_false(self) -> None:
        """``reasoning_effort='none'`` disables thinking (``thinking=False``)."""
        wrapper = KompendModel(TestModel(), reasoning_effort="none")
        merged = wrapper._merge_reasoning(None)
        assert merged is not None
        assert merged["thinking"] is False

    def test_thinking_does_not_override_existing_setting(self) -> None:
        """If thinking already in settings, wrapper doesn't clobber it."""
        wrapper = KompendModel(TestModel(), reasoning_effort="low")
        merged = wrapper._merge_reasoning({"thinking": "high"})
        assert merged is not None
        assert merged["thinking"] == "high"

    def test_effort_creates_settings_from_none(self) -> None:
        wrapper = KompendModel(TestModel(), reasoning_effort="medium")
        merged = wrapper._merge_reasoning(None)
        assert merged is not None
        assert merged["thinking"] == "medium"

    def test_unknown_effort_raises(self) -> None:
        """An unknown effort fails at construction instead of warn+ignore."""
        with pytest.raises(ValueError, match="bogus"):
            KompendModel(TestModel(), reasoning_effort="bogus")

    def test_default_member_leaves_settings_untouched(self) -> None:
        """``default`` means "no thinking override" — nothing is injected."""
        wrapper = KompendModel(TestModel(), reasoning_effort="default")
        assert wrapper._merge_reasoning(None) is None  # noqa: SLF001


class TestDelegation:
    def test_model_name_delegates(self) -> None:
        delegate = TestModel()
        wrapper = KompendModel(delegate)
        assert wrapper.model_name == delegate.model_name

    def test_system_delegates(self) -> None:
        delegate = TestModel()
        wrapper = KompendModel(delegate)
        assert wrapper.system == delegate.system

    def test_delegate_property_exposes_inner_model(self) -> None:
        delegate = TestModel()
        wrapper = KompendModel(delegate)
        assert wrapper.delegate is delegate

    @pytest.mark.asyncio
    async def test_request_returns_delegate_response(self) -> None:
        expected = _make_response("test-output")
        delegate = MagicMock(spec=TestModel)
        delegate.request = AsyncMock(return_value=expected)
        delegate.model_name = "test-model"
        wrapper = KompendModel(delegate)
        params = ModelRequestParameters()
        msgs: list[ModelRequest | ModelResponse] = [ModelRequest(parts=[UserPromptPart(content="x")])]
        result = await wrapper.request(msgs, None, params)
        assert result is expected


class TestDebugSaving:
    @pytest.mark.asyncio
    async def test_debug_not_saved_when_no_agent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without an agent reference, debug saving is a no-op even if dir set."""
        monkeypatch.setenv(EnvVar.DEBUG_LLM_DIR, str(tmp_path))
        wrapper = KompendModel(TestModel(), agent=None)
        params = ModelRequestParameters()
        msgs: list[ModelRequest | ModelResponse] = [ModelRequest(parts=[UserPromptPart(content="ping")])]
        await wrapper.request(msgs, None, params)
        # No files written.
        assert list(tmp_path.iterdir()) == []


def _make_response(text: str = "hello", *, input_tokens: int = 10, output_tokens: int = 5) -> ModelResponse:
    """Build a ModelResponse with usage, matching native-model shape."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return ModelResponse(
        parts=[TextPart(content=text)],
        usage=usage,
        model_name="test-model",
    )
