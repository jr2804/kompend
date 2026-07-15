"""Tests for KnoxLiteLLMModel — the pydantic-ai-litellm bridge with KNOX enhancements.

Verifies retry enrichment (valid partial JSON injected into RetryPromptPart
feedback) and schema-aware JSON repair.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from pydantic import BaseModel
from pydantic_ai.messages import TextPart

from knox.agents.litellm_model import KnoxLiteLLMModel


class _FakeSchema(BaseModel):
    name: str
    value: int


class TestRetryEnrichment:
    """Verify retry messages are enriched with valid partial JSON."""

    def test_no_schema_no_enrichment(self) -> None:
        """Without output_schema, retry messages pass through unchanged."""
        model = KnoxLiteLLMModel("test-model")
        messages = [
            {"role": "assistant", "content": '{"name": "x", "value": 1}'},
            {"role": "user", "content": "Validation errors — fix and retry:\n- Field required"},
        ]
        result = model._enrich_retry_messages(messages)
        assert result == messages  # unchanged

    def test_with_schema_enriches_retry(self) -> None:
        """With output_schema, valid partial JSON is injected into retry."""
        model = KnoxLiteLLMModel("test-model", output_schema=_FakeSchema)
        messages = [
            {"role": "assistant", "content": '{"name": "x", "value": 1}'},
            {"role": "user", "content": "Validation errors — fix and retry:\n- Field required"},
        ]
        result = model._enrich_retry_messages(messages)
        assert len(result) == 2
        assert "valid parts extracted" in result[1]["content"]
        assert '"name": "x"' in result[1]["content"]

    def test_no_assistant_before_retry_no_enrichment(self) -> None:
        """Retry without preceding assistant message is not enriched."""
        model = KnoxLiteLLMModel("test-model", output_schema=_FakeSchema)
        messages = [
            {"role": "user", "content": "Validation errors — fix and retry:\n- Field required"},
        ]
        result = model._enrich_retry_messages(messages)
        assert result == messages

    def test_unparseable_assistant_content_no_enrichment(self) -> None:
        """When salvage_json returns None, retry message is unchanged."""
        model = KnoxLiteLLMModel("test-model", output_schema=_FakeSchema)
        messages = [
            {"role": "assistant", "content": "not json at all"},
            {"role": "user", "content": "Validation errors — fix and retry:\n- Field required"},
        ]
        result = model._enrich_retry_messages(messages)
        assert result == messages

    def test_non_retry_user_message_not_enriched(self) -> None:
        """Only messages containing 'Validation errors' are enriched."""
        model = KnoxLiteLLMModel("test-model", output_schema=_FakeSchema)
        messages = [
            {"role": "assistant", "content": '{"name": "x", "value": 1}'},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        result = model._enrich_retry_messages(messages)
        assert result == messages


class TestProcessResponse:
    """Verify schema-aware JSON repair in _process_response."""

    def test_no_schema_passthrough(self) -> None:
        """Without output_schema, response is unchanged."""
        model = KnoxLiteLLMModel("test-model")
        raw = MagicMock()
        raw.choices = [MagicMock()]
        raw.choices[0].message = MagicMock()
        raw.choices[0].message.content = '{"name": "x", "value": 1}'
        raw.choices[0].message.tool_calls = None
        raw.usage = None
        raw.model = "test-model"
        raw.id = None
        raw.created = None

        result = model._process_response(raw)
        parts = result.parts
        assert len(parts) == 1
        assert isinstance(parts[0], TextPart)
        assert parts[0].content == '{"name": "x", "value": 1}'

    def test_with_schema_repairs_json(self) -> None:
        """With output_schema, salvage_json is applied to text parts."""
        model = KnoxLiteLLMModel("test-model", output_schema=_FakeSchema)
        raw = MagicMock()
        raw.choices = [MagicMock()]
        raw.choices[0].message = MagicMock()
        raw.choices[0].message.content = '{"name": "x", "value": 1}'
        raw.choices[0].message.tool_calls = None
        raw.usage = None
        raw.model = "test-model"
        raw.id = None
        raw.created = None

        result = model._process_response(raw)
        parts = result.parts
        assert len(parts) == 1
        # salvage_json should parse and re-serialize (valid JSON stays valid)
        parsed = json.loads(parts[0].content)
        assert parsed["name"] == "x"
        assert parsed["value"] == 1
