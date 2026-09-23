"""Tests for the LLM response post-processing pipeline and its registry wiring.

``default_pipeline`` is now built via ``pipeline_from_registry`` with explicit
ordered names. These tests pin the resulting processor list — types *and*
order — to the historical contract, so the registry indirection can never
silently reorder or drop a processor.
"""

from __future__ import annotations

import pytest

from knox.llm.postprocess import (
    ResponsePipeline,
    SurrogateScrubber,
    default_pipeline,
    list_registered_processors,
    pipeline_from_registry,
)


class _StubDebugAgent:
    """Minimal ``_DebugSource``-shaped object (only DebugSaver construction needs it)."""

    stage = "test"
    current_source_id = "src-1"
    _last_debug_file = None


class TestDefaultPipelineOrder:
    def test_order_matches_contract(self) -> None:
        """Registry-built pipeline preserves the historical order exactly.

        ``SurrogateScrubber`` must stay first (memory #618): ``ToolCallRepair``
        calls ``json.dumps`` internally, which crashes on lone surrogates.
        """
        pipeline = default_pipeline(
            schema_resolver=lambda: None,
            agent=None,
            model_name="test-model",
            usage_tracker=None,
            messages=[],
        )
        assert _processor_types(pipeline) == [
            "SurrogateScrubber",
            "ToolCallRepair",
            "ThinkingPartRepair",
            "UsageRecorder",
        ]
        assert isinstance(pipeline._processors[0], SurrogateScrubber)

    def test_debug_saver_appended_only_when_agent_set(self) -> None:
        pipeline = default_pipeline(
            schema_resolver=lambda: None,
            agent=_StubDebugAgent(),
            model_name="test-model",
            usage_tracker=None,
            messages=[],
        )
        assert _processor_types(pipeline)[-1] == "DebugSaver"


class TestRegistry:
    def test_builtin_processors_registered(self) -> None:
        assert list_registered_processors() == [
            "surrogate_scrub",
            "tool_call_repair",
            "thinking_part_repair",
            "usage_recorder",
            "debug_saver",
        ]

    def test_pipeline_from_registry_raises_on_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown post-processor"):
            pipeline_from_registry(["not_a_processor"])


def _processor_types(pipeline: ResponsePipeline) -> list[str]:
    return [type(p).__name__ for p in pipeline._processors]
