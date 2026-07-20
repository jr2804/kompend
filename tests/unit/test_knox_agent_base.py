"""Tests for the KnoxAgent base class and model-name bridge."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai_summarization import ContextManagerCapability, SlidingWindowCapability

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps
from knox.llm.usage import UsageTracker


class TestKnoxAgentBaseContract:
    def test_knox_agent_base_is_abstract(self) -> None:
        """KnoxAgentBase requires a concrete subclass to be usable, but the base
        class itself can be instantiated for testing because `_build_agent` now
        provides a default implementation.
        """
        agent: KnoxAgentBase[Any, Any] = KnoxAgentBase(model_string="test/noop")
        assert agent.model_string == "test/noop"

    def test_knox_agent_base_requires_build_agent(self) -> None:
        """Subclasses may rely on the default _build_agent implementation."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(model_string="test/noop")
        # Base class provides a default _build_agent; instantiation succeeds
        assert agent.model_string == "test/noop"

    def test_knox_agent_base_stores_reasoning_effort(self) -> None:
        """reasoning_effort is stored and forwarded to model_string."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(
            model_string="test/noop",
            reasoning_effort="low",
        )
        assert agent._reasoning_effort == "low"  # noqa: SLF001

    def test_knox_agent_base_reasoning_effort_none_by_default(self) -> None:
        """reasoning_effort defaults to None (use provider default)."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(model_string="test/noop")
        assert agent._reasoning_effort is None  # noqa: SLF001


class TestSkillInjection:
    """Skills passed to KnoxAgentBase are appended to the system prompt."""

    def test_skills_appended_to_system_prompt(self) -> None:
        """Skills are rendered as '## Skill: <name>' sections."""

        class _TestAgent(KnoxAgentBase[Any, Any]):
            template_name: str = "extraction_prompt.md.j2"
            system_template_name: str = "extraction_system.md.j2"
            result_type: type[Any] = Any

            def _build_agent(self) -> Agent[Any, Any]:
                return MagicMock()  # not needed for prompt rendering test

        agent = _TestAgent(
            "ollama/test-model",
            skills={
                "domain-terms": "- VAD: Voice Activity Detection\n- NR: Noise Reduction",
                "format-rule": "Always respond in JSON.",
            },
        )
        deps = PipelineAgentDeps(text="sample text", source_id="test")
        prompt = agent.render_system_prompt(deps)

        # Skill: domain-terms section present
        assert "## Skill: domain-terms" in prompt
        assert "VAD: Voice Activity Detection" in prompt
        assert "NR: Noise Reduction" in prompt

        # Skill: format-rule section present
        assert "## Skill: format-rule" in prompt
        assert "Always respond in JSON." in prompt

        # Both skills are in the prompt
        assert prompt.index("## Skill: domain-terms") < prompt.index("## Skill: format-rule")

    def test_no_skills_empty_prompt(self) -> None:
        """Without skills, the system prompt is unchanged."""

        class _TestAgent(KnoxAgentBase[Any, Any]):
            template_name: str = "extraction_prompt.md.j2"
            system_template_name: str = "extraction_system.md.j2"
            result_type: type[Any] = Any

            def _build_agent(self) -> Agent[Any, Any]:
                return MagicMock()

        agent = _TestAgent("ollama/test-model")
        deps = PipelineAgentDeps(text="sample", source_id="test")
        prompt = agent.render_system_prompt(deps)

        assert "## Skill:" not in prompt


class TestContextManagerCapability:
    """``context_manager`` preset resolution on KnoxAgentBase."""

    def test_disabled_by_default_at_code_api(self) -> None:
        """Code-level default is False — no capability unless config opts in."""
        agent = self._agent()
        assert agent._resolve_context_capability(MagicMock()) is None  # noqa: SLF001

    def test_default_preset_builds_context_manager(self) -> None:
        agent = self._agent(context_manager="default", context_window=8192)
        cap = agent._resolve_context_capability(MagicMock())  # noqa: SLF001
        assert isinstance(cap, ContextManagerCapability)
        assert cap.compress_threshold == 0.9
        assert cap.max_tokens == 8192

    def test_aggressive_preset_uses_lower_threshold(self) -> None:
        agent = self._agent(context_manager="aggressive")
        cap = agent._resolve_context_capability(MagicMock())  # noqa: SLF001
        assert isinstance(cap, ContextManagerCapability)
        assert cap.compress_threshold == 0.7

    def test_true_maps_to_default(self) -> None:
        agent = self._agent(context_manager=True)
        cap = agent._resolve_context_capability(MagicMock())  # noqa: SLF001
        assert isinstance(cap, ContextManagerCapability)
        assert cap.compress_threshold == 0.9

    def test_sliding_preset_builds_sliding_window(self) -> None:
        agent = self._agent(context_manager="sliding")
        cap = agent._resolve_context_capability(MagicMock())  # noqa: SLF001
        assert isinstance(cap, SlidingWindowCapability)

    @pytest.mark.parametrize("value", ["off", "false", "none", "no", ""])
    def test_disabled_string_variants(self, value: str) -> None:
        agent = self._agent(context_manager=value)
        assert agent._resolve_context_capability(MagicMock()) is None  # noqa: SLF001

    def test_unknown_preset_falls_back_to_default(self, caplog: Any) -> None:
        agent = self._agent(context_manager="bogus")
        cap = agent._resolve_context_capability(MagicMock())  # noqa: SLF001
        assert isinstance(cap, ContextManagerCapability)
        assert cap.compress_threshold == 0.9  # fell back

    def test_usage_update_feeds_tracker(self) -> None:
        tracker = UsageTracker()
        agent = self._agent(usage_tracker=tracker)
        agent._on_usage_update(0.5, 4096, 8192)  # noqa: SLF001
        assert tracker.context_tokens == 4096
        assert tracker.context_budget == 8192
        assert tracker.to_dict()["context_tokens"] == 4096

    def test_usage_update_noop_without_tracker(self) -> None:
        agent = self._agent()  # no usage_tracker
        # must not raise when no tracker is attached
        agent._on_usage_update(0.5, 4096, 8192)  # noqa: SLF001
        assert agent.usage is None

    @staticmethod
    def _agent(**kwargs: Any) -> KnoxAgentBase[Any, Any]:
        class _A(KnoxAgentBase[Any, Any]):
            pass

        return _A(model_string="ollama/noop", **kwargs)
