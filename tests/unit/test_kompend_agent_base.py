"""Tests for the KompendAgent base class and model-name bridge."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai_summarization import ContextManagerCapability, SlidingWindowCapability

from kompend.agents.base import KompendAgentBase, resolve_retry_budget
from kompend.agents.deps import PipelineAgentDeps
from kompend.core.config._configs import BudgetConfig, LLMConfig
from kompend.llm.usage import UsageTracker


class TestKompendAgentBaseContract:
    def test_kompend_agent_base_is_abstract(self) -> None:
        """KompendAgentBase requires a concrete subclass to be usable, but the base
        class itself can be instantiated for testing because `_build_agent` now
        provides a default implementation.
        """
        agent: KompendAgentBase[Any, Any] = KompendAgentBase(model_string="test/noop")
        assert agent.model_string == "test/noop"

    def test_kompend_agent_base_requires_build_agent(self) -> None:
        """Subclasses may rely on the default _build_agent implementation."""

        class DefaultBuilder(KompendAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(model_string="test/noop")
        # Base class provides a default _build_agent; instantiation succeeds
        assert agent.model_string == "test/noop"

    def test_kompend_agent_base_stores_reasoning_effort(self) -> None:
        """reasoning_effort is stored and forwarded to model_string."""

        class DefaultBuilder(KompendAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(
            model_string="test/noop",
            reasoning_effort="low",
        )
        assert agent.reasoning_effort == "low"

    def test_kompend_agent_base_reasoning_effort_none_by_default(self) -> None:
        """reasoning_effort defaults to None (use provider default)."""

        class DefaultBuilder(KompendAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(model_string="test/noop")
        assert agent.reasoning_effort is None


class TestSkillInjection:
    """Skills passed to KompendAgentBase are appended to the system prompt."""

    def test_skills_appended_to_system_prompt(self) -> None:
        """Skills are rendered as '## Skill: <name>' sections."""

        class _TestAgent(KompendAgentBase[Any, Any]):
            template_name: ClassVar[str] = "extraction_prompt.md.j2"
            system_template_name: ClassVar[str] = "extraction_system.md.j2"
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

        class _TestAgent(KompendAgentBase[Any, Any]):
            template_name: ClassVar[str] = "extraction_prompt.md.j2"
            system_template_name: ClassVar[str] = "extraction_system.md.j2"
            result_type: type[Any] = Any

            def _build_agent(self) -> Agent[Any, Any]:
                return MagicMock()

        agent = _TestAgent("ollama/test-model")
        deps = PipelineAgentDeps(text="sample", source_id="test")
        prompt = agent.render_system_prompt(deps)

        assert "## Skill:" not in prompt


class TestContextManagerCapability:
    """``context_manager`` preset resolution on KompendAgentBase."""

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
    def _agent(**kwargs: Any) -> KompendAgentBase[Any, Any]:
        class _A(KompendAgentBase[Any, Any]):
            pass

        return _A(model_string="ollama/noop", **kwargs)


class TestConfigDrivenDefaults:
    """Every limit a construction site does not set comes from the config."""

    def test_limits_come_from_the_config(self) -> None:
        llm = LLMConfig(
            timeout=45.0,
            tool_retries=4,
            output_retries=5,
            output_retries_enabled=True,
            reasoning_effort="high",
            context_manager="aggressive",
            context_window=1234,
            fallback_chain=["other/model"],
            delta_retry_enabled=False,
        )
        agent = self._agent(llm=llm, budget=BudgetConfig(agent_output_max=999))
        assert agent.timeout == 45.0
        assert agent.retries == {"tools": 4, "output": 5}
        assert agent.max_tokens == 999
        assert agent.reasoning_effort == "high"
        assert agent.context_manager == "aggressive"
        assert agent.context_window == 1234
        assert agent.fallback_models == ["other/model"]
        assert agent.delta_retry_enabled is False

    def test_explicit_values_win_over_the_config(self) -> None:
        llm = LLMConfig(timeout=45.0, output_retries=5)
        agent = self._agent(
            llm=llm,
            budget=BudgetConfig(agent_output_max=999),
            timeout=7.0,
            retries=1,
            max_tokens=11,
        )
        assert (agent.timeout, agent.retries, agent.max_tokens) == (7.0, 1, 11)

    def test_output_retries_disabled_switches_retries_off(self) -> None:
        llm = LLMConfig(output_retries=5, output_retries_enabled=False)
        assert self._agent(llm=llm).retries == 0

    def test_dedicated_output_budget_key_wins(self) -> None:
        class _Ref(KompendAgentBase[Any, Any]):
            output_budget_key = "reference_output_max"

        budget = BudgetConfig(agent_output_max=999, reference_output_max=4242)
        assert _Ref(model_string="ollama/noop", budget=budget).max_tokens == 4242

    def test_defaults_without_config(self) -> None:
        agent = self._agent()
        assert agent.timeout == 120.0
        assert agent.retries == 2
        assert agent.max_tokens is None
        assert agent.context_manager is False
        assert agent.delta_retry_enabled is True

    @staticmethod
    def _agent(**kwargs: Any) -> KompendAgentBase[Any, Any]:
        class _A(KompendAgentBase[Any, Any]):
            pass

        return _A(model_string="ollama/noop", **kwargs)


class TestResolveRetryBudget:
    def test_splits_tool_and_output_retries(self) -> None:
        llm = LLMConfig(
            tool_retries=3,
            output_retries=3,
            agentic_output_retries=6,
            output_retries_enabled=True,
        )
        assert resolve_retry_budget(llm, retries=None, output_retries=None) == {"tools": 3, "output": 6}

    def test_disabled_output_retries_zero_both(self) -> None:
        llm = LLMConfig(
            tool_retries=3,
            output_retries=3,
            agentic_output_retries=6,
            output_retries_enabled=False,
        )
        assert resolve_retry_budget(llm, retries=None, output_retries=None) == {"tools": 0, "output": 0}

    def test_explicit_retries_win(self) -> None:
        llm = LLMConfig(tool_retries=3, output_retries=3, agentic_output_retries=6)
        assert resolve_retry_budget(llm, retries=9, output_retries=None) == {"tools": 9, "output": 6}

    def test_tool_budget_comes_from_its_own_key(self) -> None:
        """A tool that keeps raising ModelRetry has its own budget.

        The output-validation budget must not decide how often a tool may be
        retried - the two failure modes have different remedies.
        """
        llm = LLMConfig(tool_retries=7, output_retries=1, agentic_output_retries=1)
        assert resolve_retry_budget(llm, retries=None, output_retries=None) == {"tools": 7, "output": 1}


class TestToolRetryBudget:
    """The tool half of the retry budget is reachable by the agent."""

    def test_agent_exposes_the_tool_budget(self) -> None:
        class _A(KompendAgentBase[Any, Any]):
            pass

        agent = _A(model_string="ollama/noop", llm=LLMConfig(tool_retries=6, output_retries=2))
        assert agent.tool_retry_budget() == 6

    def test_plain_retry_count_is_the_tool_budget(self) -> None:
        class _A(KompendAgentBase[Any, Any]):
            pass

        agent = _A(model_string="ollama/noop", retries=4)
        assert agent.tool_retry_budget() == 4
