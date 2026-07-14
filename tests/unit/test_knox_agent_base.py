"""Tests for the KnoxAgent base class and model-name bridge."""

from __future__ import annotations

from typing import Any

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class TestKnoxAgentBaseContract:
    def test_knox_agent_base_is_abstract(self) -> None:
        """KnoxAgentBase requires a concrete subclass to be usable, but the base
        class itself can be instantiated for testing because `_build_agent` now
        provides a default implementation.
        """
        agent: KnoxAgentBase[Any, Any] = KnoxAgentBase(litellm_model="test/noop")
        assert agent.litellm_model == "test/noop"

    def test_knox_agent_base_requires_build_agent(self) -> None:
        """Subclasses may rely on the default _build_agent implementation."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(litellm_model="test/noop")
        # Base class provides a default _build_agent; instantiation succeeds
        assert agent.litellm_model == "test/noop"

    def test_knox_agent_base_stores_reasoning_effort(self) -> None:
        """reasoning_effort is stored and forwarded to litellm_model."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(
            litellm_model="test/noop",
            reasoning_effort="low",
        )
        assert agent._reasoning_effort == "low"  # noqa: SLF001

    def test_knox_agent_base_reasoning_effort_none_by_default(self) -> None:
        """reasoning_effort defaults to None (use provider default)."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(litellm_model="test/noop")
        assert agent._reasoning_effort is None  # noqa: SLF001


class TestSkillInjection:
    """Skills passed to KnoxAgentBase are appended to the system prompt."""

    def test_skills_appended_to_system_prompt(self) -> None:
        """Skills are rendered as '## Skill: <name>' sections."""

        class _TestAgent(KnoxAgentBase[Any, Any]):
            template_name: str = "extraction_prompt.md.j2"
            system_template_name: str = "extraction_system.md.j2"
            result_type: type[Any] = Any

            def _build_agent(self) -> None:
                return None  # not needed for prompt rendering test

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

            def _build_agent(self) -> None:
                return None

        agent = _TestAgent("ollama/test-model")
        deps = PipelineAgentDeps(text="sample", source_id="test")
        prompt = agent.render_system_prompt(deps)

        assert "## Skill:" not in prompt
