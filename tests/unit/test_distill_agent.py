"""Tests for AgenticDistillAgent tool wiring.

Regression guard: commit ``d443c583`` dropped the ``make_agentic_*`` factory
shims, which were the only place the distill tools were passed to the agent.
The system prompt still instructs the model to call ``get_group()`` /
``save_page()``, so the agent must register exactly those tools — otherwise
the model is told it has tools and has none.
"""

from __future__ import annotations

from kompend.agents.distill_agentic import make_agentic_distill_agent
from kompend.agents.types import DistillAgentDeps, DistillGroup, DistillNote

# NOTE: system-prompt-tools == registered-tools is guarded centrally for every
# agent in tests/unit/test_agent_templates.py; only the pinned tool list remains
# here.


class TestAgenticDistillAgentTools:
    def test_factory_registers_prompt_declared_tools(self) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        built = agent._build_agent()
        assert sorted(built._function_toolset.tools.keys()) == ["get_group", "save_page"]


class TestAgenticDistillAgentPrompt:
    def test_render_user_prompt_with_groups(self) -> None:
        """render_user_prompt succeeds with deps built by capture/distill.py.

        Regression guard: the user-prompt template once declared a required
        ``max_groups`` variable the deps never carried, so every agentic
        distill render raised at run start and kompend silently fell back to
        deterministic per-group merging.
        """
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        deps = DistillAgentDeps(
            groups=[DistillGroup(tag="dll", notes=[DistillNote(title="T", content="b", source="n.md")])],
            source_id="distill-all",
            timeout=180.0,
        )
        user_prompt = agent.render_user_prompt(deps)
        assert user_prompt.strip()
        system_prompt = agent.render_system_prompt(deps)
        assert "get_group" in system_prompt
        assert "save_page" in system_prompt
