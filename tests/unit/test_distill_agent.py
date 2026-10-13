"""Tests for AgenticDistillAgent tool wiring.

Regression guard: commit ``d443c583`` dropped the ``make_agentic_*`` factory
shims, which were the only place the distill tools were passed to the agent.
The system prompt still instructs the model to call ``get_group()`` /
``save_page()``, so the agent must register exactly those tools — otherwise
the model is told it has tools and has none.
"""

from __future__ import annotations

from kompend.agents.distill_agentic import make_agentic_distill_agent

# NOTE: system-prompt-tools == registered-tools is guarded centrally for every
# agent in tests/unit/test_agent_templates.py; only the pinned tool list remains
# here.


class TestAgenticDistillAgentTools:
    def test_factory_registers_prompt_declared_tools(self) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        built = agent._build_agent()
        assert sorted(built._function_toolset.tools.keys()) == ["get_group", "save_page"]
