"""Tests for the RLM capability (code-exploration tools on KNOX agents)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RequestUsage,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.settings import ModelSettings

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import GenerationDeps, PipelineAgentDeps, QueryDeps
from knox.agents.generation import GenerationAgent
from knox.agents.knox_model import KnoxModel
from knox.agents.query import QueryAgent
from knox.agents.rlm import (
    RLMCapability,
    build_rlm_instructions,
    cleanup_rlm_repls,
    resolve_rlm_preset,
)
from knox.core.config import RlmConfig
from knox.llm.usage import UsageTracker


class ScriptedModel(Model):
    """Model that calls ``execute_code`` once, then returns the given text."""

    model_name = "scripted"
    system = "scripted"

    def __init__(self, code: str, final: str, *, usage: RequestUsage | None = None) -> None:
        self._code = code
        self._final = final
        self._usage = usage or RequestUsage(input_tokens=1, output_tokens=1)
        self.requests: list[list[ModelMessage]] = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.requests.append(messages)
        if len(self.requests) == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="execute_code", args=json.dumps({"code": self._code}))],
                model_name=self.model_name,
                usage=self._usage,
            )
        return ModelResponse(
            parts=[TextPart(content=self._final)],
            model_name=self.model_name,
            usage=self._usage,
        )


class TextOnlyModel(Model):
    """Model that always returns the given text (for the llm_query sub-model)."""

    model_name = "textonly"
    system = "textonly"

    def __init__(self, text: str, *, usage: RequestUsage | None = None) -> None:
        self._text = text
        self._usage = usage or RequestUsage(input_tokens=1, output_tokens=1)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content=self._text)],
            model_name=self.model_name,
            usage=self._usage,
        )


class TestKnoxAgentRlmWiring:
    def test_rlm_disabled_by_default(self) -> None:
        """Rlm defaults to False: no capability, no usage limits, no tool."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(model_string="ollama/llama3.1")
        assert agent.rlm is False
        # No capability is resolved, so no toolset and no tool-call budget.
        assert agent._resolve_rlm_capability(agent._build_top_level_model()) is None
        agent._build_agent()
        assert agent._rlm_usage_limits is None

    def test_rlm_enabled_sets_usage_limits(self) -> None:
        """rlm='default' attaches the capability and a tool-call budget."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(model_string="ollama/llama3.1", rlm="default")
        agent._build_agent()
        assert agent._rlm_usage_limits is not None
        assert agent._rlm_usage_limits.tool_calls_limit == 50
        # request_limit must not bind below the tool-call budget: each tool
        # call costs >=1 model request, retries can double a turn.
        assert agent._rlm_usage_limits.request_limit is not None
        assert agent._rlm_usage_limits.request_limit >= 50
        assert agent._rlm_usage_limits.request_limit == 50 * 2 + 5

    def test_rlm_off_string_disables(self) -> None:
        """'off'-like string values disable RLM entirely."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        for setting in ("off", "false", "none", ""):
            agent = DefaultBuilder(model_string="ollama/llama3.1", rlm=setting)
            assert agent._resolve_rlm_capability(agent._build_top_level_model()) is None

    def test_rlm_config_group_overrides_preset(self) -> None:
        """A RlmConfig (compile.llm.rlm group) overrides the preset budget."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(
            model_string="ollama/llama3.1",
            rlm=RlmConfig(preset="aggressive", max_tool_calls=150, code_timeout=90.0, node_budget=32),
        )
        cap = agent._resolve_rlm_capability(agent._build_top_level_model())
        assert cap is not None
        assert cap.max_tool_calls == 150
        assert cap._code_timeout == 90.0
        assert cap._node_budget == 32
        agent._build_agent()
        assert agent._rlm_usage_limits is not None
        assert agent._rlm_usage_limits.tool_calls_limit == 150
        assert agent._rlm_usage_limits.request_limit == 150 * 2 + 5

    def test_rlm_config_disabled_returns_none(self) -> None:
        """A RlmConfig with enabled=False resolves to no capability."""

        class DefaultBuilder(KnoxAgentBase[Any, Any]):
            pass

        agent = DefaultBuilder(model_string="ollama/llama3.1", rlm=RlmConfig(enabled=False))
        assert agent._resolve_rlm_capability(agent._build_top_level_model()) is None


class TestRlmCapabilityToolset:
    def test_execute_code_runs_against_deps_context(self) -> None:
        """The REPL loads the deps context and executes the model's code."""
        model = ScriptedModel(code="print(len(context))", final="DONE")
        result = _run_rlm_agent(model, context="hello world")
        assert result.output == "DONE"
        # The execute_code tool return carries the REPL stdout.
        tool_returns = [p.content for m in result.all_messages() for p in getattr(m, "parts", []) if "ToolReturn" in type(p).__name__]
        assert len(tool_returns) == 1
        assert "11" in str(tool_returns[0])  # len("hello world")

    def test_execute_code_reports_errors_to_model(self) -> None:
        """A crashing script is surfaced as tool output, not a run failure."""
        model = ScriptedModel(code="raise ValueError('boom')", final="DONE")
        result = _run_rlm_agent(model)
        assert result.output == "DONE"
        tool_returns = [p.content for m in result.all_messages() for p in getattr(m, "parts", []) if "ToolReturn" in type(p).__name__]
        assert "boom" in str(tool_returns[0])

    def test_rlm_instructions_in_system_prompt(self) -> None:
        """RLM mode instructions are merged into the system prompt."""
        ScriptedModel(code="pass", final="DONE")
        seen_instructions: list[str] = []

        def handler(messages: list[ModelMessage], info: Any) -> ModelResponse:
            seen_instructions.append(info.instructions or "")
            usage = RequestUsage(input_tokens=1, output_tokens=1)
            if len(seen_instructions) == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="execute_code", args=json.dumps({"code": "pass"}))],
                    model_name="scripted",
                    usage=usage,
                )
            return ModelResponse(parts=[TextPart(content="DONE")], model_name="scripted", usage=usage)

        agent = Agent(
            FunctionModel(handler),
            output_type=str,
            capabilities=[RLMCapability(context_field="text")],
        )
        agent.run_sync(
            "analyze",
            deps=PipelineAgentDeps(text="x", source_id="d"),
            instructions="Custom instructions.",
        )
        assert "## RLM Mode" in seen_instructions[0]
        assert "Custom instructions." in seen_instructions[0]


class TestRlmLlMQuery:
    def test_llm_query_routes_through_sub_model_and_tracks_usage(self) -> None:
        """llm_query() calls the sub-model and records usage via KnoxModel."""
        usage = RequestUsage(input_tokens=7, output_tokens=3)
        main = ScriptedModel(code="print(llm_query('summarize'))", final="DONE")
        sub = TextOnlyModel("sub-answer", usage=usage)
        tracker = UsageTracker()
        knox_sub = KnoxModel(sub, usage_tracker=tracker)

        agent = Agent(
            main,
            output_type=str,
            capabilities=[RLMCapability(context_field="text", sub_model=knox_sub)],
        )
        result = agent.run_sync(
            "analyze",
            deps=PipelineAgentDeps(text="big document", source_id="d"),
        )
        assert result.output == "DONE"
        # The sub-model was called once (the llm_query inside the REPL).
        assert tracker.call_count == 1
        assert tracker.prompt_tokens == 7
        assert tracker.completion_tokens == 3

    def test_build_rlm_instructions_mentions_llm_query_only_with_sub_model(self) -> None:
        with_sub = build_rlm_instructions(sub_model=True)
        without = build_rlm_instructions(sub_model=False)
        assert "llm_query" in with_sub
        assert "llm_query" not in without


class QueuedModel(Model):
    """Model popping scripted actions: ``"CODE: <py>"`` → execute_code call,
    any other string → final text response.
    """

    model_name = "queued"
    system = "queued"

    def __init__(self, actions: list[str]) -> None:
        self._actions = actions
        self.requests: list[list[ModelMessage]] = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.requests.append(messages)
        action = self._actions.pop(0)
        usage = RequestUsage(input_tokens=1, output_tokens=1)
        if action.startswith("CODE:"):
            return ModelResponse(
                parts=[ToolCallPart(tool_name="execute_code", args=json.dumps({"code": action[5:]}))],
                model_name=self.model_name,
                usage=usage,
            )
        return ModelResponse(parts=[TextPart(content=action)], model_name=self.model_name, usage=usage)


class TestRlmRecursion:
    def test_rlm_query_recurses_to_child_agent(self) -> None:
        """rlm_query spawns a child agent over the given slice; its answer
        flows back to the parent's code.
        """
        model = QueuedModel(
            [
                "CODE: print(rlm_query('summarize', context[:5]))",
                "child-done",  # child final text
                "DONE",  # parent final text
            ]
        )
        agent = self._recursive_agent(model, depth=1)
        result = agent.run_sync("analyze", deps=PipelineAgentDeps(text="hello world", source_id="d"))
        assert result.output == "DONE"
        returns = _tool_returns(result)
        assert "child-done" in returns[0]
        # Three model requests: parent code, child answer, parent final.
        assert len(model.requests) == 3

    def test_rlm_query_two_level_chain(self) -> None:
        """depth=2 allows a child to recurse again (grandchild)."""
        model = QueuedModel(
            [
                "CODE: print(rlm_query('level1', 'AAAA'))",  # parent → child
                "CODE: print(rlm_query('level2', 'BBBB'))",  # child → grandchild
                "grand-done",  # grandchild final
                "child-done",  # child final
                "DONE",  # parent final
            ]
        )
        agent = self._recursive_agent(model, depth=2)
        result = agent.run_sync("analyze", deps=PipelineAgentDeps(text="top", source_id="d"))
        assert result.output == "DONE"
        assert len(model.requests) == 5

    def test_depth_zero_has_no_rlm_query(self) -> None:
        """depth=0 (or no main_model) leaves rlm_query undefined in the REPL."""
        model = QueuedModel(
            [
                "CODE: print(rlm_query('x'))",
                "DONE",
            ]
        )
        agent = self._recursive_agent(model, depth=0)
        result = agent.run_sync("analyze", deps=PipelineAgentDeps(text="t", source_id="d"))
        assert result.output == "DONE"
        assert "rlm_query' is not defined" in _tool_returns(result)[0]

    def test_node_budget_exhausted(self) -> None:
        """The shared budget caps total rlm_query calls across the tree."""
        model = QueuedModel(
            [
                "CODE: print(rlm_query('a', 'AAAA')); print(rlm_query('b', 'BBBB'))",
                "first-done",  # first child
                "DONE",
            ]
        )
        agent = self._recursive_agent(model, depth=1, node_budget=1)
        result = agent.run_sync("analyze", deps=PipelineAgentDeps(text="t", source_id="d"))
        assert result.output == "DONE"
        returns = _tool_returns(result)[0]
        assert "first-done" in returns
        assert "node budget exhausted" in returns

    def test_cycle_detection_rejects_identical_query(self) -> None:
        """The same (prompt, context) twice is a cycle; different contexts are not."""
        model = QueuedModel(
            [
                "CODE: print(rlm_query('same', 'CTX')); print(rlm_query('same', 'CTX'))",
                "first-done",
                "DONE",
            ]
        )
        agent = self._recursive_agent(model, depth=1)
        result = agent.run_sync("analyze", deps=PipelineAgentDeps(text="t", source_id="d"))
        returns = _tool_returns(result)[0]
        assert "first-done" in returns
        assert "cycle" in returns

    def test_recursion_usage_tracked_through_knox_model(self) -> None:
        """Child agent runs through the (KnoxModel-wrapped) main model, so
        every request in the tree lands in the usage tracker.
        """
        inner = QueuedModel(
            [
                "CODE: print(rlm_query('q', 'CTX'))",
                "child-done",
                "DONE",
            ]
        )
        tracker = UsageTracker()
        knox = KnoxModel(inner, usage_tracker=tracker)
        agent = Agent(
            knox,
            output_type=str,
            capabilities=[RLMCapability(context_field="text", main_model=knox, depth=1)],
        )
        result = agent.run_sync("analyze", deps=PipelineAgentDeps(text="t", source_id="d"))
        assert result.output == "DONE"
        assert tracker.call_count == 3  # parent code + child answer + parent final

    def _recursive_agent(self, model: Model, **cap_kwargs: Any) -> Agent[Any, str]:
        return Agent(
            model,
            output_type=str,
            capabilities=[RLMCapability(context_field="text", main_model=model, **cap_kwargs)],
        )


class TestPromptPlaceholder:
    def test_query_agent_rlm_placeholder_in_system_prompt(self) -> None:
        """QueryAgent inlines context in its system prompt — RLM replaces it
        with a placeholder so the model explores via code instead.
        """
        big = "BIG CONTENT " * 50
        deps = QueryDeps(question="q", context=big, source_id="s")
        # Disabled: context inlined.
        plain = QueryAgent(model_string="ollama/x")
        assert big in plain.render_system_prompt(deps)
        # Enabled: placeholder instead.
        rlm_agent = QueryAgent(model_string="ollama/x", rlm="default", rlm_context_field="context")
        rendered = rlm_agent.render_system_prompt(deps)
        assert big not in rendered
        assert "execute_code" in rendered
        # The deps still carry the full context (the REPL reads it).
        assert deps.context == big

    def test_base_render_user_prompt_rlm_placeholder(self) -> None:
        """The base render_user_prompt swaps the rlm_context_field var."""
        big = "BIG CONTENT " * 50
        deps = GenerationDeps(concepts_json="[]", content=big, source_id="s")
        plain = GenerationAgent(model_string="ollama/x")
        assert big in plain.render_user_prompt(deps)
        rlm_agent = GenerationAgent(
            model_string="ollama/x",
            rlm="default",
            rlm_context_field="content",
        )
        assert big not in rlm_agent.render_user_prompt(deps)
        assert "execute_code" in rlm_agent.render_user_prompt(deps)


class TestRlmPresets:
    def test_resolve_rlm_preset(self) -> None:
        assert resolve_rlm_preset(True) == (50, 60.0)
        assert resolve_rlm_preset("default") == (50, 60.0)
        assert resolve_rlm_preset("aggressive") == (100, 120.0)
        # Unknown presets warn and fall back to default.
        assert resolve_rlm_preset("bogus") == (50, 60.0)

    def test_resolve_rlm_preset_disabled(self) -> None:
        for setting in (False, None, "off", "false", "none", ""):
            with pytest.raises(ValueError, match="RLM disabled"):
                resolve_rlm_preset(setting)

    def test_cleanup_rlm_repls(self) -> None:
        """Cleanup drops cached REPL environments (no crash when empty)."""
        cleanup_rlm_repls()
        cleanup_rlm_repls()


def _run_rlm_agent(
    model: Model,
    *,
    context: str = "hello world",
    extra_instructions: str | None = None,
) -> Any:
    """Run a bare pydantic-ai Agent wired with the RLM capability."""
    agent = Agent(
        model,
        output_type=str,
        capabilities=[RLMCapability(context_field="text")],
    )
    deps = PipelineAgentDeps(text=context, source_id="doc1")
    return agent.run_sync("analyze", deps=deps, instructions=extra_instructions)


def _tool_returns(result: Any) -> list[str]:
    """Extract formatted execute_code tool returns from an agent result."""
    return [str(p.content) for m in result.all_messages() for p in getattr(m, "parts", []) if "ToolReturn" in type(p).__name__]
