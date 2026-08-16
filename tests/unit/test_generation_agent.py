"""Tests for the GenerationAgent (knox-pyda Phase 3).

The agent wraps a PydanticAI Agent around the existing generation
prompt templates. We mock the underlying PydanticAI ``run_sync`` so
the test never actually contacts an LLM — we only verify that the
KNOX-side wiring (template rendering, result handling, error mapping)
is correct.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from knox.agents.deps import GenerationDeps
from knox.agents.generation import GenerationAgent
from knox.compilation.flows.generation import _allowed_kind, _categories_block
from knox.compilation.schema import GenerationOutput, GenerationPage


class TestGenerationAgentConstruction:
    def test_class_metadata(self) -> None:
        """Template names and result type are class-level so the registry can introspect."""
        agent = GenerationAgent(model_string="test")
        assert GenerationAgent.template_name == "generation_prompt.md.j2"
        assert GenerationAgent.system_template_name == "generation_system.md.j2"
        assert agent.result_type is GenerationOutput
        assert agent.deps_type is GenerationDeps

    def test_constructs_with_model_string(self) -> None:
        agent = GenerationAgent(model_string="ollama/gpt-oss:20b")
        assert agent.model_string == "ollama/gpt-oss:20b"


class TestGenerationAgentRun:
    def test_run_returns_typed_generation_output(self) -> None:
        agent, _ = self._make_agent_and_mock()
        deps = GenerationDeps(
            concepts_json='[{"name": "Foo"}]',
            content="### doc1\nSome text.",
            source_id="batch-1",
        )
        result = agent.run(deps)
        assert isinstance(result, GenerationOutput)
        assert len(result.pages) == 1
        assert result.pages[0].title == "Foo"
        assert result.pages[0].content == "A page about Foo."

    def test_run_passes_user_prompt_to_agent(self) -> None:
        agent, mock_sync = self._make_agent_and_mock()
        deps = GenerationDeps(
            concepts_json='[{"name": "PAYLOAD-CONCEPT"}]',
            content="### doc1\nPAYLOAD-CONTENT",
            source_id="batch-1",
        )
        agent.run(deps)
        mock_sync.assert_called_once()
        positional_args = mock_sync.call_args.args
        assert len(positional_args) == 1
        assert "PAYLOAD-CONCEPT" in positional_args[0]
        assert "PAYLOAD-CONTENT" in positional_args[0]
        assert mock_sync.call_args.kwargs.get("deps") is deps

    def test_run_wraps_pydantic_ai_exception_as_llm_output_error(self) -> None:
        agent = GenerationAgent(model_string="ollama/gpt-oss:20b")
        agent._agent = MagicMock()
        agent._agent.run_sync = MagicMock(side_effect=RuntimeError("boom"))
        deps = GenerationDeps(
            concepts_json="[]",
            content="x",
            source_id="batch-1",
        )
        with pytest.raises(Exception, match="Agent failed for model") as exc_info:
            agent.run(deps)
        assert "ollama/gpt-oss:20b" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def _make_agent_and_mock(self) -> tuple[GenerationAgent, MagicMock]:
        """Build an agent with a mocked pydantic_ai.Agent.run_sync."""
        agent = GenerationAgent(model_string="ollama/gpt-oss:20b")
        mock_sync = MagicMock()
        mock_output = GenerationOutput(
            pages=[
                GenerationPage(
                    title="Foo",
                    slug="foo",
                    content="A page about Foo.",
                    provenance=["doc1"],
                    related=["Bar"],
                ),
            ],
        )
        mock_sync.return_value = MagicMock(output=mock_output)
        agent._agent = MagicMock()
        agent._agent.run_sync = mock_sync
        return agent, mock_sync


class TestGenerationDeps:
    def test_default_pinned_facts_empty(self) -> None:
        deps = GenerationDeps(
            concepts_json="[]",
            content="x",
            source_id="y",
        )
        assert deps.pinned_facts == []

    def test_default_timeout(self) -> None:
        deps = GenerationDeps(
            concepts_json="[]",
            content="x",
            source_id="y",
        )
        assert deps.timeout == 120.0

    def test_custom_pinned_facts(self) -> None:
        deps = GenerationDeps(
            concepts_json="[]",
            content="x",
            source_id="y",
            pinned_facts=["fact1", "fact2"],
        )
        assert deps.pinned_facts == ["fact1", "fact2"]


class TestPageKinds:
    """Category taxonomy: mandatory kinds, configured extras, coercion."""

    def test_generation_page_defaults_to_concept(self) -> None:
        page = GenerationPage(title="T", content="c")
        assert page.kind == "concept"

    def test_generation_page_accepts_any_kind(self) -> None:
        # The schema is permissive; strict validation happens at node creation.
        assert GenerationPage(title="T", content="c", kind="entity").kind == "entity"

    def test_allowed_kind_accepts_mandatory_and_extras(self) -> None:
        assert _allowed_kind("concept", {}) == "concept"
        assert _allowed_kind("entity", {}) == "entity"
        assert _allowed_kind("summary", {}) == "summary"
        assert _allowed_kind("comparison", {}) == "comparison"
        assert _allowed_kind("procedure", {"procedure": "how-to"}) == "procedure"

    def test_allowed_kind_falls_back_to_concept(self) -> None:
        assert _allowed_kind("garbage", {}) == "concept"
        assert _allowed_kind("procedure", {}) == "concept"  # extra not configured

    def test_categories_block_renders_instructions(self) -> None:
        block = _categories_block({"procedure": "how-to pages.", "case-study": "worked example."})
        assert '- "procedure" — how-to pages' in block
        assert '- "case-study" — worked example' in block

    def test_categories_block_empty_for_no_extras(self) -> None:
        assert _categories_block({}) == ""
