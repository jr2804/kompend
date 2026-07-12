"""Tests for the ExtractionAgent (knox-pyda Phase 1).

The agent wraps a PydanticAI Agent around the existing extraction
prompt templates. We mock the underlying PydanticAI ``run_sync`` so the
test never actually contacts an LLM — we only verify that the KNOX
side of the wiring (template rendering, result handling, error
mapping) is correct.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from knox.agents.deps import PipelineAgentDeps
from knox.agents.extraction import ExtractionAgent
from knox.compilation.schema import Concept, Entity, ExtractionOutput, Finding


class TestExtractionAgentConstruction:
    def test_class_metadata(self) -> None:
        """Template names and result type are class-level so the registry can introspect."""
        assert ExtractionAgent.template_name == "extraction_prompt.md.j2"
        assert ExtractionAgent.system_template_name == "extraction_system.md.j2"
        assert ExtractionAgent.result_type is ExtractionOutput

    def test_constructs_with_litellm_model(self) -> None:
        # Use a real model string; the underlying pydantic_ai.Agent only
        # tries to resolve the provider at run-time, not at construction.
        agent = ExtractionAgent(litellm_model="ollama/gpt-oss:20b")
        assert agent.litellm_model == "ollama/gpt-oss:20b"


class TestExtractionAgentRun:
    def _make_agent_and_mock(self) -> tuple[ExtractionAgent, MagicMock]:
        """Build an agent with a mocked pydantic_ai.Agent.run_sync.

        The underlying PydanticAI Agent is replaced with a MagicMock to
        avoid provider-resolution at construction time. This means the
        ``_build_agent()`` path is never exercised in this test — we
        only verify the KNOX-side wiring (prompt rendering, result
        handling, error mapping).
        """
        agent = ExtractionAgent(litellm_model="ollama/gpt-oss:20b")
        mock_sync = MagicMock()
        mock_output = ExtractionOutput(
            summary="A summary.",
            entities=[Entity(name="Foo", type="Author")],
            concepts=[Concept(name="Bar", definition="A concept.")],
            findings=[Finding(claim="X", evidence="Y")],
            tags=["a", "b"],
        )
        mock_sync.return_value = MagicMock(output=mock_output)
        # Replace the lazily-created PydanticAI Agent with a mock.
        agent._agent = MagicMock()
        agent._agent.run_sync = mock_sync
        return agent, mock_sync

    def test_run_returns_typed_extraction_output(self) -> None:
        agent, _ = self._make_agent_and_mock()
        deps = PipelineAgentDeps(text="Some text.", source_id="doc1")
        result = agent.run(deps)
        assert isinstance(result, ExtractionOutput)
        assert result.summary == "A summary."
        assert result.entities[0].name == "Foo"
        assert result.concepts[0].name == "Bar"
        assert result.tags == ["a", "b"]

    def test_run_passes_user_prompt_to_agent(self) -> None:
        agent, mock_sync = self._make_agent_and_mock()
        deps = PipelineAgentDeps(text="PAYLOAD-TEXT-XYZ", source_id="doc1")
        agent.run(deps)
        # The mock was called once with a user prompt that includes the text.
        mock_sync.assert_called_once()
        positional_args = mock_sync.call_args.args
        assert len(positional_args) == 1
        assert "PAYLOAD-TEXT-XYZ" in positional_args[0]
        # Deps are passed as keyword argument.
        assert mock_sync.call_args.kwargs.get("deps") is deps

    def test_run_wraps_pydantic_ai_exception_as_llm_output_error(self) -> None:
        agent = ExtractionAgent(litellm_model="ollama/gpt-oss:20b")
        agent._agent = MagicMock()
        agent._agent.run_sync = MagicMock(side_effect=RuntimeError("boom"))
        deps = PipelineAgentDeps(text="x", source_id="doc1")
        with pytest.raises(Exception) as exc_info:
            agent.run(deps)
        # The exception is an LLMOutputError (knox.compilation.schema) with the
        # original RuntimeError as __cause__.
        assert "Agent failed for model" in str(exc_info.value)
        assert "ollama/gpt-oss:20b" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestPipelineAgentDeps:
    def test_default_timeout(self) -> None:
        deps = PipelineAgentDeps(text="x", source_id="y")
        assert deps.timeout == 120.0

    def test_custom_timeout(self) -> None:
        deps = PipelineAgentDeps(text="x", source_id="y", timeout=30.0)
        assert deps.timeout == 30.0

    def test_template_vars_excludes_underscore_keys(self) -> None:
        """Internal Python attributes should not leak into Jinja2 context."""
        deps = PipelineAgentDeps(text="hello", source_id="world")
        vars_ = ExtractionAgent._deps_to_template_vars_static(deps)
        assert "text" in vars_
        assert "source_id" in vars_
        assert "timeout" in vars_
        for k in vars_:
            assert not k.startswith("_")


# Static accessor for the staticmethod (Python 3.10+ allows this directly
# but we expose a wrapper for test ergonomics).
ExtractionAgent._deps_to_template_vars_static = staticmethod(  # type: ignore[attr-defined]
    ExtractionAgent._deps_to_template_vars
)
