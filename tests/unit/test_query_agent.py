"""Tests for the QueryAgent (knox-pyda Phase 4).

The agent wraps a PydanticAI Agent around the existing query synthesis
prompt template. Unlike ExtractionAgent/GenerationAgent (structured
JSON output), the query agent returns free text (``output_type=str``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from knox.agents.deps import QueryDeps
from knox.agents.query import QueryAgent


class TestQueryAgentConstruction:
    def test_class_metadata(self) -> None:
        assert QueryAgent.system_template_name == "synthesize_system.md.j2"
        assert QueryAgent.result_type is str

    def test_constructs_with_litellm_model(self) -> None:
        agent = QueryAgent(litellm_model="ollama/gpt-oss:20b")
        assert agent.litellm_model == "ollama/gpt-oss:20b"


class TestQueryAgentRun:
    def _make_agent_and_mock(self) -> tuple[QueryAgent, MagicMock]:
        """Build an agent with a mocked pydantic_ai.Agent.run_sync."""
        agent = QueryAgent(litellm_model="ollama/gpt-oss:20b")
        mock_sync = MagicMock()
        mock_sync.return_value = MagicMock(output="VAD stands for Voice Activity Detection.")
        agent._agent = MagicMock()
        agent._agent.run_sync = mock_sync
        return agent, mock_sync

    def test_run_returns_string(self) -> None:
        agent, _ = self._make_agent_and_mock()
        deps = QueryDeps(
            question="What is VAD?",
            context="[1] Voice Activity Detection\n    ...",
            source_id="query-1",
        )
        result = agent.run(deps)
        assert isinstance(result, str)
        assert "Voice Activity Detection" in result

    def test_run_passes_question_as_user_prompt(self) -> None:
        agent, mock_sync = self._make_agent_and_mock()
        deps = QueryDeps(
            question="PAYLOAD-QUESTION-XYZ",
            context="[1] Some context",
            source_id="query-1",
        )
        agent.run(deps)
        mock_sync.assert_called_once()
        positional_args = mock_sync.call_args.args
        assert len(positional_args) == 1
        assert "PAYLOAD-QUESTION-XYZ" in positional_args[0]
        assert mock_sync.call_args.kwargs.get("deps") is deps

    def test_run_wraps_exception_as_llm_output_error(self) -> None:
        agent = QueryAgent(litellm_model="ollama/gpt-oss:20b")
        agent._agent = MagicMock()
        agent._agent.run_sync = MagicMock(side_effect=RuntimeError("boom"))
        deps = QueryDeps(
            question="q",
            context="c",
            source_id="query-1",
        )
        with pytest.raises(Exception, match="Agent failed for model") as exc_info:
            agent.run(deps)
        assert "ollama/gpt-oss:20b" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestQueryDeps:
    def test_defaults(self) -> None:
        deps = QueryDeps(question="q", context="c", source_id="s")
        assert deps.verbosity == "standard"
        assert deps.timeout == 120.0

    def test_custom_verbosity(self) -> None:
        deps = QueryDeps(question="q", context="c", source_id="s", verbosity="brief")
        assert deps.verbosity == "brief"


class TestQueryAgentRenderUserPrompt:
    def test_user_prompt_is_raw_question(self) -> None:
        agent = QueryAgent(litellm_model="ollama/gpt-oss:20b")
        deps = QueryDeps(question="What is noise reduction?", context="ctx", source_id="q1")
        prompt = agent.render_user_prompt(deps)
        assert prompt == "What is noise reduction?"
