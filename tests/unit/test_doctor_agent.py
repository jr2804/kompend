"""Tests for the DoctorAgent (PydanticAI migration #11)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kompend.agents.doctor import DoctorAgent, DoctorDeps
from kompend.core.utils.exceptions import LLMOutputError


class TestDoctorAgentConstruction:
    def test_class_metadata(self) -> None:
        """Template names and result type are class-level."""
        agent = DoctorAgent(model_string="ollama/test-model")
        assert agent.result_type is str
        assert agent.deps_type is DoctorDeps

    def test_constructs_with_model_string(self) -> None:
        agent = DoctorAgent(model_string="ollama/test-model")
        assert agent.model_string == "ollama/test-model"

    def test_constructs_with_max_tokens(self) -> None:
        agent = DoctorAgent(model_string="ollama/test-model", max_tokens=128)
        assert agent.max_tokens == 128


class TestDoctorAgentRun:
    def test_run_returns_string(self) -> None:
        """DoctorAgent returns free text from the model."""
        agent = DoctorAgent(model_string="ollama/test-model")
        # Mock the underlying PydanticAI agent
        mock_pydantic = MagicMock()
        mock_pydantic.run_sync.return_value.output = "ok"
        agent._agent = mock_pydantic

        deps = DoctorDeps(ping="Respond with exactly one word: ok")
        result = agent.run(deps)
        assert result == "ok"

    def test_run_passes_ping_as_user_prompt(self) -> None:
        """The ping string is passed as the user prompt."""
        agent = DoctorAgent(model_string="ollama/test-model")
        mock_pydantic = MagicMock()
        mock_pydantic.run_sync.return_value.output = "ok"
        agent._agent = mock_pydantic

        deps = DoctorDeps(ping="test ping")
        agent.run(deps)
        # Verify the user prompt was rendered correctly
        user_prompt = agent.render_user_prompt(deps)
        assert user_prompt == "test ping"

    def test_run_raises_on_failure(self) -> None:
        """When the agent fails, the exception is wrapped in LLMOutputError."""
        agent = DoctorAgent(model_string="ollama/test-model")
        mock_pydantic = MagicMock()
        mock_pydantic.run_sync.side_effect = RuntimeError("LLM unavailable")
        agent._agent = mock_pydantic

        deps = DoctorDeps(ping="ping")
        with pytest.raises(LLMOutputError, match="LLM unavailable"):
            agent.run(deps)

    def test_empty_response_returns_empty_string(self) -> None:
        """An empty model response returns an empty string (not None)."""
        agent = DoctorAgent(model_string="ollama/test-model")
        mock_pydantic = MagicMock()
        mock_pydantic.run_sync.return_value.output = ""
        agent._agent = mock_pydantic

        deps = DoctorDeps(ping="ping")
        result = agent.run(deps)
        assert result == ""
