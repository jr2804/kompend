"""Tests for the KnoxAgent base class and model-name bridge."""

from __future__ import annotations

from typing import Any

import pytest

from knox.agents.base import _litellm_to_pydanticai_model


class TestLitellmToPydanticaiModel:
    def test_simple_ollama(self) -> None:
        assert _litellm_to_pydanticai_model("ollama/gpt-oss:20b") == "ollama:gpt-oss:20b"

    def test_simple_openai(self) -> None:
        assert _litellm_to_pydanticai_model("openai/gpt-4o") == "openai:gpt-4o"

    def test_anthropic(self) -> None:
        assert _litellm_to_pydanticai_model("anthropic/claude-3-5-sonnet") == "anthropic:claude-3-5-sonnet"

    def test_no_slash_unchanged(self) -> None:
        assert _litellm_to_pydanticai_model("plain-model-name") == "plain-model-name"

    def test_only_first_slash_replaced(self) -> None:
        """The slash inside the model name (e.g. HuggingFace) stays as-is."""
        # Edge case: provider has no slash but model does — only first one replaced
        result = _litellm_to_pydanticai_model("custom-org/model-name")
        assert result == "custom-org:model-name"

    def test_empty_string(self) -> None:
        assert _litellm_to_pydanticai_model("") == ""

    def test_trailing_slash(self) -> None:
        # "openai/" -> "openai:" (provider with empty model). Pathological but legal.
        assert _litellm_to_pydanticai_model("openai/") == "openai:"


class TestKnoxAgentBaseContract:
    def test_knox_agent_base_is_abstract(self) -> None:
        """KnoxAgentBase cannot be instantiated directly — it requires an agent."""
        from knox.agents.base import KnoxAgentBase

        with pytest.raises(TypeError):
            KnoxAgentBase()  # type: ignore[abstract]

    def test_knox_agent_base_requires_build_agent(self) -> None:
        """Subclasses must implement _build_agent to be instantiable."""
        from knox.agents.base import KnoxAgentBase

        class Incomplete(KnoxAgentBase[Any, Any]):
            pass

        with pytest.raises(TypeError):
            Incomplete("ollama/foo")  # type: ignore[abstract]
