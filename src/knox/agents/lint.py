"""LintRuleAgent — PydanticAI agent for LLM-powered lint rules."""

from __future__ import annotations

from dataclasses import field
from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage

from knox.agents.base import KnoxAgentBase
from knox.agents.types import PipelineAgentDeps
from knox.types import AgentResult


class LintRuleAgent(KnoxAgentBase[BaseModel | str, PipelineAgentDeps]):
    """PydanticAI agent for LLM-powered lint rules.

    Each rule (factual-consistency, clarity-score, coverage-gap,
    contradiction-detection) has its own template and expected output schema.
    The prompt is rendered by the rule itself and passed via ``deps.text``.

    To support multiple output schemas with a single agent instance,
    :meth:`run` accepts an optional ``result_type`` keyword. When the type
    changes, the underlying PydanticAI Agent is rebuilt (cached per type).
    """

    template_name: ClassVar[str] = ""  # not used; render_user_prompt is overridden
    system_template_name: ClassVar[str] = ""  # not used; system prompt is empty
    _agents_by_type: dict[type, Agent[Any, Any]] = field(default_factory=dict, init=False, repr=False)

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:  # noqa: PLR6301
        """Return the pre-rendered prompt from deps.text.

        Lint rules render their own templates in ``_call_agent()`` and
        pass the result as ``deps.text``, so the base class template
        rendering is bypassed.
        """
        return deps.text

    def run(
        self,
        deps: PipelineAgentDeps,
        *,
        result_type: type | None = None,
        message_history: list[ModelMessage] | None = None,
    ) -> AgentResult:
        """Single-turn agent invocation with optional per-call result type.

        When *result_type* differs from the cached agent's type, a new
        PydanticAI Agent is built and cached. This lets lint rules share
        a single LintRuleAgent while using different output schemas.
        """
        if result_type is not None and result_type != self.result_type:
            self.result_type = result_type
            self._agent = None  # force rebuild in super().run()
        return super().run(deps, message_history=message_history)
