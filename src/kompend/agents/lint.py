"""LintRuleAgent — PydanticAI agent for LLM-powered lint rules."""

from __future__ import annotations

from dataclasses import field
from typing import Any, ClassVar, Final

from pydantic import BaseModel
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.types import PipelineAgentDeps
from kompend.types import AgentResult


def _lint_user_prompt(agent: Any, deps: PipelineAgentDeps) -> str:
    """Return the pre-rendered prompt from deps.text."""
    return deps.text


SPEC: Final[AgentSpec] = AgentSpec(
    name="lint_rule",
    result_type=BaseModel | str,
    deps_type=PipelineAgentDeps,
    template_name="",
    system_template_name="",
    user_prompt=_lint_user_prompt,
)


class LintRuleAgent(KompendAgentBase[BaseModel | str, PipelineAgentDeps]):
    """PydanticAI agent for LLM-powered lint rules.

    Each rule (factual-consistency, clarity-score, coverage-gap,
    contradiction-detection) has its own template and expected output schema.
    The prompt is rendered by the rule itself and passed via ``deps.text``.

    To support multiple output schemas with a single agent instance,
    :meth:`run` accepts an optional ``result_type`` keyword. When the type
    changes, the underlying PydanticAI Agent is rebuilt (cached per type).
    """

    spec: ClassVar[AgentSpec | None] = SPEC
    _agents_by_type: dict[type, Agent[Any, Any]] = field(default_factory=dict, init=False, repr=False)

    def run(
        self,
        deps: PipelineAgentDeps,
        *,
        result_type: type | None = None,
        message_history: list[ModelMessage] | None = None,
        user_prompt: str | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentResult:
        """Single-turn agent invocation with optional per-call result type.

        When *result_type* differs from the cached agent's type, a new
        PydanticAI Agent is built and cached. This lets lint rules share
        a single LintRuleAgent while using different output schemas.

        The transcript kwargs exist for the base signature; lint rules are
        single-turn and never persist.
        """
        if result_type is not None and result_type != self.result_type:
            self.result_type = result_type
            self._agent = None  # force rebuild in super().run()
        return super().run(
            deps,
            message_history=message_history,
            user_prompt=user_prompt,
            conversation_id=conversation_id,
            run_id=run_id,
        )
