"""DoctorAgent — PydanticAI agent for the ``kompend doctor`` connectivity check.

Replaces the direct ``LLMClient.try_complete()`` call in the doctor command
with a typed PydanticAI agent. The agent returns free text (a string) —
the model's response to a simple ping prompt.

Usage::

    from kompend.agents.doctor import DoctorAgent

    agent = DoctorAgent(model_string="ollama/gemma4:e2b")
    response: str = agent.run(DoctorDeps(ping="Respond with exactly one word: ok"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase


@dataclass
class DoctorDeps:
    """Deps for the doctor ping agent.

    Attributes:
        ping: The ping prompt to send to the model.
        timeout: Per-call timeout in seconds.
        source_id: Log/provenance label.
    """

    ping: str
    timeout: float = 30.0
    source_id: str = "doctor"


def _doctor_user_prompt(agent: Any, deps: DoctorDeps) -> str:
    """The user prompt is the ping string — no template needed."""
    return deps.ping


def _doctor_system_prompt(agent: Any, deps: DoctorDeps) -> str:
    """No system prompt for the doctor ping."""
    return ""


SPEC: Final[AgentSpec] = AgentSpec(
    name="doctor",
    result_type=str,
    deps_type=DoctorDeps,
    template_name="",
    system_template_name="",
    user_prompt=_doctor_user_prompt,
    system_prompt=_doctor_system_prompt,
)


class DoctorAgent(KompendAgentBase[str, DoctorDeps]):
    """PydanticAI agent for the doctor connectivity check.

    Returns free text (the model's response to a ping prompt).
    """

    spec: ClassVar[AgentSpec | None] = SPEC
