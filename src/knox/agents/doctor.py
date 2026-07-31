"""DoctorAgent — PydanticAI agent for the ``knox doctor`` connectivity check.

Replaces the direct ``LLMClient.try_complete()`` call in the doctor command
with a typed PydanticAI agent. The agent returns free text (a string) —
the model's response to a simple ping prompt.

Usage::

    from knox.agents.doctor import DoctorAgent

    agent = DoctorAgent(model_string="ollama/gemma4:e2b")
    response: str = agent.run(DoctorDeps(ping="Respond with exactly one word: ok"))
"""

from __future__ import annotations
from typing import ClassVar

from dataclasses import dataclass

from knox.agents.base import KnoxAgentBase


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


class DoctorAgent(KnoxAgentBase[str, DoctorDeps]):
    """PydanticAI agent for the doctor connectivity check.

    Returns free text (the model's response to a ping prompt).
    """

    template_name: ClassVar[str] = ""  # not used; render_user_prompt is overridden
    system_template_name: ClassVar[str] = ""  # not used; render_system_prompt is overridden

    def render_user_prompt(self, deps: DoctorDeps) -> str:  # noqa: PLR6301
        """The user prompt is the ping string — no template needed."""
        return deps.ping

    def render_system_prompt(self, deps: DoctorDeps) -> str:  # noqa: PLR6301
        """No system prompt for the doctor ping."""
        return ""
