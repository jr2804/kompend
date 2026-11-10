"""DecisionProbeAgent — the live ``kompend doctor`` probe of a decision model.

A decision model answers typed questions on the SystemOne endpoint
(``/v1/systemone``), a different endpoint family from the chat models the
other doctor checks ping, with its own failure modes: a provider base URL that
keeps the API version suffix (``/v1/v1/systemone``, 404), a missing api key
(401), the zstd frame the SDK sends by default ("invalid character '('"), and
a result type the decision grammar refuses - which degrades every judge tier
to its fallback without a single error reaching the caller.

The probe is one request: one short statement as the material, one typed
yes/no question about it as the result field's description. SystemOne turns
that field into the question, so the answer proves the endpoint speaks the
protocol, not merely that the host resolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from pydantic import BaseModel, Field

from kompend.agents.base import AgentSpec, KompendAgentBase, build_agent
from kompend.llm.models import build_decision_model


@dataclass
class DecisionProbeDeps:
    """Deps for the decision probe: the material being judged.

    Attributes:
        material: The statement the model is asked about. Kept to one short
            sentence so the probe costs a single cheap request.
        timeout: Per-call timeout in seconds.
        source_id: Log/provenance label.
    """

    material: str = "The decision endpoint answered this doctor probe."
    timeout: float = 10.0
    source_id: str = "doctor-decision"


class DecisionProbeResult(BaseModel):
    """One typed answer from a decision model — the doctor's liveness proof."""

    replied: bool = Field(description="True when the statement above says that the endpoint answered.")


SPEC: Final[AgentSpec] = AgentSpec(
    name="decision_probe",
    result_type=DecisionProbeResult,
    deps_type=DecisionProbeDeps,
    template_name="doctor_probe.md.j2",
    system_template_name="doctor_probe_system.md.j2",
    template_prefix="decision/",
    temperature=0.0,
    retries=0,
    model_builder=build_decision_model,
)


class DecisionProbeAgent(KompendAgentBase[DecisionProbeResult, DecisionProbeDeps]):
    """Decision model ping: one typed yes/no through the SystemOne endpoint."""

    spec: ClassVar[AgentSpec | None] = SPEC


def probe_decision_model(model_string: str, *, timeout: float) -> DecisionProbeResult:
    """Ask the configured decision model one typed question, once.

    Raises whatever the agent raises: the caller (the ``connectivity-decision``
    doctor check) is the boundary that turns a failure into a reported fact.
    """
    agent = build_agent(SPEC, model_string=model_string, timeout=timeout)
    return agent.run(DecisionProbeDeps(timeout=timeout))
