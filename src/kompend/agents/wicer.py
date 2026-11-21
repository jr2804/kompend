r"""WicerAgents — PydanticAI wrappers for the WiCER compile-evaluate-refine loop."""

from __future__ import annotations

from typing import Any, ClassVar, Final

from pydantic_ai import UsageLimits

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.deps import WicerDiagnoseDeps, WicerEvaluateDeps, WicerProbeDeps
from kompend.compilation.schema import WicerDiagnoseOutput, WicerEvaluateOutput, WicerProbeOutput


def _wicer_probe_system_prompt(agent: Any, deps: Any) -> str:
    """WiCER system message plus any skills."""
    prompt = "You are a test-probe generator. Return only valid JSON."
    if agent.skills:
        prompt += "\n\n" + "\n\n---\n\n".join(f"## Skill: {n}\n\n{i}" for n, i in agent.skills.items())
    return prompt


SPEC_PROBE: Final[AgentSpec] = AgentSpec(
    name="wicer_probe",
    result_type=WicerProbeOutput,
    deps_type=WicerProbeDeps,
    template_name="wicer_probe.md.j2",
    system_template_name="wicer_probe.md.j2",
    system_prompt=_wicer_probe_system_prompt,
)


class WicerProbeAgent(KompendAgentBase[WicerProbeOutput, WicerProbeDeps]):
    """PydanticAI agent for WiCER probe generation (Stage 1)."""

    spec: ClassVar[AgentSpec | None] = SPEC_PROBE

    #: Requests one probe batch costs: a single-shot generation over the batch text.
    requests_per_batch: ClassVar[int] = 1

    def _usage_limits_for(self, deps: WicerProbeDeps) -> UsageLimits | None:
        """Budget the one batch this run generates probes for.

        One run covers exactly one batch, so the count is fixed — the
        value is the wall-clock deadline the fit prices: a hung batch
        is bounded by its own deadline instead of consuming the whole
        refinement pass.
        """
        return self._session_limits(units=1, per_unit=self.requests_per_batch)


def _wicer_evaluate_system_prompt(agent: Any, deps: Any) -> str:
    """WiCER system message plus any skills."""
    prompt = "You are an impartial judge of factual correctness. Return only valid JSON."
    if agent.skills:
        prompt += "\n\n" + "\n\n---\n\n".join(f"## Skill: {n}\n\n{i}" for n, i in agent.skills.items())
    return prompt


SPEC_EVALUATE: Final[AgentSpec] = AgentSpec(
    name="wicer_evaluate",
    result_type=WicerEvaluateOutput,
    deps_type=WicerEvaluateDeps,
    template_name="wicer_evaluate.md.j2",
    system_template_name="wicer_evaluate.md.j2",
    system_prompt=_wicer_evaluate_system_prompt,
)


class WicerEvaluateAgent(KompendAgentBase[WicerEvaluateOutput, WicerEvaluateDeps]):
    """PydanticAI agent for WiCER evaluation (Stage 2 — LLM-as-judge)."""

    spec: ClassVar[AgentSpec | None] = SPEC_EVALUATE

    #: Requests one wiki segment costs: a single-shot judgement over the segment.
    requests_per_segment: ClassVar[int] = 1

    def _usage_limits_for(self, deps: WicerEvaluateDeps) -> UsageLimits | None:
        """Budget the one segment this run judges.

        Each run covers exactly one wiki segment (the loop re-arms per
        segment), so a hung segment is bounded by its own
        fit-priced deadline instead of consuming the whole pass.
        """
        return self._session_limits(units=1, per_unit=self.requests_per_segment)


def _wicer_diagnose_system_prompt(agent: Any, deps: Any) -> str:
    """WiCER system message plus any skills."""
    prompt = "You are a diagnostic assistant. Return only valid JSON."
    if agent.skills:
        prompt += "\n\n" + "\n\n---\n\n".join(f"## Skill: {n}\n\n{i}" for n, i in agent.skills.items())
    return prompt


SPEC_DIAGNOSE: Final[AgentSpec] = AgentSpec(
    name="wicer_diagnose",
    result_type=WicerDiagnoseOutput,
    deps_type=WicerDiagnoseDeps,
    template_name="wicer_diagnose.md.j2",
    system_template_name="wicer_diagnose.md.j2",
    system_prompt=_wicer_diagnose_system_prompt,
)


class WicerDiagnoseAgent(KompendAgentBase[WicerDiagnoseOutput, WicerDiagnoseDeps]):
    """PydanticAI agent for WiCER diagnosis (Stage 3 — extract critical facts)."""

    spec: ClassVar[AgentSpec | None] = SPEC_DIAGNOSE

    #: Requests one diagnosis costs: a single-shot extraction over the failures.
    requests_per_diagnosis: ClassVar[int] = 1

    def _usage_limits_for(self, deps: WicerDiagnoseDeps) -> UsageLimits | None:
        """Budget the one diagnosis this run performs.

        One run carries the whole failure set in a single request, so
        the deadline is one fit-priced turn plus retry headroom — a hung
        diagnosis cannot consume the whole refinement pass.
        """
        return self._session_limits(units=1, per_unit=self.requests_per_diagnosis)
