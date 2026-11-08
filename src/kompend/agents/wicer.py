r"""WicerAgents — PydanticAI wrappers for the WiCER compile-evaluate-refine loop."""

from __future__ import annotations

from typing import Any, ClassVar, Final

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
