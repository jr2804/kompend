r"""WicerAgents — PydanticAI wrappers for the WiCER compile-evaluate-refine loop."""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.base import KompendAgentBase
from kompend.agents.deps import WicerDiagnoseDeps, WicerEvaluateDeps, WicerProbeDeps
from kompend.compilation.schema import WicerDiagnoseOutput, WicerEvaluateOutput, WicerProbeOutput


class WicerProbeAgent(KompendAgentBase[WicerProbeOutput, WicerProbeDeps]):
    """PydanticAI agent for WiCER probe generation (Stage 1)."""

    template_name: ClassVar[str] = "wicer_probe.md.j2"
    system_template_name: ClassVar[str] = "wicer_probe.md.j2"
    output_budget_key: ClassVar[str] = "wicer_probe_output_max"

    def render_system_prompt(self, deps: WicerProbeDeps) -> str:
        """WiCER probe uses a simple system message, not a template."""
        prompt = "You are a test-probe generator. Return only valid JSON."
        if self.skills:
            skill_blocks = [f"## Skill: {n}\n\n{i}" for n, i in self.skills.items()]
            prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
        return prompt


class WicerEvaluateAgent(KompendAgentBase[WicerEvaluateOutput, WicerEvaluateDeps]):
    """PydanticAI agent for WiCER evaluation (Stage 2 — LLM-as-judge)."""

    template_name: ClassVar[str] = "wicer_evaluate.md.j2"
    system_template_name: ClassVar[str] = "wicer_evaluate.md.j2"
    output_budget_key: ClassVar[str] = "wicer_eval_output_max"

    def render_system_prompt(self, deps: WicerEvaluateDeps) -> str:
        """WiCER evaluate uses a simple system message."""
        prompt = "You are an impartial judge of factual correctness. Return only valid JSON."
        if self.skills:
            skill_blocks = [f"## Skill: {n}\n\n{i}" for n, i in self.skills.items()]
            prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
        return prompt


class WicerDiagnoseAgent(KompendAgentBase[WicerDiagnoseOutput, WicerDiagnoseDeps]):
    """PydanticAI agent for WiCER diagnosis (Stage 3 — extract critical facts)."""

    template_name: ClassVar[str] = "wicer_diagnose.md.j2"
    system_template_name: ClassVar[str] = "wicer_diagnose.md.j2"
    output_budget_key: ClassVar[str] = "wicer_eval_output_max"

    def render_system_prompt(self, deps: WicerDiagnoseDeps) -> str:
        """WiCER diagnose uses a simple system message."""
        prompt = "You are a diagnostic assistant. Return only valid JSON."
        if self.skills:
            skill_blocks = [f"## Skill: {n}\n\n{i}" for n, i in self.skills.items()]
            prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
        return prompt
