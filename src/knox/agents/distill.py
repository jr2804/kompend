"""DistillAgent — PydanticAI agent for synthesizing captured notes into wiki pages."""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.capture_models import DistillOutput
from knox.agents.deps import PipelineAgentDeps


class DistillAgent(KnoxAgentBase[DistillOutput, PipelineAgentDeps]):
    """PydanticAI agent for synthesizing grouped captured notes into a single wiki page."""

    template_name: str = "distill.md.j2"
    system_template_name: str = "distill.md.j2"
    result_type: type = DistillOutput
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
    temperature: float | None = 0.2
