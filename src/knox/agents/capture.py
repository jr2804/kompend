"""CaptureAgent — PydanticAI agent for knowledge extraction from text."""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.capture_models import CaptureOutput
from knox.agents.deps import PipelineAgentDeps


class CaptureAgent(KnoxAgentBase[CaptureOutput, PipelineAgentDeps]):
    """PydanticAI agent for extracting structured knowledge items from text."""

    template_name: str = "capture_system.md.j2"
    system_template_name: str = "capture_system.md.j2"
    result_type: type = CaptureOutput
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
    temperature: float | None = 0.0
