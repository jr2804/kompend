"""CaptureAgent — PydanticAI agent for knowledge extraction from text."""

from __future__ import annotations

from typing import ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.capture_models import CaptureOutput
from kompend.agents.types import PipelineAgentDeps

SPEC: Final[AgentSpec] = AgentSpec(
    name="capture",
    result_type=CaptureOutput,
    deps_type=PipelineAgentDeps,
    template_name="capture_system.md.j2",
    system_template_name="capture_system.md.j2",
    template_prefix="capture/",
    temperature=0.0,
)


class CaptureAgent(KompendAgentBase[CaptureOutput, PipelineAgentDeps]):
    """PydanticAI agent for extracting structured knowledge items from text."""

    spec: ClassVar[AgentSpec | None] = SPEC
    temperature: float | None = 0.0
