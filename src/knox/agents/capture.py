"""CaptureAgent — PydanticAI agent for knowledge extraction from text."""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.capture_models import CaptureOutput
from knox.agents.types import PipelineAgentDeps


class CaptureAgent(KnoxAgentBase[CaptureOutput, PipelineAgentDeps]):
    """PydanticAI agent for extracting structured knowledge items from text."""

    template_name: ClassVar[str] = "capture_system.md.j2"
    system_template_name: ClassVar[str] = "capture_system.md.j2"
    template_prefix: ClassVar[str] = "capture/"
    temperature: float | None = 0.0
