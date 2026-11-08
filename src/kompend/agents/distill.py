"""DistillAgent — PydanticAI agent for synthesizing captured notes into wiki pages."""

from __future__ import annotations

from typing import ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.capture_models import DistillOutput
from kompend.agents.deps import DistillDeps

SPEC: Final[AgentSpec] = AgentSpec(
    name="distill",
    result_type=DistillOutput,
    deps_type=DistillDeps,
    template_name="distill.md.j2",
    template_prefix="capture/",
    temperature=0.2,
)


class DistillAgent(KompendAgentBase[DistillOutput, DistillDeps]):
    """PydanticAI agent for synthesizing grouped captured notes into a single wiki page."""

    spec: ClassVar[AgentSpec | None] = SPEC
    # The distill template is a self-contained user prompt (role + notes +
    # instructions). No separate static system prompt — rendering the same
    # template as system would fail its required ``notes``/``group_tag`` vars.
    temperature: float | None = 0.2
