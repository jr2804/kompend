"""DistillAgent — PydanticAI agent for synthesizing captured notes into wiki pages."""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.capture_models import DistillOutput
from knox.agents.deps import DistillDeps


class DistillAgent(KnoxAgentBase[DistillOutput, DistillDeps]):
    """PydanticAI agent for synthesizing grouped captured notes into a single wiki page."""

    template_name: ClassVar[str] = "distill.md.j2"
    # The distill template is a self-contained user prompt (role + notes +
    # instructions). No separate static system prompt — rendering the same
    # template as system would fail its required ``notes``/``group_tag`` vars.
    system_template_name: ClassVar[str] = ""
    template_prefix: ClassVar[str] = "capture/"
    temperature: float | None = 0.2
