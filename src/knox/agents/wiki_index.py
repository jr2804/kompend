"""WikiIndexAgent — PydanticAI agent for wiki index/overview generation."""

from __future__ import annotations

from typing import Any, ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class WikiIndexAgent(KnoxAgentBase[dict[str, Any], PipelineAgentDeps]):
    """PydanticAI agent for generating wiki overview/index text."""

    template_name: ClassVar[str] = "output/synthesize.md.j2"
    system_template_name: ClassVar[str] = "output/synthesize.md.j2"
