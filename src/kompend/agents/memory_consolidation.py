"""MemoryConsolidationAgent — consolidates the agentic extraction digest.

The AGENTIC extraction strategy carries an :class:`~kompend.compilation.extraction_memory.ExtractionMemory`
across sections and injects its digest into every request as prior-section
context. On a document with many sections the digest grows without bound and
eventually no longer fits beside the section text in the model's context
window. Rather than cropping it (forbidden — see ``.agents/CODE.md`` "LLM
calls"), the model itself consolidates the digest down to the token budget
the window leaves for it: this agent is that consolidation call.
"""

from __future__ import annotations

from typing import ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.deps import MemoryConsolidationDeps

SPEC: Final[AgentSpec] = AgentSpec(
    name="memory_consolidation",
    result_type=str,
    deps_type=MemoryConsolidationDeps,
    template_name="memory_consolidation.md.j2",
)


class MemoryConsolidationAgent(KompendAgentBase[str, MemoryConsolidationDeps]):
    """PydanticAI agent consolidating the running extraction digest.

    Takes the previous consolidated context plus the material gained since
    the last consolidation and returns the merged running context, dense
    enough to serve as prior-section context for the sections that follow.
    """

    spec: ClassVar[AgentSpec | None] = SPEC
