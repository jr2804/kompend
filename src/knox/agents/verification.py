"""VerificationAgent — PydanticAI agent for claim/citation verification."""

from __future__ import annotations

from typing import ClassVar, Any

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class VerificationAgent(KnoxAgentBase[dict[str, Any], PipelineAgentDeps]):
    """PydanticAI agent for verifying claims against source excerpts."""

    template_name: ClassVar[str] = "verify_claim.md.j2"
    system_template_name: ClassVar[str] = "verify_system.md.j2"
    template_prefix: ClassVar[str] = "verification/"
