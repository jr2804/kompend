"""VerificationAgent — PydanticAI agent for claim/citation verification."""

from __future__ import annotations

from typing import Any

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class VerificationAgent(KnoxAgentBase[dict[str, Any], PipelineAgentDeps]):
    """PydanticAI agent for verifying claims against source excerpts."""

    template_name: str = "verify_claim.md.j2"
    system_template_name: str = "verify_system.md.j2"
    template_prefix: str = "verification/"
    result_type: type = dict  # type: ignore[arg-type]
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
