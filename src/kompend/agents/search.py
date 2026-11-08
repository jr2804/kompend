"""Search agents — query expansion, CRAG grading, HyDE."""

from __future__ import annotations

from typing import Any, ClassVar, Final

from pydantic import BaseModel

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps


class QueryExpansionOutput(BaseModel):
    """Output for query expansion: lists of rewritten queries."""

    keyword_rewrites: list[str] = []
    semantic_rewrites: list[str] = []


class CragGradeOutput(BaseModel):
    """Output for CRAG relevance grading: a single 0-1 score."""

    relevance: float = 0.5


def _search_user_prompt(agent: Any, deps: PipelineAgentDeps) -> str:
    """The caller renders ``search/*.md.j2`` and passes it as ``deps.text``."""
    return deps.text


SPEC_QUERY_EXPANSION: Final[AgentSpec] = AgentSpec(
    name="query_expansion",
    result_type=QueryExpansionOutput,
    deps_type=PipelineAgentDeps,
    temperature=0.0,
    user_prompt=_search_user_prompt,
)


class QueryExpansionAgent(KompendAgentBase[QueryExpansionOutput, PipelineAgentDeps]):
    """PydanticAI agent for multi-query expansion (RAG-Fusion)."""

    spec: ClassVar[AgentSpec | None] = SPEC_QUERY_EXPANSION


SPEC_CRAG_GRADE: Final[AgentSpec] = AgentSpec(
    name="crag_grade",
    result_type=CragGradeOutput,
    deps_type=PipelineAgentDeps,
    temperature=0.0,
    user_prompt=_search_user_prompt,
)


class CragGradeAgent(KompendAgentBase[CragGradeOutput, PipelineAgentDeps]):
    """PydanticAI agent for CRAG relevance grading."""

    spec: ClassVar[AgentSpec | None] = SPEC_CRAG_GRADE


SPEC_HYDE: Final[AgentSpec] = AgentSpec(
    name="hyde",
    result_type=str,
    deps_type=PipelineAgentDeps,
    temperature=0.7,
    user_prompt=_search_user_prompt,
)


class HydeAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for HyDE hypothetical answer generation."""

    spec: ClassVar[AgentSpec | None] = SPEC_HYDE
