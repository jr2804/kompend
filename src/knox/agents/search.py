"""Search agents — query expansion, CRAG grading, HyDE."""

from __future__ import annotations
from typing import ClassVar

from pydantic import BaseModel

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps


class QueryExpansionOutput(BaseModel):
    """Output for query expansion: lists of rewritten queries."""

    keyword_rewrites: list[str] = []
    semantic_rewrites: list[str] = []


class CragGradeOutput(BaseModel):
    """Output for CRAG relevance grading: a single 0-1 score."""

    relevance: float = 0.5


class QueryExpansionAgent(KnoxAgentBase[QueryExpansionOutput, PipelineAgentDeps]):
    """PydanticAI agent for multi-query expansion (RAG-Fusion)."""

    template_name: ClassVar[str] = "expand_query.md.j2"
    system_template_name: ClassVar[str] = "expand_query.md.j2"
    template_prefix: ClassVar[str] = "search/"
    temperature: float | None = 0.0


class CragGradeAgent(KnoxAgentBase[CragGradeOutput, PipelineAgentDeps]):
    """PydanticAI agent for CRAG relevance grading."""

    template_name: ClassVar[str] = "crag_grade.md.j2"
    system_template_name: ClassVar[str] = "crag_grade.md.j2"
    template_prefix: ClassVar[str] = "search/"
    temperature: float | None = 0.0


class HydeAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for HyDE hypothetical answer generation."""

    template_name: ClassVar[str] = "hyde_generate.md.j2"
    system_template_name: ClassVar[str] = "hyde_generate.md.j2"
    template_prefix: ClassVar[str] = "search/"
    temperature: float | None = 0.7
