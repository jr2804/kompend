"""Search agents — query expansion, CRAG grading, HyDE."""

from __future__ import annotations

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


class _PreRenderedPromptAgent:
    """Mixin for search agents whose prompt is rendered by the caller.

    ``EnhancedRetriever`` renders the ``search/*.md.j2`` template with the
    query/title/snippet variables and passes the finished prompt as
    ``deps.text``. Declaring that template as ``template_name`` /
    ``system_template_name`` made the base render it a second time with no
    variables, which raised at runtime; these agents instead return
    ``deps.text`` verbatim and carry no system prompt.
    """

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:  # noqa: PLR6301 — overrides KnoxAgentBase for its subclasses
        return deps.text


class QueryExpansionAgent(_PreRenderedPromptAgent, KnoxAgentBase[QueryExpansionOutput, PipelineAgentDeps]):
    """PydanticAI agent for multi-query expansion (RAG-Fusion)."""

    temperature: float | None = 0.0


class CragGradeAgent(_PreRenderedPromptAgent, KnoxAgentBase[CragGradeOutput, PipelineAgentDeps]):
    """PydanticAI agent for CRAG relevance grading."""

    temperature: float | None = 0.0


class HydeAgent(_PreRenderedPromptAgent, KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent for HyDE hypothetical answer generation."""

    temperature: float | None = 0.7
