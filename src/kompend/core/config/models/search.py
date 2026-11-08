"""Search and query models (``[search]``, ``[query]``)."""

from __future__ import annotations

from pydantic import Field

from kompend.core.config.enums import RerankerBackend
from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar
from kompend.core.vocab import LlmDataFormat, Verbosity


class QueryConfig(_KompendBaseModel):
    """Query command configuration."""

    format: LlmDataFormat = LlmDataFormat.MARKDOWN


class SearchExpansionConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_EXPANSION_ENABLED)
    variants: int = Field(default=3, validation_alias=EnvVar.SEARCH_EXPANSION_VARIANTS)


class SearchCragConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_CRAG_ENABLED)
    threshold: float = Field(default=0.3, validation_alias=EnvVar.SEARCH_CRAG_THRESHOLD)


class SearchRerankerConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_RERANKER_ENABLED)
    backend: RerankerBackend = Field(default=RerankerBackend.BI_ENCODER, validation_alias=EnvVar.SEARCH_RERANKER_BACKEND)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_RERANKER_MODEL)
    top_k: int = Field(default=20, validation_alias=EnvVar.SEARCH_RERANKER_TOP_K)


class SearchHydeConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_HYDE_ENABLED)


class SearchContextualConfig(_KompendBaseModel):
    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_CONTEXTUAL_ENABLED)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_CONTEXTUAL_MODEL)


class SearchAgenticConfig(_KompendBaseModel):
    """Opt-in agentic query path (agent searches the wiki via tools).

    Default off — the existing retrieve-then-synthesize path is unchanged.
    """

    enabled: bool = Field(default=False, validation_alias=EnvVar.SEARCH_AGENTIC_ENABLED)
    model: str = Field(default="", validation_alias=EnvVar.SEARCH_AGENTIC_MODEL)
    max_sub_questions: int = Field(default=5, validation_alias=EnvVar.SEARCH_AGENTIC_MAX_SUB_QUESTIONS)


class SearchSynthesisConfig(_KompendBaseModel):
    verbosity: Verbosity = Field(default=Verbosity.STANDARD, validation_alias=EnvVar.SEARCH_SYNTHESIS_VERBOSITY)


class SearchConfig(_KompendBaseModel):
    expansion: SearchExpansionConfig = Field(default_factory=SearchExpansionConfig)
    crag: SearchCragConfig = Field(default_factory=SearchCragConfig)
    reranker: SearchRerankerConfig = Field(default_factory=SearchRerankerConfig)
    hyde: SearchHydeConfig = Field(default_factory=SearchHydeConfig)
    contextual: SearchContextualConfig = Field(default_factory=SearchContextualConfig)
    agentic: SearchAgenticConfig = Field(default_factory=SearchAgenticConfig)
    synthesis: SearchSynthesisConfig = Field(default_factory=SearchSynthesisConfig)
