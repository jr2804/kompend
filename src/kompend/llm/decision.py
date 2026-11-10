"""Decision-model vocabulary and configuration helpers.

A decision model is an ordinary kompend agent (:mod:`kompend.agents.decision`)
backed by any model that can answer typed questions. TypeSafe's Jev is one
provider (``typesafe/jev-latest``); Ollama serves decision models too
(``ollama/clef``), and so do other providers. The model is therefore named
exactly like every other model in kompend - a ``provider/model`` string whose
credentials and base URL come from the ``[providers.*]`` blocks - instead of
through a hardcoded provider.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

from kompend.core.config._configs import DecisionConfig


class ErrorAction(StrEnum):
    """What to do with a failed unit of work."""

    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"


class FigureRelevanceVerdict(StrEnum):
    """Whether a figure carries information worth a VLM caption."""

    INFORMATIVE = "informative"
    DECORATIVE = "decorative"


class QueryIntentVerdict(StrEnum):
    """What a user query is asking for.

    The vocabulary mirrors :class:`kompend.search.classifier.QueryIntent`: the
    regex classifier and the decision model answer the same question, so one
    pair of enums keeps both branches comparable.
    """

    KNOWLEDGE = "knowledge"
    GREETING = "greeting"
    GRATITUDE = "gratitude"
    HELP = "help"
    OFFTOPIC = "offtopic"


class ErrorDecision(BaseModel):
    """A decision model's verdict on a failed pipeline step."""

    action: ErrorAction = Field(description="What the pipeline should do next with the failed unit of work.")
    shrink: bool = Field(
        default=False,
        description="Retry the unit at half size. Only the generation shrink loop reads this; every other caller ignores it.",
    )


class QueryIntentDecision(BaseModel):
    """A decision model's verdict on a user query's intent."""

    intent: QueryIntentVerdict = Field(description="What the user is asking for.")


class FigureRelevanceDecision(BaseModel):
    """A decision model's verdict on whether a figure deserves a caption.

    Choices and bools only, never free-form text: the caption is the VLM's
    job, the decision is the classifier's. A ``decorative`` verdict skips
    the caption call; it is never published as a caption.
    """

    verdict: FigureRelevanceVerdict = Field(description="Whether the figure is worth captioning.")


class CandidateVerdict(BaseModel):
    """Whether one candidate chunk carries material that answers the query.

    A single boolean field is the widest verdict the decision grammar can
    carry here. A list of models is refused before any request is sent —
    "Output field 'verdicts' is not supported by this model: a list must be of
    two or more string options" — and a list of string options collapses
    verdicts (3 candidates answered as 1), so the tier asks one candidate at
    a time instead.
    """

    relevant: bool = Field(description="True when the chunk contains material that answers the query.")


#: Every result type kompend hands to a decision model, with the stable label
#: the doctor reports it under. One registry, one place: a decision grammar
#: that refuses a type makes that type's tier silently degrade to a fallback,
#: so ``kompend doctor`` validates this tuple offline (check
#: ``decision-schema``) instead of finding out from a dead tier.
DECISION_RESULT_TYPES: Final[tuple[tuple[str, type[BaseModel]], ...]] = (
    ("error-triage", ErrorDecision),
    ("query-intent", QueryIntentDecision),
    ("figure-relevance", FigureRelevanceDecision),
    ("source-search-verdict", CandidateVerdict),
)


def decision_model_string(config: DecisionConfig | None) -> str | None:
    """Return the configured ``provider/model`` string, if a decision model is enabled.

    The string is used verbatim: provider selection, credentials and base URL
    are the same ones every other model uses, so a decision model needs no
    separate provider configuration.

    ``None`` means "no decision model": the caller keeps its deterministic
    fallback instead of asking for a judgement call.
    """
    if config is None or not config.enabled or not config.model:
        return None
    return config.model
