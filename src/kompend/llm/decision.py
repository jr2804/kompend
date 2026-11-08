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
