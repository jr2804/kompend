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


class ErrorDecision(BaseModel):
    """A decision model's verdict on a failed pipeline step."""

    action: ErrorAction = Field(description="What the pipeline should do next with the failed unit of work.")


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
