"""Decision-model sidecar models (``[decision]``)."""

from __future__ import annotations

from pydantic import Field

from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar


class DecisionConfig(_KompendBaseModel):
    """Configuration for the optional decision-model sidecar (TypeSafe SystemOne).

    A decision model answers questions the pipeline cannot answer from types
    alone: "is this failure worth repeating?", later "which passage answers
    this?". It is optional - with ``enabled = false`` callers use the
    deterministic fallback in :mod:`kompend.llm.errors`.
    """

    enabled: bool = Field(default=False, validation_alias=EnvVar.DECISION_ENABLED)
    #: Full ``provider/model`` string (for example ``ollama/clef`` or
    #: ``typesafe/jev-latest``). Provider credentials come from the
    #: ``[providers.*]`` blocks - a decision model is not a special provider.
    model: str | None = Field(default=None, validation_alias=EnvVar.DECISION_MODEL)
    timeout: float = Field(default=10.0, validation_alias=EnvVar.DECISION_TIMEOUT)
