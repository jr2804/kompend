"""Shared base model for every config section."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _KompendBaseModel(BaseModel):
    """Base for all config models.

    ``populate_by_name=True`` allows construction via field name (e.g.
    ``LLMConfig(default_model="x")``) even when a ``validation_alias`` is set
    (e.g. ``"KOMP_LLM_DEFAULT_MODEL"``).
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
