"""Deduplication models (``[dedup]``)."""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import BeforeValidator, Field

from kompend.core.config.enums import MatcherName
from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar

DEFAULT_MATCHERS: Final[tuple[MatcherName, ...]] = (MatcherName.EMBEDDING, MatcherName.MINHASH)


class DedupConfig(_KompendBaseModel):
    """Deduplication configuration (opt-in by default)."""

    enabled: bool = Field(default=False, validation_alias=EnvVar.DEDUP_ENABLED)
    #: Comma-string (env var, legacy TOML) or array; both parse through
    #: :func:`parse_matchers`, which raises on an unknown name.
    matchers: Annotated[list[MatcherName], BeforeValidator(parse_matchers)] = Field(
        default_factory=lambda: list(DEFAULT_MATCHERS),
        validation_alias=EnvVar.DEDUP_MATCHERS,
    )
    threshold_identical: float = Field(default=0.95, validation_alias=EnvVar.DEDUP_THRESHOLD_IDENTICAL)
    threshold_similar: float = Field(default=0.80, validation_alias=EnvVar.DEDUP_THRESHOLD_SIMILAR)
    model: str = Field(default="", validation_alias=EnvVar.DEDUP_MODEL)
    minhash_lsh_threshold: float = Field(default=0.8, validation_alias=EnvVar.DEDUP_MINHASH_LSH_THRESHOLD)


def parse_matchers(value: object) -> list[MatcherName]:
    """Parse a ``dedup.matchers`` value into matcher members.

    Accepts the comma-string form (env var, legacy TOML) and the array form;
    an unknown name raises instead of being skipped, and a non-string,
    non-sequence value raises ``TypeError`` (raw config dicts are a trust
    boundary — anything can be in them).
    """
    if isinstance(value, str):
        parts: list[str | MatcherName] = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        msg = f"dedup.matchers must be a comma-string or a list, got {type(value).__name__}"
        raise TypeError(msg)
    matchers: list[MatcherName] = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, MatcherName):
            matchers.append(part)
            continue
        try:
            matchers.append(MatcherName(part.strip()))
        except ValueError as e:
            expected = ", ".join(member.value for member in MatcherName)
            msg = f"unknown dedup matcher {part!r}; expected one of: {expected}"
            raise ValueError(msg) from e
    return matchers
