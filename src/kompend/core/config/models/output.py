"""Output, export, and display models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import BeforeValidator, Field

from kompend.core.config.enums import Flavor, OutputFormat
from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar
from kompend.core.vocab import ExportMode

FLAVOR_ALIASES: Final[dict[str, Flavor]] = {"github": Flavor.GFM}  # nocheck: str-registry — product-name aliases are an open input domain


class OutputConfig(_KompendBaseModel):
    """Configuration for output formatting and rendering.

    ``format`` defaults to ``markdown`` (an OKF v0.2-compliant vault).
    """

    format: OutputFormat = Field(default=OutputFormat.MARKDOWN, validation_alias=EnvVar.FORMAT)
    flavor: Annotated[Flavor | None, BeforeValidator(_coerce_flavor)] = Field(default=None, validation_alias=EnvVar.OUTPUT_FLAVOR)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    rumdl_config: Path | None = Field(default=None, validation_alias=EnvVar.RUMDL_CONFIG)
    max_related_links: int = Field(default=15, validation_alias=EnvVar.OUTPUT_MAX_RELATED_LINKS)


class ExportConfig(_KompendBaseModel):
    """Where extracted knowledge lands in ``kompend sources export``."""

    annotations: ExportMode = Field(default=ExportMode.INLINE, validation_alias=EnvVar.EXPORT_ANNOTATIONS)
    summary: ExportMode = Field(default=ExportMode.INLINE, validation_alias=EnvVar.EXPORT_SUMMARY)


class DisplayConfig(_KompendBaseModel):
    """Display/preview truncation limits.

    Each field caps how many items or how much text a single rendered
    section (index, log report, related-titles list, etc.) shows.
    Limits are deliberately configurable so users with long source lists
    or large captured notes can widen them without touching code.
    """

    source_names: int = Field(default=20, validation_alias=EnvVar.DISPLAY_SOURCE_NAMES)
    related_titles: int = Field(default=15, validation_alias=EnvVar.DISPLAY_RELATED_TITLES)
    doc_headings: int = Field(default=5, validation_alias=EnvVar.DISPLAY_DOC_HEADINGS)
    concept_names: int = Field(default=3, validation_alias=EnvVar.DISPLAY_CONCEPT_NAMES)
    memory_entries: int = Field(default=10, validation_alias=EnvVar.DISPLAY_MEMORY_ENTRIES)
    excerpt_chars: int = Field(default=300, validation_alias=EnvVar.DISPLAY_EXCERPT_CHARS)
    description_max_chars: int = Field(default=200, validation_alias=EnvVar.DISPLAY_DESCRIPTION_MAX_CHARS)
    cli_element_chars: int = Field(default=80, validation_alias=EnvVar.DISPLAY_CLI_ELEMENT_CHARS)
    tier_sample_size: int = Field(default=5, validation_alias=EnvVar.DISPLAY_TIER_SAMPLE_SIZE)
    cli_arg_preview: int = Field(default=4, validation_alias=EnvVar.DISPLAY_CLI_ARG_PREVIEW)
    log_error_chars: int = Field(default=500, validation_alias=EnvVar.DISPLAY_LOG_ERROR_CHARS)
    lint_feedback_chars: int = Field(default=200, validation_alias=EnvVar.DISPLAY_LINT_FEEDBACK_CHARS)
    memory_query_chars: int = Field(default=50, validation_alias=EnvVar.DISPLAY_MEMORY_QUERY_CHARS)
    mcp_element_chars: int = Field(default=500, validation_alias=EnvVar.DISPLAY_MCP_ELEMENT_CHARS)
    web_nodes: int = Field(default=10, validation_alias=EnvVar.DISPLAY_WEB_NODES)
    search_snippet_chars: int = Field(default=300, validation_alias=EnvVar.DISPLAY_SEARCH_SNIPPET_CHARS)


def _coerce_flavor(value: object) -> object:
    """Normalize a raw ``compile.output.flavor`` value for the schema.

    Lowercases, resolves aliases, then lets pydantic validate against the
    members; ``None`` passes through.
    """
    if isinstance(value, str) and value:
        return FLAVOR_ALIASES.get(value.strip().lower(), value.strip().lower())
    return value


def resolve_flavor(value: str) -> Flavor:
    """Resolve a programmatic flavor string to its member.

    Applies the same alias handling as the config schema; raises
    :class:`ValueError` for an unknown flavor.
    """
    normalized = FLAVOR_ALIASES.get(value.strip().lower(), value.strip().lower())
    return Flavor(normalized)
