"""Sub-config dataclasses for KNOX."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ImageMode(StrEnum):
    """Image handling mode for document parsing.

    - ``FILE``: save images as files (default)
    - ``BASE64``: embed images as base64-encoded data URIs
    """

    FILE = "file"
    BASE64 = "base64"


class OutputFormat(StrEnum):
    """Output format for the compiled wiki.

    - ``MARKDOWN``: Markdown files (default)
    - ``JSON``: JSON file
    - ``TYPST``: Typst file
    """

    MARKDOWN = "markdown"
    JSON = "json"
    TYPST = "typst"


@dataclass
class LLMConfig:
    """Configuration for LLM models, timeouts, and fallback behavior."""

    default_model: str
    vlm_model: str
    fallback_chain: list[str] = field(default_factory=list)
    fallback_max_retries: int = 2
    context_budget: int = 100_000
    timeout: float = 120.0
    role_models: dict[str, str] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """Token/character budgets for various LLM operations."""

    prompt: int = 200_000
    extract: int = 6_000
    generate_source_segment: int = 20_000
    concepts_per_batch: int = 8
    overview: int = 4_000
    verify: int = 3_000
    verify_min_ngram: int = 3
    lint_page: int = 100_000
    wicer_probe: int = 3_000
    wicer_eval: int = 4_000
    wicer_max_iterations: int = 3


@dataclass
class OutputConfig:
    """Configuration for output formatting and rendering."""

    format: OutputFormat = OutputFormat.MARKDOWN
    flavor: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    rumdl_config: Path | None = None


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and dimensions."""

    model: str
    dim: int = 384
