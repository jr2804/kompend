"""Sub-config dataclasses for KNOX."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class LLMConfig:
    """Configuration for LLM models, timeouts, and fallback behavior."""
    default_model: str = "ollama/llama3.2"
    provider: str = "ollama"
    fallback_chain: list[str] = field(default_factory=list)
    fallback_max_retries: int = 2
    context_budget: int = 100_000
    timeout: float = 120.0
    role_models: dict[str, str] = field(default_factory=dict)
    vlm_model: str = "phi-3-vision"

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
    format: str = "markdown"
    flavor: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    rumdl_config: Path | None = None

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and dimensions."""
    model: str | None = None
    provider: str = "auto"
    dim: int = 384
