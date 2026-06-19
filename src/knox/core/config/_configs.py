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
    timeout: float = 120.0
    fallback_max_input: int = 8192
    fallback_max_output: int = 4096
    num_ctx_min: int = 2048
    num_ctx_max: int = 131072
    context_window_cap: int | None = None
    role_models: dict[str, str] = field(default_factory=dict)
    api_base: str | None = None
    api_key: str | None = None


@dataclass
class BudgetConfig:
    """Adaptive token budgets for LLM operations.

    Fixed char budgets have been replaced with adaptive sizing:
    - Input: context window discovered from model (knox-0rq)
    - Output: clamp(input_tokens * ratio, output_min, output_max) per phase (knox-vxd)
    - Chunking: clamp(available_tokens, chunk_size_min, chunk_size_max) (knox-7sp)
    """

    # Structural controls (not token budgets)
    concepts_per_batch: int = 8
    verify_min_ngram: int = 3
    wicer_max_iterations: int = 3
    min_content_length: int = 50
    min_concept_sources: int = 2
    max_concept_pages: int = 50

    # Chunk sizing safety rails (tokens)
    chunk_size_min: int = 2048
    chunk_size_max: int = 131072

    # Extraction output: ratio × input, clamped
    extract_output_ratio: float = 0.3
    extract_output_min: int = 512
    extract_output_max: int = 8192

    # Generation output: fixed ceiling per concept (expansion, not compression)
    gen_output_max: int = 16384

    # Overview output: ratio × total sources, clamped
    overview_output_ratio: float = 0.1
    overview_output_min: int = 256
    overview_output_max: int = 4096

    # Verification output: ratio × excerpt, clamped
    verify_output_ratio: float = 0.2
    verify_output_min: int = 256
    verify_output_max: int = 4096

    # Lint output
    lint_output_max: int = 8192

    # WiCER probe/eval output
    wicer_probe_output_max: int = 4096
    wicer_eval_output_max: int = 4096


@dataclass
class OutputConfig:
    """Configuration for output formatting and rendering."""

    format: OutputFormat = OutputFormat.MARKDOWN
    flavor: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    rumdl_config: Path | None = None
    max_related_links: int = 15


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models.

    Dimensions are auto-detected from the model at runtime — never hardcoded.
    """

    model: str
    batch_size: int = 10


@dataclass
class PdfConfig:
    """Configuration for PDF parsing to prevent OOM on large scanned PDFs.

    When a PDF exceeds *max_pages_before_chunking*, Docling processes it in
    overlapping page-range segments to keep ONNX/OCR memory bounded.
    """

    max_pages_before_chunking: int = 50
    page_chunk_size: int = 30
    page_chunk_overlap: int = 5


@dataclass
class OrganizeConfig:
    """Configuration for ``knox organize`` LLM-based file categorization.

    The command reads existing extraction summaries from a prior compilation
    and asks the LLM to propose a hierarchical directory tree for the
    ``sources/`` folder. Files are then moved into the new structure and
    wiki links are updated to match the new paths.
    """

    max_depth: int = 3
    min_subcategories: int = 2
    collapse_single_file: bool = True
