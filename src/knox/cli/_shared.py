"""Shared CLI helpers used across multiple command modules.

These functions resolve configuration, build CLI override dicts, and manage
hooks.  They are deliberately kept free of Typer-specific code so they can
be imported and unit-tested without a running CLI context.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from knox.core.config import resolve_config
from knox.core.hooks import clear_hooks, load_hooks_from_config

# Silence noisy third-party loggers that corrupt the rich progress bar.
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_hooks() -> None:
    """Load event hooks from the resolved config into the global registry."""
    clear_hooks()
    try:
        cfg = resolve_config()
        load_hooks_from_config(cfg.raw)
    except Exception:
        logger.debug("Could not load hooks from config", exc_info=True)


def resolve_project_paths(cfg: Any) -> tuple[Path, Path, Path, Path]:  # noqa: ANN401
    """Return resolved source, output, cache, and state directories.

    Values come from the resolved config ``paths`` section.
    """
    sources_dir = Path(cfg.get("paths.sources_dir"))
    output_dir = Path(cfg.get("paths.output_dir"))
    cache_dir = Path(cfg.get("paths.cache_dir"))
    state_dir = Path(cfg.get("paths.state_dir"))
    return sources_dir, output_dir, cache_dir, state_dir


def build_cli_overrides(
    *,
    wiki_dir: str | None = None,
    target_format: str | None = None,
    image_mode: str | None = None,
    start_at_tier: str | None = None,
    stop_at_tier: str | None = None,
    no_cache: bool | None = None,
    workers: int | None = None,
    force: bool | None = None,
    review: bool | None = None,
    provenance: bool | None = None,
    private: bool | None = None,
    no_privacy: bool | None = None,
    verify: bool | None = None,
    synthesize: bool | None = None,
    index_store: bool | None = None,
    compile_lint: bool | None = None,
    flavor: str | None = None,
    frontmatter: list[str] | None = None,
    ignore: list[str] | None = None,
    llm_default_model: str | None = None,
    vlm_model: str | None = None,
    resolve_relations: bool | None = None,
    relation_confidence_threshold: float | None = None,
    memory_enabled: bool | None = None,
    memory_half_life_days: float | None = None,
    memory_decay_threshold: float | None = None,
    llm_fallback_chain: list[str] | None = None,
    llm_fallback_max_retries: int | None = None,
    llm_role_model: list[str] | None = None,
    kind: str | None = None,
    compiler: str | None = None,
    budget_concepts: int | None = None,
    budget_gen_segment: int | None = None,
    embed_model: str | None = None,
    embed_batch_size: int | None = None,
    llm_timeout: float | None = None,
) -> dict[str, Any]:
    """Build a flat dotted-key dict of non-None CLI overrides."""
    overrides: dict[str, Any] = {}

    _simple: list[tuple[str, Any]] = [
        ("paths.wiki_dir", wiki_dir),
        ("compile.format", target_format),
        ("compile.image_mode", image_mode),
        ("compile.start_at_tier", start_at_tier),
        ("compile.stop_at_tier", stop_at_tier),
        ("compile.workers", workers),
        ("compile.output.flavor", flavor),
        ("compile.llm.default_model", llm_default_model),
        ("compile.llm.vlm_model", vlm_model),
        ("compile.relation_confidence_threshold", relation_confidence_threshold),
        ("compile.memory_half_life_days", memory_half_life_days),
        ("compile.memory_decay_threshold", memory_decay_threshold),
        ("compile.page_kind", kind),
        ("compile.compiler", compiler),
        ("compile.budget.concepts_per_batch", budget_concepts),
        ("compile.budget.gen_output_max", budget_gen_segment),
        ("compile.embedding.model", embed_model),
        ("compile.embedding.batch_size", embed_batch_size),
        ("compile.llm.timeout", llm_timeout),
    ]
    for key, val in _simple:
        if val is not None:
            overrides[key] = val

    _flags: list[tuple[str, bool | None]] = [
        ("compile.no_cache", no_cache),
        ("compile.force", force),
        ("compile.review", review),
        ("compile.provenance", provenance),
        ("compile.private", private),
        ("compile.verify", verify),
        ("compile.synthesize", synthesize),
        ("compile.index", index_store),
        ("compile.lint", compile_lint),
        ("compile.resolve_relations", resolve_relations),
        ("compile.memory_enabled", memory_enabled),
    ]
    for key, val in _flags:
        if val:
            overrides[key] = val

    if no_privacy:
        overrides["trust.privacy_filter"] = False
    if llm_fallback_chain is not None:
        overrides["compile.llm.fallback_chain"] = list(llm_fallback_chain)
    if llm_fallback_max_retries is not None:
        overrides["compile.llm.fallback_max_retries"] = llm_fallback_max_retries
    if llm_role_model is not None:
        parsed: dict[str, str] = {}
        for item in llm_role_model:
            if "=" in item:
                role, model = item.split("=", 1)
                parsed[role.strip()] = model.strip()
        if parsed:
            overrides["compile.llm.role_models"] = parsed
    return overrides
