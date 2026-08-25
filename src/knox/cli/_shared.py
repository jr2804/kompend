"""Shared CLI helpers used across multiple command modules.

These functions resolve configuration, build CLI override dicts, and manage
hooks.  They are deliberately kept free of Typer-specific code so they can
be imported and unit-tested without a running CLI context.
"""

from __future__ import annotations

import logging
from typing import Any

import typer

from knox.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_FORCE,
    COMPILE_IN_PLACE_SOURCES,
    COMPILE_INDEX,
    COMPILE_LINT,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_FALLBACK_MAX_RETRIES,
    COMPILE_LLM_ROLE_MODELS,
    COMPILE_LLM_VLM_MODEL,
    COMPILE_MEMORY_ENABLED,
    COMPILE_NO_CACHE,
    COMPILE_PRIVATE,
    COMPILE_PROVENANCE,
    COMPILE_REVIEW,
    COMPILE_START_AT_TIER,
    COMPILE_STOP_AT_TIER,
    COMPILE_SYNTHESIZE,
    COMPILE_VERIFY,
    COMPILE_WORKERS,
    INGEST_PARSER,
    PATHS_WIKI_DIR,
    SEARCH_SYNTHESIS_VERBOSITY,
    TRUST_PRIVACY_FILTER,
    TRUST_USER_TRACKING,
)
from knox.core.paths import VaultPaths, vault_paths
from knox.logging import get_logger

logger = get_logger(__name__)

# Silence noisy third-party loggers that flood CLI output with
# model-loading chatter, empty-OCR warnings, and full ONNX/docling
# traceback boxes.  CRITICAL suppresses everything below — these
# errors are not actionable by the user; KNOX logs its own parse
# failures at the appropriate level.
logging.getLogger("docling").setLevel(logging.CRITICAL)
logging.getLogger("RapidOCR").setLevel(logging.CRITICAL)
logging.getLogger("rapidocr").setLevel(logging.CRITICAL)
logging.getLogger("onnxruntime").setLevel(logging.CRITICAL)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)


def vault_paths_from_ctx(ctx: typer.Context | None, *, require_vault: bool = True) -> VaultPaths:
    """Resolve VaultPaths from a Typer CLI context.

    Extracts ``--wiki-dir`` from the callback context and delegates to
    :func:`knox.core.paths.vault_paths`. ``require_vault=False`` is for
    config-introspection commands only (see :func:`vault_paths`).
    """
    wiki_dir = ctx.obj.get("wiki_dir") if ctx and ctx.obj else None
    return vault_paths(wiki_dir, require_vault=require_vault)


def build_cli_overrides(
    *,
    wiki_dir: str | None = None,
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
    ignore: list[str] | None = None,
    llm_default_model: str | None = None,
    vlm_model: str | None = None,
    memory_enabled: bool | None = None,
    llm_fallback_chain: list[str] | None = None,
    llm_fallback_max_retries: int | None = None,
    llm_role_model: list[str] | None = None,
    embed_model: str | None = None,
    in_place_sources: bool | None = None,
    no_user_tracking: bool | None = None,
    parser: str | None = None,
    synthesis_verbosity: str | None = None,
    cli_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat dotted-key dict of non-None CLI overrides.

    Output-shape and tuning values (format, flavor, frontmatter, kind,
    compiler, image_mode, budgets, timeouts, decay params) are NOT accepted
    here — they are config-only, set via ``knox config set`` or ``--set``.
    The escape hatch is *cli_set*: values already coerced by
    :func:`coerce_config_value` for a ``--set key=val`` option, merged last so
    it overrides everything else below CLI.
    """
    overrides: dict[str, Any] = {}

    _simple: list[tuple[str, Any]] = [
        (PATHS_WIKI_DIR, wiki_dir),
        (COMPILE_START_AT_TIER, start_at_tier),
        (COMPILE_STOP_AT_TIER, stop_at_tier),
        (COMPILE_WORKERS, workers),
        (COMPILE_LLM_DEFAULT_MODEL, llm_default_model),
        (COMPILE_LLM_VLM_MODEL, vlm_model),
        (COMPILE_EMBEDDING_MODEL, embed_model),
        (INGEST_PARSER, parser),
        (SEARCH_SYNTHESIS_VERBOSITY, synthesis_verbosity),
    ]
    for key, val in _simple:
        if val is not None:
            overrides[key] = val

    _flags: list[tuple[str, bool | None]] = [
        (COMPILE_NO_CACHE, no_cache),
        (COMPILE_FORCE, force),
        (COMPILE_IN_PLACE_SOURCES, in_place_sources),
        (COMPILE_REVIEW, review),
        (COMPILE_PROVENANCE, provenance),
        (COMPILE_PRIVATE, private),
        (COMPILE_VERIFY, verify),
        (COMPILE_SYNTHESIZE, synthesize),
        (COMPILE_INDEX, index_store),
        (COMPILE_LINT, compile_lint),
        (COMPILE_MEMORY_ENABLED, memory_enabled),
    ]
    for key, val in _flags:
        if val is not None:
            overrides[key] = val

    if no_privacy is not None:
        overrides[TRUST_PRIVACY_FILTER] = not no_privacy
    if no_user_tracking:
        overrides[TRUST_USER_TRACKING] = False
    if llm_fallback_chain is not None:
        overrides[COMPILE_LLM_FALLBACK_CHAIN] = list(llm_fallback_chain)
    if llm_fallback_max_retries is not None:
        overrides[COMPILE_LLM_FALLBACK_MAX_RETRIES] = llm_fallback_max_retries
    if llm_role_model is not None:
        parsed: dict[str, str] = {}
        for item in llm_role_model:
            if "=" in item:
                role, model = item.split("=", 1)
                parsed[role.strip()] = model.strip()
        if parsed:
            overrides[COMPILE_LLM_ROLE_MODELS] = parsed
    if cli_set:
        # --set values are already coerced and validated; merged last = highest precedence.
        overrides.update(cli_set)
    return overrides
