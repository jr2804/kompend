"""The settings a command derives from its run context.

Turns the parsed options and the run context into the vault paths, the
config file and the override bag the pipeline consumes.
"""

from __future__ import annotations

import logging
from typing import Any

from kompend.cli.context import RunContext, current_run_context
from kompend.cli.errors import usage_error
from kompend.core.keys import (
    COMPILE_CACHE_IGNORE_MODEL,
    COMPILE_EMBEDDING_MODEL,
    COMPILE_FORCE,
    COMPILE_IN_PLACE_SOURCES,
    COMPILE_INDEX,
    COMPILE_LINT,
    COMPILE_LINT_ALL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_VLM_MODEL,
    COMPILE_MEMORY_ENABLED,
    COMPILE_PRIVATE,
    COMPILE_PROVENANCE,
    COMPILE_REPARSE,
    COMPILE_REVIEW,
    COMPILE_START_AT_TIER,
    COMPILE_STOP_AT_TIER,
    COMPILE_SYNTHESIZE,
    COMPILE_VERIFY,
    COMPILE_WORKERS,
    PATHS_RECURSIVE,
    PATHS_WIKI_DIR,
    SEARCH_SYNTHESIS_VERBOSITY,
    TRUST_PRIVACY_FILTER,
    TRUST_USER_TRACKING,
)
from kompend.core.paths import VaultPaths, vault_paths
from kompend.core.vocab import LintOutput
from kompend.logging import get_logger

logger = get_logger(__name__)

logging.getLogger("docling").setLevel(logging.CRITICAL)
logging.getLogger("RapidOCR").setLevel(logging.CRITICAL)
logging.getLogger("rapidocr").setLevel(logging.CRITICAL)
logging.getLogger("onnxruntime").setLevel(logging.CRITICAL)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)


def config_file_from_ctx(ctx: RunContext | None = None) -> str | None:
    """The ``--config`` path given to the root command, if any.

    ``--config`` is a run-level input like ``--wiki-dir``: it is declared once
    on the root app, published to the run context, and read here, so a command
    never has to repeat the flag (and cannot drift from the others).
    """
    return (ctx or current_run_context()).config_file


def vault_paths_from_ctx(ctx: RunContext | None = None, *, require_vault: bool = True) -> VaultPaths:
    """Resolve VaultPaths from the run context.

    Reads ``--wiki-dir`` from the ambient context and delegates to
    :func:`kompend.core.paths.vault_paths`. ``require_vault=False`` is for
    config-introspection commands only (see :func:`vault_paths`).
    """
    return vault_paths((ctx or current_run_context()).wiki_dir, require_vault=require_vault)


def resolve_lint_output(configured: LintOutput, *, verbose: bool, quiet: bool) -> LintOutput:
    """Resolve how much lint report reaches the terminal.

    ``--verbose-lint`` / ``--quiet-lint`` override ``lint.output``; passing both
    is a usage error rather than one silently winning.
    """
    if verbose and quiet:
        usage_error("--verbose-lint and --quiet-lint are mutually exclusive")
    if verbose:
        return LintOutput.FULL
    if quiet:
        return LintOutput.QUIET
    return configured


def build_cli_overrides(
    *,
    wiki_dir: str | None = None,
    start_at_tier: str | None = None,
    stop_at_tier: str | None = None,
    reparse: bool | None = None,
    ignore_model: bool | None = None,
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
    compile_lint_all: bool | None = None,
    ignore: list[str] | None = None,
    llm_default_model: str | None = None,
    vlm_model: str | None = None,
    memory_enabled: bool | None = None,
    llm_fallback_chain: list[str] | None = None,
    embed_model: str | None = None,
    in_place_sources: bool | None = None,
    recursive: bool | None = None,
    no_user_tracking: bool | None = None,
    synthesis_verbosity: str | None = None,
    cli_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat dotted-key dict of non-None CLI overrides.

    Output-shape and tuning values (format, flavor, frontmatter, kind,
    compiler, image_mode, budgets, timeouts, decay params) are NOT accepted
    here — they are config-only, set via ``kompend config set`` or ``--set``.
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
        (SEARCH_SYNTHESIS_VERBOSITY, synthesis_verbosity),
    ]
    for key, val in _simple:
        if val is not None:
            overrides[key] = val

    _flags: list[tuple[str, bool | None]] = [
        (COMPILE_REPARSE, reparse),
        (COMPILE_CACHE_IGNORE_MODEL, ignore_model),
        (COMPILE_FORCE, force),
        (COMPILE_IN_PLACE_SOURCES, in_place_sources),
        (PATHS_RECURSIVE, recursive),
        (COMPILE_REVIEW, review),
        (COMPILE_PROVENANCE, provenance),
        (COMPILE_PRIVATE, private),
        (COMPILE_VERIFY, verify),
        (COMPILE_SYNTHESIZE, synthesize),
        (COMPILE_INDEX, index_store),
        (COMPILE_LINT, compile_lint),
        (COMPILE_LINT_ALL, compile_lint_all),
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
    if cli_set:
        # --set values are already coerced and validated; merged last = highest precedence.
        overrides.update(cli_set)
    return overrides
