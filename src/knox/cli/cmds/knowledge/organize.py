"""``knox organize`` command — LLM-driven source file categorisation."""

from __future__ import annotations

from pathlib import Path

import typer

from knox.agents.categorize import CategorizeAgent
from knox.cli._shared import resolve_project_paths
from knox.cli.app import app
from knox.cli.args import (
    OrganizeCollapseSingle,
    OrganizeDryRun,
    OrganizeMaxDepth,
    OrganizeMinSubcategories,
)
from knox.core.config import resolve_config
from knox.core.config.keys import (
    COMPILE_LLM_TIMEOUT,
    ORGANIZE_COLLAPSE_SINGLE_FILE,
    ORGANIZE_MAX_DEPTH,
    ORGANIZE_MIN_SUBCATEGORIES,
    PATHS_WIKI_DIR,
)
from knox.logging import get_console, get_err_console, get_logger
from knox.organize.engine import OrganizeResult, organize_sources, resolve_organize_config

logger = get_logger(__name__)


@app.command(name="organize")
def organize_cmd(
    ctx: typer.Context,
    max_depth: OrganizeMaxDepth = None,
    min_subcategories: OrganizeMinSubcategories = None,
    collapse_single_file: OrganizeCollapseSingle = None,
    dry_run: OrganizeDryRun = False,
) -> None:
    """Categorise files in ``sources/`` into a hierarchical directory tree."""
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    overrides: dict[str, object] = {}
    if wiki_dir_override is not None:
        overrides[PATHS_WIKI_DIR] = wiki_dir_override
    if max_depth is not None:
        overrides[ORGANIZE_MAX_DEPTH] = max_depth
    if min_subcategories is not None:
        overrides[ORGANIZE_MIN_SUBCATEGORIES] = min_subcategories
    if collapse_single_file is not None:
        overrides[ORGANIZE_COLLAPSE_SINGLE_FILE] = collapse_single_file

    cfg = resolve_config(cli_overrides=overrides or None)
    sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)
    base_dir = Path(cfg.get(PATHS_WIKI_DIR) or sources_dir.parent).resolve()

    config = resolve_organize_config(overrides or None)

    timeout = cfg.get(COMPILE_LLM_TIMEOUT)
    default_model = cfg.get("compile.llm.default_model", "")
    if not default_model:
        get_err_console().print("No LLM model configured. Set compile.llm.default_model in config.")
        raise typer.Exit(1)

    categorize_agent = CategorizeAgent(
        default_model,
        timeout=float(timeout) if timeout else 120,
    )

    result = organize_sources(
        base_dir=base_dir,
        sources_dir=sources_dir,
        output_dir=output_dir,
        categorize_agent=categorize_agent,
        config=config,
        dry_run=dry_run,
    )

    _print_result(result, dry_run)


def _print_result(result: OrganizeResult, dry_run: bool) -> None:
    prefix = "Dry run — " if dry_run else ""
    n_cats = len(result.tree.categories)
    get_console().print(f"{prefix}proposed {n_cats} top-level categor{'y' if n_cats == 1 else 'ies'}.")
    _print_categories(result.tree.categories, indent=1)
    if result.tree.unfiled:
        get_console().print(f"  (unfiled: {len(result.tree.unfiled)} file(s) kept at top level)")
    if result.files_without_summary:
        get_err_console().print(
            f"Note: {len(result.files_without_summary)} file(s) had no extraction summary and were ignored.",
        )
    if dry_run:
        get_console().print("No files moved. Re-run without --dry-run to apply.")
        return
    get_console().print(f"Moved {result.moved_count} file(s).")
    if result.link_updates:
        get_console().print("Wiki link updates:")
        for location, count in result.link_updates.items():
            if count:
                get_console().print(f"  - {location}: {count}")


def _print_categories(categories: list, *, indent: int) -> None:
    pad = "  " * indent
    for cat in categories:
        n_sub = len(cat.subcategories)
        get_console().print(f"{pad}{cat.name}/ ({len(cat.files)} file(s), {n_sub} sub-categor{'y' if n_sub == 1 else 'ies'})")
        _print_categories(cat.subcategories, indent=indent + 1)
