"""``knox organize`` command — LLM-driven source file categorisation."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

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
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_TIMEOUT,
    ORGANIZE_COLLAPSE_SINGLE_FILE,
    ORGANIZE_MAX_DEPTH,
    ORGANIZE_MIN_SUBCATEGORIES,
    PATHS_WIKI_DIR,
)
from knox.llm.client import LLMClient
from knox.organize.engine import OrganizeResult, organize_sources, resolve_organize_config

logger = logging.getLogger(__name__)


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
    sources_dir_path, output_dir_path, _cache_dir, _state_dir = resolve_project_paths(cfg)
    base_dir = Path(cfg.get(PATHS_WIKI_DIR) or sources_dir_path.parent).resolve()

    config = resolve_organize_config(overrides or None)

    try:
        default_model = cfg.get(COMPILE_LLM_DEFAULT_MODEL)
        timeout = cfg.get(COMPILE_LLM_TIMEOUT)
        llm_client = LLMClient(default_model=default_model, timeout=timeout)
    except Exception as exc:
        typer.echo(f"Failed to initialize LLM client: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    result = organize_sources(
        base_dir=base_dir,
        sources_dir=sources_dir_path,
        output_dir=output_dir_path,
        llm_client=llm_client,
        config=config,
        dry_run=dry_run,
    )

    _print_result(result, dry_run)


def _print_result(result: OrganizeResult, dry_run: bool) -> None:
    prefix = "Dry run — " if dry_run else ""
    n_cats = len(result.tree.categories)
    typer.echo(f"{prefix}proposed {n_cats} top-level categor{'y' if n_cats == 1 else 'ies'}.")
    _print_categories(result.tree.categories, indent=1)
    if result.tree.unfiled:
        typer.echo(f"  (unfiled: {len(result.tree.unfiled)} file(s) kept at top level)")
    if result.files_without_summary:
        typer.echo(
            f"Note: {len(result.files_without_summary)} file(s) had no extraction summary and were ignored.",
            err=True,
        )
    if dry_run:
        typer.echo("No files moved. Re-run without --dry-run to apply.")
        return
    typer.echo(f"Moved {result.moved_count} file(s).")
    if result.link_updates:
        typer.echo("Wiki link updates:")
        for location, count in result.link_updates.items():
            if count:
                typer.echo(f"  - {location}: {count}")


def _print_categories(categories: list, *, indent: int) -> None:
    pad = "  " * indent
    for cat in categories:
        n_sub = len(cat.subcategories)
        typer.echo(f"{pad}{cat.name}/ ({len(cat.files)} file(s), {n_sub} sub-categor{'y' if n_sub == 1 else 'ies'})")
        _print_categories(cat.subcategories, indent=indent + 1)
