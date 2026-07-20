"""Source management subcommands: list, tag, untag.

Displays ingested source files with their tags and allows
managing tags on individual sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from knox.cli._shared import resolve_project_paths
from knox.cli.app import app
from knox.cli.args.other import SourceTags
from knox.core.config import resolve_config
from knox.logging import get_console, get_logger

logger = get_logger(__name__)


def _load_parse_cache(cache_dir: Path) -> dict[str, Any]:
    """Load the parse cache and return per-file metadata."""
    cache_path = cache_dir / "parse_cache.json"
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load parse cache: %s", exc)
        return {}


def _get_source_tags(file_path: str, cache_data: dict[str, Any]) -> list[str]:
    """Extract tags for a source file from the parse cache."""
    entry = cache_data.get(file_path, {})
    source_meta = entry.get("source_metadata", {})
    return source_meta.get("tags", [])  # nocheck: source metadata, not wiki metadata


@app.command(name="sources")
def sources_list(
    ctx: typer.Context,
    tags: SourceTags = None,
    config: str | None = None,
) -> None:
    """List ingested source files with their tags.

    Optionally filter by tag using --tags (same semantics as query --tags).
    """
    cfg = resolve_config(config_path=config)
    _, _, cache_dir, state_dir = resolve_project_paths(cfg)

    cache_data = _load_parse_cache(cache_dir)
    if not cache_data:
        get_console().print("[yellow]No sources found. Run 'knox ingest' first.[/]")
        raise typer.Exit(0)

    # Parse tag filter groups
    tag_groups = []
    if tags:
        for value in tags:
            group = [t.strip() for t in value.split(",") if t.strip()]
            if group:
                tag_groups.append(group)

    # Collect all sources
    results = []
    for file_path, _entry in sorted(cache_data.items()):
        source_tags = _get_source_tags(file_path, cache_data)

        # Apply tag filter if specified
        if tag_groups:
            tag_set = set(source_tags)
            if not any(all(t in tag_set for t in group) for group in tag_groups):
                continue

        results.append((file_path, source_tags))

    if not results:
        get_console().print("[yellow]No sources match the given tag filter.[/]")
        raise typer.Exit(0)

    # Display results
    for file_path, source_tags in results:
        tag_str = ", ".join(source_tags) if source_tags else "[dim]no tags[/]"
        get_console().print(f"  {file_path}  [{tag_str}]")


@app.command(name="sources-tag")
def sources_tag(
    ctx: typer.Context,
    file_path: str,
    tag: str,
    config: str | None = None,
) -> None:
    """Add a tag to an ingested source file.

    The tag is persisted in the parse cache and will propagate to
    wiki nodes on the next compile.
    """
    cfg = resolve_config(config_path=config)
    _, _, cache_dir, _ = resolve_project_paths(cfg)
    cache_path = cache_dir / "parse_cache.json"

    if not cache_path.exists():
        get_console().print("[red]No parse cache found. Run 'knox ingest' first.[/]")
        raise typer.Exit(1)

    try:
        with cache_path.open("r") as f:
            cache_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        get_console().print(f"[red]Could not read parse cache: {exc}[/]")
        raise typer.Exit(1)

    entry = cache_data.get(file_path, {})
    source_meta = entry.get("source_metadata", {})
    existing_tags = source_meta.get("tags", [])  # nocheck: source metadata, not wiki metadata

    if tag in existing_tags:
        get_console().print(f"[yellow]Tag '{tag}' already present on {file_path}[/]")
        return

    existing_tags.append(tag)
    source_meta["tags"] = existing_tags  # nocheck: source metadata, not wiki metadata
    entry["source_metadata"] = source_meta
    cache_data[file_path] = entry

    try:
        with cache_path.open("w") as f:
            json.dump(cache_data, f, indent=2)
    except OSError as exc:
        get_console().print(f"[red]Could not write parse cache: {exc}[/]")
        raise typer.Exit(1)

    get_console().print(f"[green]Added tag '{tag}' to {file_path}[/]")


@app.command(name="sources-untag")
def sources_untag(
    ctx: typer.Context,
    file_path: str,
    tag: str,
    config: str | None = None,
) -> None:
    """Remove a tag from an ingested source file.

    The tag is removed from the parse cache. Wiki nodes will need
    to be recompiled to reflect the change.
    """
    cfg = resolve_config(config_path=config)
    _, _, cache_dir, _ = resolve_project_paths(cfg)
    cache_path = cache_dir / "parse_cache.json"

    if not cache_path.exists():
        get_console().print("[red]No parse cache found. Run 'knox ingest' first.[/]")
        raise typer.Exit(1)

    try:
        with cache_path.open("r") as f:
            cache_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        get_console().print(f"[red]Could not read parse cache: {exc}[/]")
        raise typer.Exit(1)

    entry = cache_data.get(file_path, {})
    source_meta = entry.get("source_metadata", {})
    existing_tags = source_meta.get("tags", [])  # nocheck: source metadata, not wiki metadata

    if tag not in existing_tags:
        get_console().print(f"[yellow]Tag '{tag}' not found on {file_path}[/]")
        return

    existing_tags.remove(tag)
    source_meta["tags"] = existing_tags  # nocheck: source metadata, not wiki metadata
    entry["source_metadata"] = source_meta
    cache_data[file_path] = entry

    try:
        with cache_path.open("w") as f:
            json.dump(cache_data, f, indent=2)
    except OSError as exc:
        get_console().print(f"[red]Could not write parse cache: {exc}[/]")
        raise typer.Exit(1)

    get_console().print(f"[green]Removed tag '{tag}' from {file_path}[/]")
