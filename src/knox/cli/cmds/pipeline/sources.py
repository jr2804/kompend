"""The ``knox sources`` command group.

Sub-commands
------------

* ``knox sources list``  — list ingested source files (optionally filter by tag).
* ``knox sources tag``   — add a tag to an ingested source file.
* ``knox sources untag`` — remove a tag from an ingested source file.

Tags are persisted in the per-file parse cache (``metadata.tags``), the same
store written by ``--tag`` at ingestion time. Wiki nodes pick up tag changes on
the next ``knox compile``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape

from knox.cli._shared import resolve_project_paths
from knox.cli.app import app
from knox.cli.args.other import SourceTags
from knox.core.config import resolve_config
from knox.core.pipeline.ingest import parse_cache_glob
from knox.logging import get_console, get_logger

logger = get_logger(__name__)

sources_app = typer.Typer(
    name="sources",
    help="Manage ingested source files and their tags.",
    no_args_is_help=True,
)
app.add_typer(sources_app)


# ── Sub-command: list ───────────────────────────────────────────────────


@sources_app.command("list", help="List ingested source files with their tags.")
def sources_list(
    ctx: typer.Context,
    tags: SourceTags = None,
    config: str | None = None,
) -> None:
    """List ingested source files with their tags.

    Optionally filter by tag using --tags (same semantics as query --tags).
    """
    cfg = resolve_config(config_path=config)
    _, _, cache_dir, _ = resolve_project_paths(cfg)

    cache_data = _load_all_cache_entries(cache_dir)
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
    for file_path in sorted(cache_data):
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

    # Display results.
    # Escape user-controlled text (file paths, tag names) so values containing
    # Rich markup metacharacters render verbatim. The literal brackets around
    # the tag list are themselves escaped (``\[``) so a tag named e.g.
    # "original" is not parsed as a ``[original]`` style and swallowed.
    for file_path, source_tags in results:
        if source_tags:
            tag_str = escape(", ".join(source_tags))
            get_console().print(rf"  {escape(file_path)}  \[{tag_str}]")
        else:
            get_console().print(f"  {escape(file_path)}  [dim]no tags[/]")


# ── Sub-command: tag / untag ────────────────────────────────────────────


@sources_app.command("tag", help="Add a tag to an ingested source file.")
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
    _mutate_tags(cache_dir, file_path, tag, add=True)


@sources_app.command("untag", help="Remove a tag from an ingested source file.")
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
    _mutate_tags(cache_dir, file_path, tag, add=False)


# ── Helpers ─────────────────────────────────────────────────────────────


def _load_all_cache_entries(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all per-file parse cache entries into a dict keyed by source_file."""
    entries: dict[str, dict[str, Any]] = {}
    for cache_path in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            source_file = data.get("source_file") or data.get("source_path", "")
            if source_file:
                entries[source_file] = data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load cache entry %s: %s", cache_path.name, exc)
    return entries


def _get_source_tags(file_path: str, cache_data: dict[str, Any]) -> list[str]:
    """Extract tags for a source file from the parse cache."""
    entry = cache_data.get(file_path, {})
    meta = entry.get("metadata", {})
    return meta.get("tags", [])  # nocheck: source metadata, not wiki metadata


def _mutate_tags(cache_dir: Path, file_path: str, tag: str, add: bool) -> None:
    """Add or remove a tag on a source file's cache entry."""
    cache_path = _find_cache_path(cache_dir, file_path)
    if cache_path is None:
        get_console().print(f"[red]No cache entry found for '{escape(file_path)}'. Run 'knox ingest' first.[/]")
        raise typer.Exit(1)

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        get_console().print(f"[red]Could not read cache entry: {exc}[/]")
        raise typer.Exit(1)

    meta = data.setdefault("metadata", {})
    existing_tags: list[str] = meta.get("tags", [])  # nocheck: source metadata, not wiki metadata

    if add:
        if tag in existing_tags:
            get_console().print(f"[yellow]Tag '{escape(tag)}' already present on {escape(file_path)}[/]")
            return
        existing_tags.append(tag)
        action = "Added"
    else:
        if tag not in existing_tags:
            get_console().print(f"[yellow]Tag '{escape(tag)}' not found on {escape(file_path)}[/]")
            return
        existing_tags.remove(tag)
        action = "Removed"

    meta["tags"] = existing_tags  # nocheck: source metadata, not wiki metadata

    try:
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        get_console().print(f"[red]Could not write cache entry: {exc}[/]")
        raise typer.Exit(1)

    get_console().print(f"[green]{action} tag '{escape(tag)}' on {escape(file_path)}[/]")


def _find_cache_path(cache_dir: Path, file_path: str) -> Path | None:
    """Find the cache file for a given source file path.

    Uses the stem of the filename to glob for matching cache entries.
    Returns the first match, or None if not found.
    """
    stem = Path(file_path).stem
    matches = sorted(cache_dir.glob(parse_cache_glob(stem)))
    if not matches:
        return None
    # If multiple matches (same stem, different content), verify by source_file field.
    if len(matches) == 1:
        return matches[0]
    for m in matches:
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
            if data.get("source_file") == file_path or data.get("source_path") == file_path:
                return m
        except (json.JSONDecodeError, OSError):
            continue
    return matches[0]  # fallback: first match
