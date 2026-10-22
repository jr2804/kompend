"""The ``kompend sources`` command group.

Sub-commands
------------

* ``kompend sources list``  — list ingested source files (optionally filter by tag).
* ``kompend sources tag``   — add a tag to an ingested source file.
* ``kompend sources untag`` — remove a tag from an ingested source file.
* ``kompend sources export`` — render an ingested source as clean markdown.

Tags are persisted in the per-file parse cache (``metadata.tags``), the same
store written by ``--tag`` at ingestion time. Wiki nodes pick up tag changes on
the next ``kompend compile``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.markup import escape

from kompend.cli._shared import build_cli_overrides, vault_paths_from_ctx
from kompend.cli.app import app
from kompend.cli.args import (
    ExportAll,
    ExportAnnotations,
    ExportInline,
    ExportSourceId,
    ExportSummary,
    SourceTags,
)
from kompend.core import source_metadata
from kompend.core.config import resolve_config
from kompend.core.config.keys import EXPORT_ANNOTATIONS, EXPORT_SUMMARY
from kompend.core.paths import EXPORTED_DIR_NAME, INDEX_FILE, safe_stem
from kompend.core.pipeline.export import export_source
from kompend.core.utils.dicts import as_str
from kompend.logging import get_console, get_logger
from kompend.output.pages.index import sync_exported_sources_section

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
    vp = vault_paths_from_ctx(ctx)
    resolve_config(base_dir=vp.base_dir, config_file=config)
    cache_dir = vp.cache_dir

    cache_data = source_metadata.load_source_entries(cache_dir)
    if not cache_data:
        get_console().print("[yellow]No sources found. Run 'kompend ingest' first.[/]")
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
    for file in sorted(cache_data):
        source_tags = source_metadata.source_tags(cache_data[file])

        # Apply tag filter if specified
        if tag_groups:
            tag_set = set(source_tags)
            if not any(all(t in tag_set for t in group) for group in tag_groups):
                continue

        results.append((file, source_tags))

    if not results:
        get_console().print("[yellow]No sources match the given tag filter.[/]")
        raise typer.Exit(0)

    # Display results.
    # Escape user-controlled text (file paths, tag names) so values containing
    # Rich markup metacharacters render verbatim. The literal brackets around
    # the tag list are themselves escaped (``\[``) so a tag named e.g.
    # "original" is not parsed as a ``[original]`` style and swallowed.
    for file, source_tags in results:
        if source_tags:
            tag_str = escape(", ".join(source_tags))
            get_console().print(rf"  {escape(file)}  \[{tag_str}]")
        else:
            get_console().print(f"  {escape(file)}  [dim]no tags[/]")


# ── Sub-command: tag / untag ────────────────────────────────────────────


@sources_app.command("tag", help="Add a tag to an ingested source file.")
def sources_tag(
    ctx: typer.Context,
    file: str,
    tag: str,
    config: str | None = None,
) -> None:
    """Add a tag to an ingested source file.

    The tag is persisted in the parse cache and will propagate to
    wiki nodes on the next compile.
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(base_dir=vp.base_dir, config_file=config)
    _mutate_tags(vp.cache_dir, file, tag, add=True)


@sources_app.command("untag", help="Remove a tag from an ingested source file.")
def sources_untag(
    ctx: typer.Context,
    file: str,
    tag: str,
    config: str | None = None,
) -> None:
    """Remove a tag from an ingested source file.

    The tag is removed from the parse cache. Wiki nodes will need
    to be recompiled to reflect the change.
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(base_dir=vp.base_dir, config_file=config)
    _mutate_tags(vp.cache_dir, file, tag, add=False)


# ── Sub-command: attr / attr-rm ─────────────────────────────────────────


@sources_app.command("attr", help="Set a key=value attribute on an ingested source file.")
def sources_attr(
    ctx: typer.Context,
    file: str,
    key_value: str,
    config: str | None = None,
) -> None:
    """Set a key=value attribute on an ingested source file.

    The attribute is persisted in the parse cache and will propagate to
    wiki nodes on the next compile. Use KEY=VALUE format.
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(base_dir=vp.base_dir, config_file=config)
    _mutate_attributes(vp.cache_dir, file, key_value, add=True)


@sources_app.command("attr-rm", help="Remove an attribute key from an ingested source file.")
def sources_attr_rm(
    ctx: typer.Context,
    file: str,
    key: str,
    config: str | None = None,
) -> None:
    """Remove an attribute key from an ingested source file.

    The attribute is removed from the parse cache. Wiki nodes will need
    to be recompiled to reflect the change.
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(base_dir=vp.base_dir, config_file=config)
    _mutate_attributes(vp.cache_dir, file, key, add=False)


# ── Sub-command: export ─────────────────────────────────────────────────


@sources_app.command("export", help="Export an ingested source as clean markdown.")
def sources_export(  # noqa: PLR0917 — one param per typer CLI option
    ctx: typer.Context,
    source_id: ExportSourceId = "",
    export_all: ExportAll = False,
    output: str = "",
    inline: ExportInline = False,
    annotations: ExportAnnotations = None,
    summary: ExportSummary = None,
    config: str | None = None,
) -> None:
    """Render an ingested source as a clean, self-contained markdown file.

    The source is reconstructed from the Tier-1 parse cache ("as ingested"),
    not from the original file. Figures are copied to an assets directory next
    to the output file (or embedded as data URIs with --inline); attachments
    stay in the vault and are listed at the end with links.

    OUTPUT is optional — omit it to write to the vault's ``exported/``
    directory (reference material, outside wiki/); pass a path to export to
    any arbitrary file. With --all every ingested source is exported into
    ``exported/`` and the vault-root ``index.md`` is refreshed to list them.
    """
    vp = vault_paths_from_ctx(ctx)

    cli_set: dict[str, Any] = {}
    if annotations is not None:
        cli_set[EXPORT_ANNOTATIONS] = annotations.value
    if summary is not None:
        cli_set[EXPORT_SUMMARY] = summary.value
    cfg = resolve_config(
        base_dir=vp.base_dir,
        config_file=config,
        cli_overrides=build_cli_overrides(cli_set=cli_set),
    )
    ann = as_str(cfg.get(EXPORT_ANNOTATIONS)) or "off"
    summ = as_str(cfg.get(EXPORT_SUMMARY)) or "off"
    doc_headings = cfg.kompend.display.doc_headings

    if export_all:
        if source_id:
            get_console().print("[red]SOURCE_ID cannot be combined with --all.[/]")
            raise typer.Exit(1)
        if output:
            get_console().print("[red]--all exports into exported/; OUTPUT must be empty.[/]")
            raise typer.Exit(1)
        cache_data = source_metadata.load_source_entries(vp.cache_dir)
        if not cache_data:
            get_console().print("[yellow]No sources found. Run 'kompend ingest' first.[/]")
            raise typer.Exit(0)
        exported_dir = vp.base_dir / EXPORTED_DIR_NAME
        exported_dir.mkdir(parents=True, exist_ok=True)
        skipped = 0
        for file in sorted(cache_data):
            out_file = exported_dir / f"{safe_stem(file) or 'source'}.md"
            try:
                export_source(
                    base_dir=vp.base_dir,
                    cache_dir=vp.cache_dir,
                    source_id=file,
                    out_path=out_file,
                    inline=inline,
                    annotations=ann,
                    summary=summ,
                    doc_headings=doc_headings,
                )
            except FileNotFoundError as exc:
                get_console().print(f"[yellow]Skipped {escape(file)}: {escape(str(exc))}[/]")
                skipped += 1
        sync_exported_sources_section(vp.base_dir / INDEX_FILE, vp.base_dir)
        get_console().print(f"[green]Exported {len(cache_data) - skipped} source(s) -> {exported_dir}[/]")
        raise typer.Exit(0 if not skipped else 1)

    if not source_id:
        get_console().print("[red]Pass a SOURCE_ID or use --all.[/]")
        raise typer.Exit(1)
    if output:
        output_file = Path(output)
    else:
        exported_dir = vp.base_dir / EXPORTED_DIR_NAME
        output_file = exported_dir / f"{safe_stem(source_id) or 'source'}.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        get_console().print(f"No OUTPUT given — writing to {output_file}")
    if output_file.suffix.lower() != ".md":
        get_console().print(f"[red]Output file must end in .md, got '{escape(output_file.suffix or '<none>')}'[/]")
        raise typer.Exit(1)

    try:
        export_source(
            base_dir=vp.base_dir,
            cache_dir=vp.cache_dir,
            source_id=source_id,
            out_path=output_file,
            inline=inline,
            annotations=ann,
            summary=summ,
            doc_headings=doc_headings,
        )
    except FileNotFoundError as exc:
        get_console().print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    if not output:
        sync_exported_sources_section(vp.base_dir / INDEX_FILE, vp.base_dir)
    get_console().print(f"[green]Exported {escape(source_id)} -> {escape(str(output_file))}[/]")


# ── Helpers ─────────────────────────────────────────────────────────────


def _mutate_tags(cache_dir: Path, file: str, tag: str, add: bool) -> None:
    """Add or remove a tag on the source's parse-cache record."""
    try:
        change = source_metadata.apply_tag(cache_dir, file, tag, add=add)
    except LookupError:
        get_console().print(f"[red]No cache entry found for '{escape(file)}'. Run 'kompend ingest' first.[/]")
        raise typer.Exit(1) from None
    colour = "green" if change.changed else "yellow"
    get_console().print(f"[{colour}]{escape(change.message)}[/]")


def _mutate_attributes(cache_dir: Path, file: str, key_value: str, add: bool) -> None:
    """Set or remove a key=value attribute on the source's parse-cache record."""
    try:
        change = source_metadata.apply_attribute(cache_dir, file, key_value, add=add)
    except LookupError:
        get_console().print(f"[red]No cache entry found for '{escape(file)}'. Run 'kompend ingest' first.[/]")
        raise typer.Exit(1) from None
    except ValueError:
        get_console().print(f"[red]Use KEY=VALUE format for attributes, got: '{escape(key_value)}'[/]")
        raise typer.Exit(1) from None
    colour = "green" if change.changed else "yellow"
    get_console().print(f"[{colour}]{escape(change.message)}[/]")
