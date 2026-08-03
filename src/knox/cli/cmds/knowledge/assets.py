"""The ``knox assets`` command group.

Sub-commands
------------

* ``knox assets add``    — register an attachment for a source.
* ``knox assets remove`` — remove (tombstone) a registered attachment.
* ``knox assets list``   — list registered attachments.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

from knox.cli._shared import vault_paths_from_ctx
from knox.cli.app import app as main_app
from knox.core.models.attachments import (
    IMAGE_EXTENSIONS,
    INGESTIBLE_EXTENSIONS,
    Attachment,
    AttachmentManifest,
)
from knox.core.paths import attachment_slug
from knox.logging import get_console

assets_app = typer.Typer(
    name="assets",
    help="Manage source attachments (supplementary files).",
    no_args_is_help=True,
)
main_app.add_typer(assets_app)


@assets_app.command("add")
def add_asset(
    ctx: typer.Context,
    source: str = typer.Argument(help="Source file name or relative path (e.g., 'report.pdf')."),  # noqa: B008
    asset_file: Path = typer.Argument(help="Path to the asset file to register."),  # noqa: B008
    ingest: bool = typer.Option(False, "--ingest", help="Mark asset as ingestible (processed during compile)."),  # noqa: B008
    embed: bool | None = typer.Option(None, "--embed/--no-embed", help="Override global embed_assets setting for this asset."),  # noqa: B008
) -> None:
    """Register an attachment for a source.

    The asset file is copied to ``assets/<source-slug>/`` and recorded
    in the attachment manifest (``.knox/assets.jsonl``).
    """
    vp = vault_paths_from_ctx(ctx)
    console = get_console()

    # ── Validate source ────────────────────────────────────────────────
    source_path = Path(source)
    source_stem = source_path.stem  # e.g., "report" from "papers/report.pdf"
    slug = attachment_slug(source)

    # ── Validate asset file ────────────────────────────────────────────
    if not asset_file.exists():
        console.print(f"[red]Error:[/red] Asset file not found: {asset_file}")
        raise typer.Exit(1)

    if asset_file.is_dir():
        console.print(f"[red]Error:[/red] Asset must be a file, not a directory: {asset_file}")
        raise typer.Exit(1)

    ext = asset_file.suffix.lower()

    # ── If --ingest, validate extension is ingestible ──────────────────
    if ingest and ext not in INGESTIBLE_EXTENSIONS:
        console.print(f"[red]Error:[/red] Cannot ingest '{ext}' files. Ingestible types: {sorted(INGESTIBLE_EXTENSIONS)}")
        raise typer.Exit(1)

    # ── Warn about VLM requirement for images ──────────────────────────
    if ingest and ext in IMAGE_EXTENSIONS:
        console.print(f"[yellow]Note:[/yellow] '{ext}' files require VLM for captioning. Will be skipped at compile if VLM is not configured.")

    # ── Create asset directory and copy file ───────────────────────────
    asset_dir = vp.source_attachments_dir(slug)
    asset_dir.mkdir(parents=True, exist_ok=True)
    dest = asset_dir / asset_file.name

    if dest.exists():
        console.print(f"[yellow]Warning:[/yellow] Asset already exists at {dest}, overwriting.")

    shutil.copy2(asset_file, dest)

    # ── Record in manifest ─────────────────────────────────────────────
    manifest = AttachmentManifest(vp.attachments_manifest_path)
    relative_path = f"{slug}/{asset_file.name}"
    att = Attachment(
        path=relative_path,
        source=source_stem,
        ingestible=ingest,
        embed=embed,
    )
    manifest.append(att)

    console.print(f"[green]Registered asset:[/green] {relative_path}")
    if ingest:
        console.print("  → ingestible (will be processed during compile)")
    if embed is not None:
        label = "embed" if embed else "no-embed"
        console.print(f"  → {label} (overrides global config)")


@assets_app.command("remove")
def remove_asset(
    ctx: typer.Context,
    asset_path: str = typer.Argument(help="Asset relative path (e.g., 'report/photo.jpg') or just filename."),
) -> None:
    """Remove (tombstone) a registered attachment.

    The manifest entry is marked as deleted. The file on disk is NOT
    removed — use ``rm`` manually if needed.
    """
    vp = vault_paths_from_ctx(ctx)
    console = get_console()

    manifest = AttachmentManifest(vp.attachments_manifest_path)
    existing = manifest.read_all()

    # ── Resolve path (support bare filename) ───────────────────────────
    resolved_path = asset_path
    if "/" not in asset_path and "\\" not in asset_path:
        # Bare filename — find matching entry
        matches = [p for p in existing if Path(p).name == asset_path]
        if len(matches) == 1:
            resolved_path = matches[0]
        elif len(matches) > 1:
            console.print(f"[yellow]Multiple matches for '{asset_path}':[/yellow]")
            for m in matches:
                console.print(f"  {m}")
            console.print("Use the full path (e.g., 'slug/filename') to disambiguate.")
            raise typer.Exit(1)

    if resolved_path not in existing:
        console.print(f"[red]Error:[/red] Asset not found: {resolved_path}")
        raise typer.Exit(1)

    if manifest.remove(resolved_path):
        console.print(f"[green]Removed asset:[/green] {resolved_path}")
    else:
        console.print(f"[red]Error:[/red] Failed to remove asset: {resolved_path}")
        raise typer.Exit(1)


@assets_app.command("list")
def list_assets(
    ctx: typer.Context,
    source: str | None = typer.Option(None, "--source", "-s", help="Filter by source name."),
    show_all: bool = typer.Option(False, "--all", "-a", help="Include deleted entries."),
) -> None:
    """List registered attachments."""
    vp = vault_paths_from_ctx(ctx)
    console = get_console()

    manifest = AttachmentManifest(vp.attachments_manifest_path)

    if not manifest.exists():
        console.print("No attachments registered.")
        return

    if show_all:
        # Read raw JSONL (including tombstones)
        entries = []
        for raw_line in manifest.path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                entries.append(Attachment.from_dict(json.loads(stripped)))
            except (json.JSONDecodeError, KeyError):
                continue
    else:
        entries = list(manifest.read_all().values())

    if source:
        entries = [e for e in entries if e.source == source]

    if not entries:
        label = f" for source '{source}'" if source else ""
        console.print(f"No attachments{label}.")
        return

    # ── Display table ──────────────────────────────────────────────────
    console.print(f"[bold]Attachments ({len(entries)}):[/bold]")
    for att in entries:
        status_parts = []
        if att.deleted:
            status_parts.append("[red]deleted[/red]")
        else:
            if att.ingestible:
                status_parts.append("[green]ingestible[/green]")
            if att.embed is True:
                status_parts.append("[blue]embed[/blue]")
            elif att.embed is False:
                status_parts.append("[dim]no-embed[/dim]")
            else:
                status_parts.append("[dim]embed=inherit[/dim]")
        status = " | ".join(status_parts)
        console.print(f"  {att.path:<40} [{att.source}] {status}")
