"""The ``knox attachments`` command group.

Sub-commands
------------

* ``knox attachments add``    — register an attachment for a source.
* ``knox attachments remove`` — remove (tombstone) a registered attachment.
* ``knox attachments list``   — list registered attachments.

``att`` is a registered alias for ``attachments`` (e.g. ``knox att add``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from binaryornot.check import is_binary

from knox.cli._shared import vault_paths_from_ctx
from knox.cli.app import app as main_app
from knox.core.models.attachments import (
    IMAGE_EXTENSIONS,
    Attachment,
    AttachmentManifest,
)
from knox.core.paths import attachment_slug
from knox.logging import get_console

attachments_app = typer.Typer(
    name="attachments",
    help="Manage source attachments (supplementary files).",
    no_args_is_help=True,
)
main_app.add_typer(attachments_app)
main_app.add_typer(attachments_app, name="att")


@attachments_app.command("add")
def add_attachment(
    ctx: typer.Context,
    source: str = typer.Argument(help="Source file name or relative path (e.g., 'report.pdf')."),  # noqa: B008
    attachment_file: Path = typer.Argument(help="Path to the attachment file to register."),  # noqa: B008
    ingest: bool = typer.Option(False, "--ingest", help="Mark attachment as ingestible (processed during compile)."),  # noqa: B008
    embed: bool | None = typer.Option(None, "--embed/--no-embed", help="Override global embed_attachments setting for this attachment."),  # noqa: B008
) -> None:
    """Register an attachment for a source.

    The attachment file is copied to ``attachments/<source-slug>/`` and
    recorded in the attachment manifest (``.knox/attachments.jsonl``).
    """
    vp = vault_paths_from_ctx(ctx)
    console = get_console()

    # ── Validate source ────────────────────────────────────────────────
    source_file = Path(source)
    source_stem = source_file.stem  # e.g., "report" from "papers/report.pdf"
    slug = attachment_slug(source)

    # ── Validate attachment file ───────────────────────────────────────
    if not attachment_file.exists():
        console.print(f"[red]Error:[/red] Attachment file not found: {attachment_file}")
        raise typer.Exit(1)

    if attachment_file.is_dir():
        console.print(f"[red]Error:[/red] Attachment must be a file, not a directory: {attachment_file}")
        raise typer.Exit(1)

    ext = attachment_file.suffix.lower()

    # ── If --ingest, require a text file (or VLM-captionable image) ────
    if ingest and ext not in IMAGE_EXTENSIONS and is_binary(str(attachment_file)):
        console.print(
            f"[red]Error:[/red] Cannot ingest '{attachment_file.name}': binary files are not ingestible. "
            "Only text files (any extension) and images (via VLM) can be ingested."
        )
        raise typer.Exit(1)

    # ── Warn about VLM requirement for images ──────────────────────────
    if ingest and ext in IMAGE_EXTENSIONS:
        console.print(f"[yellow]Note:[/yellow] '{ext}' files require VLM for captioning. Will be skipped at compile if VLM is not configured.")

    # ── Create attachment directory and copy file ──────────────────────
    attachment_dir = vp.source_attachments_dir(slug)
    attachment_dir.mkdir(parents=True, exist_ok=True)
    dest = attachment_dir / attachment_file.name

    if dest.exists():
        console.print(f"[yellow]Warning:[/yellow] Attachment already exists at {dest}, overwriting.")

    shutil.copy2(attachment_file, dest)

    # ── Record in manifest ─────────────────────────────────────────────
    manifest = AttachmentManifest(vp.attachments_manifest_file)
    relative_file = f"{slug}/{attachment_file.name}"
    att = Attachment(
        path=relative_file,
        source=source_stem,
        ingestible=ingest,
        embed=embed,
    )
    manifest.append(att)

    console.print(f"[green]Registered attachment:[/green] {relative_file}")
    if ingest:
        console.print("  → ingestible (will be processed during compile)")
    if embed is not None:
        label = "embed" if embed else "no-embed"
        console.print(f"  → {label} (overrides global config)")


@attachments_app.command("remove")
def remove_attachment(
    ctx: typer.Context,
    attachment_file: str = typer.Argument(help="Attachment relative path (e.g., 'report/photo.jpg') or just filename."),
) -> None:
    """Remove (tombstone) a registered attachment.

    The manifest entry is marked as deleted. The file on disk is NOT
    removed — use ``rm`` manually if needed.
    """
    vp = vault_paths_from_ctx(ctx)
    console = get_console()

    manifest = AttachmentManifest(vp.attachments_manifest_file)
    existing = manifest.read_all()

    # ── Resolve path (support bare filename) ───────────────────────────
    resolved_file = attachment_file
    if "/" not in attachment_file and "\\" not in attachment_file:
        # Bare filename — find matching entry
        matches = [p for p in existing if Path(p).name == attachment_file]
        if len(matches) == 1:
            resolved_file = matches[0]
        elif len(matches) > 1:
            console.print(f"[yellow]Multiple matches for '{attachment_file}':[/yellow]")
            for m in matches:
                console.print(f"  {m}")
            console.print("Use the full path (e.g., 'slug/filename') to disambiguate.")
            raise typer.Exit(1)

    if resolved_file not in existing:
        console.print(f"[red]Error:[/red] Attachment not found: {resolved_file}")
        raise typer.Exit(1)

    if manifest.remove(resolved_file):
        console.print(f"[green]Removed attachment:[/green] {resolved_file}")
    else:
        console.print(f"[red]Error:[/red] Failed to remove attachment: {resolved_file}")
        raise typer.Exit(1)


@attachments_app.command("ingest")
def ingest_attachment(
    ctx: typer.Context,
    attachment_file: str = typer.Argument(None, help="Attachment relative path (e.g., 'report/photo.jpg') or just filename. Omit with --all."),  # noqa: B008
    all_attachments: bool = typer.Option(False, "--all", help="Apply to every registered attachment."),  # noqa: B008
) -> None:
    """Mark attachment(s) as ingestible (processed during compile).

    Independent of the global ``attachments.ingest_adopted`` gate: this
    sets the per-attachment manifest flag. Compile ingests only when the
    manifest flag AND the global gate are both true.

    Examples:
        knox attachments ingest report/photo.jpg
        knox attachments ingest --all
    """
    _set_ingestible(ctx, attachment_file, all_attachments, ingestible=True)


@attachments_app.command("no-ingest")
def no_ingest_attachment(
    ctx: typer.Context,
    attachment_file: str = typer.Argument(None, help="Attachment relative path (e.g., 'report/photo.jpg') or just filename. Omit with --all."),  # noqa: B008
    all_attachments: bool = typer.Option(False, "--all", help="Apply to every registered attachment."),  # noqa: B008
) -> None:
    """Mark attachment(s) as NOT ingestible (skipped during compile).

    Independent of the global ``attachments.ingest_adopted`` gate: this
    sets the per-attachment manifest flag. Compile ingests only when the
    manifest flag AND the global gate are both true.

    Examples:
        knox attachments no-ingest report/photo.jpg
        knox attachments no-ingest --all
    """
    _set_ingestible(ctx, attachment_file, all_attachments, ingestible=False)


def _set_ingestible(ctx: typer.Context, attachment_file: str | None, all_attachments: bool, *, ingestible: bool) -> None:
    """Shared implementation for ``ingest`` / ``no-ingest``."""
    vp = vault_paths_from_ctx(ctx)
    console = get_console()

    manifest = AttachmentManifest(vp.attachments_manifest_file)
    existing = manifest.read_all()
    if not existing:
        console.print("No attachments registered.")
        raise typer.Exit(1)

    if all_attachments:
        targets = list(existing)
    else:
        if not attachment_file:
            console.print("[red]Error:[/red] Provide an attachment path or use --all.")
            raise typer.Exit(1)
        resolved = attachment_file
        if "/" not in attachment_file and "\\" not in attachment_file:
            matches = [p for p in existing if Path(p).name == attachment_file]
            if len(matches) == 1:
                resolved = matches[0]
            elif len(matches) > 1:
                console.print(f"[yellow]Multiple matches for '{attachment_file}':[/yellow]")
                for m in matches:
                    console.print(f"  {m}")
                console.print("Use the full path (e.g., 'slug/filename') to disambiguate.")
                raise typer.Exit(1)
        if resolved not in existing:
            console.print(f"[red]Error:[/red] Attachment not found: {resolved}")
            raise typer.Exit(1)
        targets = [resolved]

    verb = "ingestible" if ingestible else "not ingestible"
    for path in targets:
        manifest.set_ingestible(path, ingestible)
        console.print(f"  {path:<40} -> {verb}")
    console.print(f"[green]Marked {len(targets)} attachment(s) {verb}.[/green]")


@attachments_app.command("list")
def list_attachments(
    ctx: typer.Context,
    source: str | None = typer.Option(None, "--source", "-s", help="Filter by source name."),
    show_all: bool = typer.Option(False, "--all", "-a", help="Include deleted entries."),
) -> None:
    """List registered attachments."""
    vp = vault_paths_from_ctx(ctx)
    console = get_console()

    manifest = AttachmentManifest(vp.attachments_manifest_file)

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
