"""Attachment management MCP tools (add/remove; list/read/info live in knowledge.py)."""

import shutil
from pathlib import Path
from typing import Any

from binaryornot.check import is_binary

from knox.core.models.attachments import (
    IMAGE_EXTENSIONS,
    Attachment,
    AttachmentManifest,
)
from knox.core.paths import attachment_slug, vault_paths
from knox.logging import get_logger

logger = get_logger(__name__)


def attachments_add(
    wiki_dir: str | None = None,
    source: str = "",
    attachment_file: str = "",
    ingest: bool = False,
    embed: bool | None = None,
) -> dict[str, Any]:
    """Register an attachment for a source.

    Copies the attachment into attachments/<source-slug>/ and records it in
    the attachment manifest.
    """
    vp = vault_paths(wiki_dir)

    source_file = Path(source)
    slug = attachment_slug(source)

    att_path = Path(attachment_file)
    if not att_path.exists():
        return {"status": "error", "message": f"Attachment file not found: {attachment_file}"}
    if att_path.is_dir():
        return {"status": "error", "message": f"Attachment must be a file, not a directory: {attachment_file}"}

    ext = att_path.suffix.lower()

    if ingest and ext not in IMAGE_EXTENSIONS and is_binary(str(att_path)):
        return {
            "status": "error",
            "message": "Cannot ingest '{att_path.name}': binary files are not ingestible. Only text files and images (via VLM) can be ingested.",
        }

    attachment_dir = vp.source_attachments_dir(slug)
    attachment_dir.mkdir(parents=True, exist_ok=True)
    dest = attachment_dir / att_path.name

    if dest.exists():
        logger.warning("Attachment already exists at %s, overwriting", dest)

    shutil.copy2(att_path, dest)

    manifest = AttachmentManifest(vp.attachments_manifest_file)
    relative_file = f"{slug}/{att_path.name}"
    att = Attachment(
        path=relative_file,
        source=source_file.stem,
        ingestible=ingest,
        embed=embed,
    )
    manifest.append(att)

    info = {
        "status": "ok",
        "registered": relative_file,
        "ingestible": ingest,
    }
    if embed is not None:
        info["embed"] = embed
    return info


def attachments_remove(
    wiki_dir: str | None = None,
    attachment_file: str = "",
) -> dict[str, Any]:
    """Remove (tombstone) a registered attachment.

    The manifest entry is marked deleted; the file on disk is NOT removed.
    """
    vp = vault_paths(wiki_dir)

    manifest = AttachmentManifest(vp.attachments_manifest_file)
    if not manifest.exists():
        return {"status": "error", "message": "No attachment manifest found."}

    existing = manifest.read_all()

    # Resolve bare filename to full path
    resolved_file = attachment_file
    if "/" not in attachment_file and "\\" not in attachment_file:
        matches = [p for p in existing if Path(p).name == attachment_file]
        if len(matches) == 1:
            resolved_file = matches[0]
        elif len(matches) > 1:
            return {
                "status": "error",
                "message": f"Multiple matches for '{attachment_file}': {matches}. Use the full path (e.g. 'slug/filename').",
            }

    if resolved_file not in existing:
        return {"status": "error", "message": f"Attachment not found: {resolved_file}"}

    if manifest.remove(resolved_file):
        return {"status": "ok", "message": f"Removed attachment: {resolved_file}"}
    return {"status": "error", "message": f"Failed to remove attachment: {resolved_file}"}


def _set_ingestible(
    wiki_dir: str | None,
    attachment_file: str,
    *,
    all_attachments: bool,
    ingestible: bool,
) -> dict[str, Any]:
    vp = vault_paths(wiki_dir)

    manifest = AttachmentManifest(vp.attachments_manifest_file)
    if not manifest.exists():
        return {"status": "error", "message": "No attachment manifest found."}

    existing = manifest.read_all()
    if not existing:
        return {"status": "error", "message": "No attachments registered."}

    if all_attachments:
        targets = list(existing)
    else:
        if not attachment_file:
            return {"status": "error", "message": "Provide attachment_file or set all_attachments=True."}
        resolved_file = attachment_file
        if "/" not in attachment_file and "\\" not in attachment_file:
            matches = [p for p in existing if Path(p).name == attachment_file]
            if len(matches) == 1:
                resolved_file = matches[0]
            elif len(matches) > 1:
                return {
                    "status": "error",
                    "message": f"Multiple matches for '{attachment_file}': {matches}. Use the full path (e.g. 'slug/filename').",
                }
        if resolved_file not in existing:
            return {"status": "error", "message": f"Attachment not found: {resolved_file}"}
        targets = [resolved_file]

    for path in targets:
        manifest.set_ingestible(path, ingestible)
    return {
        "status": "ok",
        "updated": len(targets),
        "ingestible": ingestible,
    }


def attachments_ingest(
    wiki_dir: str | None = None,
    attachment_file: str = "",
    all_attachments: bool = False,
) -> dict[str, Any]:
    """Mark attachment(s) as ingestible (per-attachment manifest flag = True).

    Independent of the global ``attachments.ingest_adopted`` gate — compile
    ingests only when the manifest flag AND the global gate are both true.
    With *all_attachments*, applies to every registered attachment.
    """
    return _set_ingestible(
        wiki_dir,
        attachment_file,
        all_attachments=all_attachments,
        ingestible=True,
    )


def attachments_no_ingest(
    wiki_dir: str | None = None,
    attachment_file: str = "",
    all_attachments: bool = False,
) -> dict[str, Any]:
    """Mark attachment(s) as NOT ingestible (per-attachment manifest flag = False)."""
    return _set_ingestible(
        wiki_dir,
        attachment_file,
        all_attachments=all_attachments,
        ingestible=False,
    )
