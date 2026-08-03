"""Attachment (supplementary file) ingestion for the KNOX pipeline.

Attachments are user-registered files that belong to a source but are not
ingested by default (photos, binaries, source code, etc.). They are tracked
by the JSONL manifest (``.knox/assets.jsonl``) and live under
``assets/<source-slug>/``.

Only attachments explicitly marked ``ingestible`` are processed at compile
time. Text/markdown attachments are parsed through the standard parser
router; PNG/JPG attachments are captioned via a VLM (if configured).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.core.models.attachments import IMAGE_EXTENSIONS, Attachment, AttachmentManifest
from knox.core.paths import VaultPaths
from knox.logging import get_logger
from knox.preprocessing.parsers.base import ParserRouter

logger = get_logger(__name__)

__all__ = [
    "ingest_attachment",
    "iter_ingestible_attachments",
]


def iter_ingestible_attachments(vp: VaultPaths) -> list[Attachment]:
    """Read the manifest and return all ingestible attachments on disk.

    Returns:
        List of :class:`Attachment` entries with ``ingestible=True`` that
        exist in the attachments directory. Missing files are logged and
        skipped.
    """
    manifest = AttachmentManifest(vp.attachments_manifest_path)
    if not manifest.exists():
        return []

    ingestible = []
    for att in manifest.read_all().values():
        if not att.ingestible:
            continue
        att_file = vp.attachments_dir / att.path
        if att_file.exists():
            ingestible.append(att)
        else:
            logger.warning("Attachment file missing, skipping: %s", att_file)
    return ingestible


def ingest_attachment(
    vp: VaultPaths,
    att: Attachment,
    router: ParserRouter,
    *,
    vlm_captioner: Any | None = None,
) -> ParsedDocument | None:
    """Ingest a single attachment through the parser pipeline.

    Text/markdown files are parsed via the standard router. PNG/JPG files
    are captioned via the VLM captioner (which must be provided).

    Args:
        vp: Vault paths.
        att: The attachment to ingest.
        router: Parser router for text files.
        vlm_captioner: Optional VLM captioner for image files.

    Returns:
        A parsed document, or ``None`` on failure/skip.
    """
    att_file = vp.attachments_dir / att.path
    if not att_file.exists():
        logger.error("Attachment file not found: %s", att_file)
        return None

    ext = att_file.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        if vlm_captioner is None:
            logger.warning(
                "Image attachment '%s' requires VLM captioning but VLM is not configured; skipping.",
                att.filename,
            )
            return None
        return _caption_image(att_file, att, vlm_captioner)

    if ext in {".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".htm", ".tex", ".ltx", ".adoc"}:
        return _parse_text(att_file, att, router)

    logger.warning("Unsupported attachment type '%s', skipping: %s", ext, att.filename)
    return None


def _parse_text(att_file: Path, att: Attachment, router: ParserRouter) -> ParsedDocument | None:
    """Parse a text/markdown attachment via the standard router."""
    try:
        doc = router.parse(att_file)
        doc.title = doc.title or att.filename
        _tag_metadata(doc, att)
        return doc
    except Exception:
        logger.debug("Failed to parse attachment %s", att_file, exc_info=True)
        return None


def _caption_image(att_file: Path, att: Attachment, vlm_captioner: Any) -> ParsedDocument | None:
    """Caption an image attachment via the VLM.

    Produces a synthetic document whose single text element holds the
    VLM-generated description.
    """
    try:
        caption = vlm_captioner.caption_image(att_file)
        if not caption:
            logger.warning("VLM returned empty caption for %s", att.filename)
            return None
    except Exception:
        logger.debug("Failed to caption image %s", att_file, exc_info=True)
        return None

    doc = ParsedDocument(
        source_file=att_file,
        title=att.filename,
        elements=[
            DocumentElement(
                type=ElementType.TEXT,
                content=caption,
            )
        ],
    )
    _tag_metadata(doc, att)
    doc.metadata["asset_type"] = "image_caption"
    return doc


def _tag_metadata(doc: ParsedDocument, att: Attachment) -> None:
    """Stamp source-asset metadata onto a parsed document.

    The metadata is also propagated onto every element so downstream
    chunking/indexing can distinguish attachment content.
    """
    if not doc.metadata:
        doc.metadata = {}
    doc.metadata.update(
        {
            "source_asset": True,
            "parent_source": att.source,
            "asset_path": att.path,
        }
    )
    for el in doc.elements:
        if not el.metadata:
            el.metadata = {}
        el.metadata.update(
            {
                "source_asset": True,
                "parent_source": att.source,
                "asset_path": att.path,
            }
        )
