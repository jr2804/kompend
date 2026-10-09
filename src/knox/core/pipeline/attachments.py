"""Attachment (supplementary file) ingestion for the KNOX pipeline.

Attachments are user-registered files that belong to a source but are not
ingested by default (photos, binaries, etc.). They are tracked
by the ``attachment_manifest`` table of ``.knox/knox.db`` and live under
``attachments/<source-slug>/``.

Only attachments explicitly marked ``ingestible`` are processed at compile
time. Non-binary (text) attachments — markdown, plain text, source code, any
unknown text suffix — are parsed through the standard parser router; PNG/JPG
attachments are captioned via a VLM (if configured).
"""

from __future__ import annotations

from pathlib import Path

from binaryornot.check import is_binary

from knox.agents.vlm import VlmAgent, VlmDeps
from knox.core.config.rumdl import resolve_rumdl_config
from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.core.models.attachments import IMAGE_EXTENSIONS, Attachment, AttachmentManifest
from knox.core.paths import VaultPaths
from knox.core.text.encoding import normalize_to_utf8_path
from knox.core.text.rumdl import normalize_element_markdown
from knox.core.verbosity import is_verbose
from knox.logging import get_logger
from knox.preprocessing.parsers.base import ParserRouter

__all__ = [
    "ingest_attachment",
    "iter_ingestible_attachments",
]

logger = get_logger(__name__)


def iter_ingestible_attachments(vp: VaultPaths) -> list[Attachment]:
    """Read the manifest and return all ingestible attachments on disk.

    Returns:
        List of :class:`Attachment` entries with ``ingestible=True`` that
        exist in the attachments directory. Missing files are logged and
        skipped.
    """
    manifest = AttachmentManifest(vp.attachments_manifest_file)
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
    vlm_captioner: VlmAgent | None = None,
) -> ParsedDocument | None:
    """Ingest a single attachment through the parser pipeline.

    Text/markdown files are parsed via the standard router after
    auto-normalizing their encoding (UTF-8 pass-through; non-UTF-8
    re-encoded to ``vp.cache_dir / "encoding" / <hash>.utf8``).
    PNG/JPG files are captioned via the VLM captioner (which must be
    provided).

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

    # Non-image attachments: any non-binary file is text. The configured backend
    # is what decides which text extensions it claims — parsecraft's native-text
    # backend covers the whole text family (.c/.h/.cpp, source code, unknown
    # suffixes). A non-binary attachment whose extension the backend does not
    # claim raises in the router and is skipped by the handler below.
    if is_binary(str(att_file)):
        logger.warning("Unsupported attachment type '%s', skipping: %s", ext, att.filename)
        return None
    return _parse_text(att_file, att, router, vp.cache_dir)


def _parse_text(att_file: Path, att: Attachment, router: ParserRouter, cache_dir: Path) -> ParsedDocument | None:
    """Parse a text/markdown attachment via the standard router.

    The path passed to the parser is the UTF-8-normalized sibling of
    *att_file* (see :func:`normalize_to_utf8_path`); the original path
    is preserved on the :class:`ParsedDocument` only if the parser
    chooses to keep it — typical behavior is to read content and leave
    *source_file* untouched, so this is transparent to downstream.
    """
    try:
        parse_target, _encoding = normalize_to_utf8_path(att_file, cache_dir)
        doc = router.parse(parse_target)
        # Same part-level markdown normalisation as parse-tier sources.
        normalize_element_markdown(doc.elements, rumdl_config=resolve_rumdl_config(cache_dir.parent.parent))
        doc.title = doc.title or att.filename
        _tag_metadata(doc, att)
        return doc
    except Exception:
        logger.debug("Failed to parse attachment %s", att_file, exc_info=is_verbose())
        return None


def _caption_image(att_file: Path, att: Attachment, vlm_captioner: VlmAgent) -> ParsedDocument | None:
    """Caption an image attachment via the VLM.

    Produces a synthetic document whose single text element holds the
    VLM-generated description.
    """
    try:
        caption = vlm_captioner.run(
            VlmDeps(
                prompt=(
                    "Describe this image in one sentence for wiki documentation. "
                    "If it is decorative or contains no substantive information, respond with just SKIP."
                ),
                image_file=str(att_file),
                source_id=f"attachment-{att.filename}",
            )
        )
        if not caption or caption.strip().upper() == "SKIP":
            logger.warning("VLM returned empty caption for %s", att.filename)
            return None
    except Exception as exc:
        logger.debug("Failed to caption image %s: %s", att_file, exc, exc_info=is_verbose())
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
            "asset_file": att.path,
        }
    )
    for el in doc.elements:
        if not el.metadata:
            el.metadata = {}
        el.metadata.update(
            {
                "source_asset": True,
                "parent_source": att.source,
                "asset_file": att.path,
            }
        )
