"""parsecraft IR → kompend elements, and the error that says "nothing usable".

``map_document`` is the boundary where parsecraft's document model becomes
kompend's: pages stay in order, chunks become typed elements, figure assets
are stored and referenced. ``ParsecraftParseError`` is raised here because
this is the place that can prove a parse produced nothing — total pass
exhaustion, or an IR without a single content block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from parsecraft.ir.models import AssetKind, ChunkKind, DocumentResult, ExtractedAsset, StructuredChunk
from parsecraft.pipeline import ALL_PASSES_FAILED_CODE

from kompend.core.models import DocumentElement, ElementType, MetadataField
from kompend.preprocessing.images import ImageMode, ImageStore, asset_for_chunk, assets_by_id

#: parsecraft chunk kind → kompend element type. Kinds without a kompend equivalent
#: (header/footer/quote/unknown/caption) map to TEXT — they carry content and
#: dropping them would lose data. Keys are :class:`parsecraft.ir.models.ChunkKind`
#: members, not bare strings, so a chunk-kind rename upstream surfaces as a
#: type error here instead of a silent ``.get()`` miss.
CHUNK_KIND_TO_ELEMENT: dict[ChunkKind, ElementType] = {
    ChunkKind.HEADING: ElementType.HEADING,
    ChunkKind.PARAGRAPH: ElementType.TEXT,
    ChunkKind.LIST: ElementType.LIST_ITEM,
    ChunkKind.TABLE: ElementType.TABLE,
    ChunkKind.FORMULA: ElementType.FORMULA,
    ChunkKind.CODE: ElementType.CODE,
    ChunkKind.FIGURE: ElementType.IMAGE,
    ChunkKind.CAPTION: ElementType.TEXT,
    ChunkKind.HEADER: ElementType.TEXT,
    ChunkKind.FOOTER: ElementType.TEXT,
    ChunkKind.QUOTE: ElementType.TEXT,
    ChunkKind.UNKNOWN: ElementType.TEXT,
}


class ParsecraftParseError(RuntimeError):
    """parsecraft produced no usable content for a source file.

    Raised on total exhaustion (every pass failed for every page) or when the
    IR carries no content blocks. Dependency problems are NOT folded in here:
    parsecraft raises its own typed errors (``DependencyUnavailableError``,
    ``NoAnalyzerError``) and those propagate as-is so callers can tell
    "install an extra" from "bad file".
    """

    def __init__(self, source_file: Path, document: DocumentResult) -> None:
        failures = [
            f"p{page.page_number}: {diagnostic.message}"
            for page in document.pages
            for diagnostic in page.diagnostics
            if diagnostic.code == ALL_PASSES_FAILED_CODE
        ]
        degradations = [f"p{signal.page_number} {signal.name}: {signal.detail}" for signal in document.quality if signal.name.startswith("degraded-")]
        message = "; ".join(failures or degradations) or "all parsecraft passes failed"
        super().__init__(f"{source_file.name}: {message}")


def map_document(document: DocumentResult, source_file: Path, *, images: ImageStore | None = None) -> list[DocumentElement]:
    """Map parsecraft IR pages/chunks to kompend elements, order + pages preserved.

    Figure images travel with their chunk: the IR names the asset a chunk owns
    in the chunk metadata, the store decides where the crop lands, and the
    element carries the reference. An asset no chunk claims still belongs to
    the document and becomes an element of its own.

    Raises :class:`ParsecraftParseError` when execution genuinely exhausted —
    discriminated by execution evidence, NOT by ``winner is None`` (a parsecraft
    cache hit also yields ``winner=None`` with plan-shaped ``attempts == []``
    groups and a perfectly valid document).
    """
    exhausted = {page.page_number for page in document.pages if any(diagnostic.code == ALL_PASSES_FAILED_CODE for diagnostic in page.diagnostics)}
    if exhausted and exhausted == {page.page_number for page in document.pages}:
        raise ParsecraftParseError(source_file, document)

    store = images if images is not None else ImageStore(None, ImageMode.NONE)
    assets = assets_by_id(document) if store.enabled else {}
    claimed: set[str] = set()

    elements: list[DocumentElement] = []
    for page in document.pages:
        for chunk in page.blocks:
            asset = asset_for_chunk(chunk, assets)
            image_metadata: dict[str, str] = {}
            if asset is not None:
                claimed.add(asset.id)
                image_metadata = store.store(asset, source_stem=source_file.stem)
            element = chunk_to_element(chunk, source_file, len(elements), image_metadata=image_metadata)
            if element is not None:
                elements.append(element)
        for asset in page.assets:
            if asset.kind != AssetKind.IMAGE or asset.id in claimed:
                continue
            element = image_element(asset, page.page_number, source_file, len(elements), store)
            if element is not None:
                elements.append(element)
    if not elements:
        # A valid parsecraft document always carries at least one block; zero
        # content is a failure regardless of whether diagnostics were attached.
        raise ParsecraftParseError(source_file, document)
    return elements


def image_element(
    asset: ExtractedAsset,
    page_number: int,
    source_file: Path,
    reading_order: int,
    store: ImageStore,
) -> DocumentElement | None:
    """An image no chunk claimed, as an element of its own."""
    metadata = store.store(asset, source_stem=source_file.stem)
    reference = metadata.get(MetadataField.IMAGE_FILE) or metadata.get(MetadataField.IMAGE_BASE64)
    if not reference:
        return None
    return DocumentElement(
        type=ElementType.IMAGE,
        content=reference,
        page=page_number,
        reading_order=reading_order,
        metadata=metadata,
        source_file=source_file,
    )


def chunk_to_element(
    chunk: StructuredChunk,
    source_file: Path,
    reading_order: int,
    *,
    image_metadata: dict[str, str] | None = None,
) -> DocumentElement | None:
    """One IR chunk → one kompend element, or ``None`` when it carries no content."""
    element_type = CHUNK_KIND_TO_ELEMENT.get(chunk.kind, ElementType.TEXT)
    content = chunk_content(chunk)
    metadata: dict[str, Any] = {
        "parsecraft_kind": chunk.kind.value,
        "confidence": chunk.confidence,
    }
    if image_metadata:
        metadata.update(image_metadata)
    if element_type is ElementType.IMAGE:
        # The image reference is the element's content — the chunker reads
        # it back as the figure reference. The figure's own text rides
        # along as its description, so nothing the backend said is lost
        # and the captioning stage can use it instead of a VLM call.
        reference = image_metadata.get(MetadataField.IMAGE_FILE) or image_metadata.get(MetadataField.IMAGE_BASE64) or "" if image_metadata else ""
        if reference:
            if isinstance(content, str) and content.strip():
                metadata[MetadataField.DESCRIPTION] = content
            content = reference
        else:
            # No image to point at (no cache directory, unreadable payload):
            # the figure's own text is the content, and it is text, not an
            # image — an image element without a reference would break the
            # figure extractor downstream.
            element_type = ElementType.TEXT
    if not content:
        return None
    return DocumentElement(
        type=element_type,
        content=content,
        page=chunk.page_number,
        reading_order=reading_order,
        heading_level=heading_level(chunk),
        metadata=metadata,
        source_file=source_file,
    )


def chunk_content(chunk: StructuredChunk) -> str | list[list[str]]:
    """Chunk content in kompend's element shape.

    Tables may arrive as Markdown text or as rows. Rows win when present — the
    IR hands them over as a tuple of tuples while a TABLE element takes
    ``list[list[str]]``, so that boundary conversion happens here.

    Which backends populate ``rows`` is their business: since parsecraft
    2026.10.14 the CSV/TSV tier always does (header row first, ragged rows
    preserved), and liteparse does when the opt-in ``extract_vector_graphics``
    backend option is on. Everything else emits Markdown, which kompend's own
    markdown converter restores to a grid downstream.
    """
    if chunk.rows is not None:
        return [list(row) for row in chunk.rows]
    return chunk.content if isinstance(chunk.content, str) else ""


def heading_level(chunk: StructuredChunk) -> int | None:
    """Heading depth for heading chunks.

    parsecraft carries the level in chunk metadata as a string (``"1"``,
    ``"2"``); 2026.10.2 renamed the key to ``n_level``, and ``heading_level``
    is the 2026.10.1 name, still read for older wheels.

    The value is the *author* level: a Markdown ``#`` is 1, and 2026.10.14 made
    docling report the author's own outline level too (its renderer's ``+1``
    shift stays inside parsecraft). A missing level therefore means the backend
    had no outline to report, not that the path has no levels. kompend clamps
    the value to 1..6 where it renders.
    """
    if chunk.kind != "heading":
        return None
    level = chunk.metadata.get("n_level") or chunk.metadata.get("heading_level")
    if level is None:
        return None
    try:
        return int(level)
    except ValueError:
        return None
