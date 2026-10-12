"""Reconstruct clean markdown from the Tier-1 parse cache.

``knox sources export`` renders a source "as ingested" from its parse-cache
JSON (a serialized :class:`~knox.core.models.elements.ParsedDocument`), not
from the original file. Images are embedded as data URIs (``--inline``) or
copied to ``<out-stem>_assets/`` next to the output file (default, no-inline).

Extracted knowledge can be attached as ``## Summary`` (directly below the
title) and ``## Annotated Items`` (at the very end), or written to
``extracted.json`` inside the assets directory. Placement is controlled by the
``export.annotations`` / ``export.summary`` config keys (values ``off`` /
``inline`` / ``asset``) and the ``--annotations`` / ``--summary`` CLI flags.

Portability rule: the rendered markdown only ever contains relative references
or data URIs — never absolute paths.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from knox.chunking.extractors import extract_structured
from knox.core import source_metadata
from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_DOC_HEADINGS
from knox.core.models import (
    AttachmentManifest,
    DocumentElement,
    ElementType,
    MetadataField,
)
from knox.core.models.enums import AssetKind
from knox.core.models.structured import structured_to_dict
from knox.core.page_builder import PageBuilder, link
from knox.core.paths import VaultPaths
from knox.core.pipeline.assets import AssetStore
from knox.core.verbosity import is_verbose
from knox.logging import get_logger

__all__ = ["export_source", "load_cache_entry", "render_document_markdown"]

#: Must match ``knox.compilation.asset_registry.ASSET_ID_HASH_LEN`` — the id is
#: content-addressed and stored in ``.knox/assets.json`` by the compilation
#: phase; export must reproduce the same id to look records up.
_ASSET_ID_HASH_LEN = 6

_IMAGE_PLACEHOLDER = re.compile(r"^\[Image:?\s*(.*)\]$")

#: Image extensions the exporter recognises. Keys are a :class:`StrEnum` so
#: the table is keyed by typed members; lookups still match bare suffix
#: strings because ``StrEnum`` is-a ``str``. Includes ``.svg`` because
#: embedded SVG figures are part of the export contract (the VLM agent's
#: narrower set lives in :data:`knox.agents.vlm._MIME_BY_EXT`).
class _ImageExt(StrEnum):
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    GIF = ".gif"
    WEBP = ".webp"
    BMP = ".bmp"
    SVG = ".svg"


_MIME_BY_SUFFIX: dict[_ImageExt, str] = {
    _ImageExt.PNG: "image/png",
    _ImageExt.JPG: "image/jpeg",
    _ImageExt.JPEG: "image/jpeg",
    _ImageExt.GIF: "image/gif",
    _ImageExt.WEBP: "image/webp",
    _ImageExt.BMP: "image/bmp",
    _ImageExt.SVG: "image/svg+xml",
}

logger = get_logger(__name__)


@dataclass
class _RenderState:
    """Mutable per-render state (asset dir, inline flag, asset counter)."""

    base_dir: Path
    assets_dir: Path
    inline: bool
    asset_counter: int = field(default=0)


def export_source(
    *,
    base_dir: Path,
    cache_dir: Path,
    source_id: str,
    out_path: Path,
    inline: bool,
    annotations: str,
    summary: str,
) -> Path:
    """Render the source identified by *source_id* to *out_path* (markdown).

    Returns the output file path. Raises :class:`FileNotFoundError` when no
    cache entry matches *source_id*.
    """
    data = load_cache_entry(source_id, cache_dir)
    assets_dir = out_path.parent / f"{out_path.stem}_assets"
    md = render_document_markdown(
        data,
        base_dir=base_dir,
        assets_dir=assets_dir,
        inline=inline,
        annotations=annotations,
        summary=summary,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path


def load_cache_entry(source_id: str, cache_dir: Path) -> dict[str, Any]:
    """Load the parse-cache record for *source_id*.

    *source_id* may be the ``source_file`` string printed by ``knox sources
    list`` (e.g. ``sources/doc.pdf``) or a bare stem. When several same-stem
    entries exist (content-discriminated cache), the entry whose
    ``source_file`` equals *source_id* wins. Read from the ``documents`` table
    of the single ``.knox/knox.db`` store (the former ``cache/<stem>-*.json``
    files are gone).
    """
    data = source_metadata.find_entry(cache_dir, source_id)
    if data is None:
        raise FileNotFoundError(f"No cache entry found for {source_id!r}; run 'knox ingest' first.")
    return data


def render_document_markdown(
    data: dict[str, Any],
    *,
    base_dir: Path,
    assets_dir: Path,
    inline: bool,
    annotations: str,
    summary: str,
) -> str:
    """Render a parse-cache dict to clean markdown.

    *annotations* / *summary* are ``off | inline | asset`` (resolved config or
    CLI values). In ``asset`` mode the extracted data is written to
    ``<assets_dir>/extracted.json`` instead of a markdown section. In
    ``no-inline`` mode image files are copied into *assets_dir* (side effect).
    """
    state = _RenderState(base_dir=base_dir, assets_dir=assets_dir, inline=inline)
    parts: list[str] = []

    title = str(data.get("title") or "").strip()
    if title:
        parts.append(f"# {title}")

    doc_summary = document_summary(data)
    if summary == "inline":
        parts.append(f"## Summary\n\n{doc_summary}")

    body = [md for el_dict in data.get("elements", []) if (md := _render_element(_element_from_dict(el_dict), state))]
    if body:
        parts.append("\n\n".join(body))

    attachments = _render_attachments(base_dir, data)
    if attachments:
        parts.append(attachments)

    annos = []
    if annotations != "off":
        store = _load_asset_store(base_dir)
        if store is not None:
            annos = collect_annotations(data, store)

    if annotations == "asset" or summary == "asset":
        _write_extracted_json(
            assets_dir,
            data,
            summary=doc_summary if summary == "asset" else None,
            annotations=annos if annotations == "asset" else None,
        )
    if annotations == "inline" and annos:
        parts.append(_render_annotations_section(annos))

    return "\n\n".join(p for p in parts if p).rstrip() + "\n"


def document_summary(data: dict[str, Any]) -> str:
    """One-line document summary: joined headings, else title, else stem.

    Mirrors ``chunking/contextual.py:_summarize_document`` (the canonical
    implementation) so exports and contextual chunking agree.
    """
    headings = [
        str(el["content"])
        for el in data.get("elements", [])
        if el.get("type") == ElementType.HEADING.value and isinstance(el.get("content"), str) and str(el["content"]).strip()
    ]
    if headings:
        limit = int(DEFAULTS.get(DISPLAY_DOC_HEADINGS, 5))
        return "; ".join(headings[:limit])
    return str(data.get("title") or Path(str(data.get(MetadataField.SOURCE_FILE, ""))).stem or "Untitled")


def collect_annotations(data: dict[str, Any], store: AssetStore) -> list[dict[str, Any]]:
    """Match table/figure elements to AssetRecords via content-addressed ids.

    The id is recomputed from the element with the same extractors the chunker
    used (``extract_structured``), so the lookup is exact — no source filter:
    identical content dedups to one record whose summary is content-derived.
    """
    out: list[dict[str, Any]] = []
    for el_dict in data.get("elements", []):
        el = _element_from_dict(el_dict)
        kind = _asset_kind(el.type)
        if kind is None:
            continue
        record_dict = _structured_dict(el)
        if record_dict is None:
            continue
        record = store.get(_asset_id(kind, record_dict))
        if record is None:
            continue
        caption = ""
        sd_data = record.data
        # All StructuredData variants carry ``caption``.
        caption = sd_data.caption
        out.append(
            {
                "id": record.id,
                "kind": record.kind.value,
                "caption": caption,
                "summary": record.summary,
                "page": el.page,
            }
        )
    return out


# ── element rendering ────────────────────────────────────────────────────


def _element_from_dict(d: dict[str, Any]) -> DocumentElement:
    src = d.get(MetadataField.SOURCE_FILE)
    return DocumentElement(
        type=ElementType(d["type"]),
        content=d.get("content", ""),
        page=d.get("page"),
        reading_order=d.get(MetadataField.READING_ORDER),
        heading_level=d.get("heading_level"),
        metadata=d.get("metadata") or {},
        source_file=Path(src) if src else None,
    )


def _render_element(el: DocumentElement, state: _RenderState) -> str:  # noqa: PLR0911 — one return per element type
    if el.type is ElementType.METADATA:
        return ""
    content = el.content
    if isinstance(content, list):
        return _render_table(content)
    if not isinstance(content, str):
        return ""
    text = content.strip()
    if not text:
        return ""
    if el.type is ElementType.HEADING:
        level = max(1, min(el.heading_level or 1, 6))
        return f"{'#' * level} {text}"
    if el.type is ElementType.LIST_ITEM:
        return f"- {text}"
    if el.type is ElementType.CODE:
        lang = ""
        if el.metadata:
            lang = str(el.metadata.get("language", "") or "").strip()
        return f"```{lang}\n{text}\n```"
    if el.type is ElementType.FORMULA:
        return f"$$\n{text}\n$$"
    if el.type is ElementType.IMAGE:
        return _render_image(el, state)
    return text


def _render_table(grid: list[list[str]]) -> str:
    """GFM pipe table; the first row is always the header."""
    if not grid:
        return ""
    b = PageBuilder()
    b.table(headers=grid[0], rows=grid[1:])
    return b.render().rstrip()


def _render_image(el: DocumentElement, state: _RenderState) -> str:
    content = el.content if isinstance(el.content, str) else ""
    caption = _image_caption(el, content)
    if not content:
        return f"![{caption}]()" if caption else ""
    if content.startswith("data:"):
        target = _materialize_data_uri(content, state)
    else:
        placeholder = _IMAGE_PLACEHOLDER.match(content)
        if placeholder:
            # No pixel data (parser fallback) — keep the placeholder as text.
            return f"*{content}*"
        content_path = Path(content)
        if not content_path.is_absolute():
            content_path = state.base_dir / content_path
        if not content_path.is_file():
            logger.warning("Image missing during export: %s", content)
            return f"*{content}*"
        target = _as_data_uri(content_path) if state.inline else _copy_asset(content_path, state)
    return f"![{caption}]({target})" if caption else f"![]({target})"


def _image_caption(el: DocumentElement, content: str) -> str:
    m = _IMAGE_PLACEHOLDER.match(content)
    if m and m.group(1):
        return m.group(1).strip()
    if el.metadata:
        for key in ("caption", "alt_text"):
            value = str(el.metadata.get(key, "") or "").strip()
            if value:
                return value
    return ""


def _copy_asset(src: Path, state: _RenderState) -> str:
    """Copy *src* into the assets dir; returns the link relative to the .md."""
    state.assets_dir.mkdir(parents=True, exist_ok=True)
    dst = state.assets_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return f"{state.assets_dir.name}/{dst.name}"


def _as_data_uri(path: Path) -> str:
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _materialize_data_uri(uri: str, state: _RenderState) -> str:
    """Decode a data URI into the assets dir (no-inline mode) or return it."""
    if state.inline:
        return uri
    header, _, b64 = uri.partition(",")
    suffix = _suffix_from_mime(header)
    state.asset_counter += 1
    name = f"image{state.asset_counter}{suffix}"
    state.assets_dir.mkdir(parents=True, exist_ok=True)
    (state.assets_dir / name).write_bytes(base64.b64decode(b64))
    return f"{state.assets_dir.name}/{name}"


def _suffix_from_mime(header: str) -> str:
    mime = header.removeprefix("data:").partition(";")[0]
    for suffix, candidate in _MIME_BY_SUFFIX.items():
        if candidate == mime:
            return suffix
    return ".bin"


# ── extracted knowledge ──────────────────────────────────────────────────


def _structured_dict(el: DocumentElement) -> dict[str, Any] | None:
    """Canonical structured dict for an element, as the chunker would produce."""
    sd = extract_structured(el)
    return structured_to_dict(sd) if sd is not None else None


def _asset_kind(el_type: ElementType) -> AssetKind | None:
    if el_type is ElementType.TABLE:
        return AssetKind.TABLE
    if el_type is ElementType.IMAGE:
        return AssetKind.FIGURE
    return None  # equations are deferred in the asset store (Phase 1)


def _asset_id(kind: AssetKind, record_dict: dict[str, Any]) -> str:
    """Content-addressed id — must match ``asset_registry.asset_id_for``."""
    raw = json.dumps(record_dict, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(raw.encode()).hexdigest()[:_ASSET_ID_HASH_LEN]
    return f"{kind.prefix}-{h}"


def _load_asset_store(base_dir: Path) -> AssetStore | None:
    try:
        return AssetStore(VaultPaths(base_dir).assets_file)
    except Exception as exc:
        logger.debug("Asset store unavailable — annotations skipped: %s", exc, exc_info=is_verbose())
        return None


def _render_annotations_section(annos: list[dict[str, Any]]) -> str:
    lines = ["## Annotated Items", ""]
    for a in annos:
        page = f" (p. {a['page']})" if a.get("page") else ""
        lines.append(f"### {a['id']}{page}")
        lines.append(f"- Kind: {a['kind']}")
        if a.get("caption"):
            lines.append(f"- Caption: {a['caption']}")
        if a.get("summary"):
            lines.append(f"- Summary: {a['summary']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _write_extracted_json(
    assets_dir: Path,
    data: dict[str, Any],
    *,
    summary: str | None,
    annotations: list[dict[str, Any]] | None,
) -> Path:
    payload: dict[str, Any] = {"source": data.get(MetadataField.SOURCE_FILE, "")}
    if summary is not None:
        payload["summary"] = summary
    if annotations is not None:
        payload["annotations"] = annotations
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = assets_dir / "extracted.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ── attachments ──────────────────────────────────────────────────────────


def _render_attachments(base_dir: Path, data: dict[str, Any]) -> str:
    """Table of registered attachments at the end of the file.

    Attachments stay in ``attachments/<source-stem>/`` (never embedded); the
    links point there relative to the exported file's directory.
    """
    vp = VaultPaths(base_dir)
    stem = Path(str(data.get(MetadataField.SOURCE_FILE, ""))).stem
    atts = AttachmentManifest(vp.attachments_manifest_file).read_for_source(stem)
    if not atts:
        return ""
    b = PageBuilder()
    b.heading(2, "Attachments")
    b.table(
        headers=["File", "Ingestible", "Link"],
        rows=[[rel.name, "yes" if a.ingestible else "no", link(rel.as_posix(), rel.name)] for a, rel in ((a, Path(a.path)) for a in atts)],
    )
    return b.render().rstrip()
