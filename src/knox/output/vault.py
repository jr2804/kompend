"""Shared vault-writing algorithm for KNOX output writers.

The vault layout — one ``.md`` file per wiki node, directory routing by kind,
stale-file clearing, related-edge resolution, in-text citation rewriting, and
the references/index/log page orchestration — is implemented exactly once
here in :class:`VaultWriter`.

OKF (Open Knowledge Format v0.2) frontmatter is the **unique, canonical
standard** every vault writer emits. The single markdown writer
(:class:`~knox.output.markdown.MarkdownWriter`) is OKF-compliant and uses
portable relative links — no Obsidian-only syntax.

Subclasses fill in the rendering hooks that differ by format:

- :meth:`_wiki_to_documents` — build the ``Document`` list (frontmatter, etc.)
- :meth:`_render_document` — render one page (body already citation-rewritten)
- :meth:`_write_references_page` — bibliography page
- :meth:`_write_index` — root index page
- :meth:`_write_log` — bundle log page
"""

from __future__ import annotations

import json
import os
import re
import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml

from knox import __version__ as _KNOX_VERSION
from knox.core.models import CompiledWiki, GraphEdge, ImageRef, MetadataField, WikiNode
from knox.core.models.assets import AssetRecord
from knox.core.models.enums import AssetKind
from knox.core.page_builder import PageBuilder, link, strong
from knox.core.paths import (
    ATTACHMENTS_DIR_NAME,
    DEFAULT_OUTPUT_DIR_NAME,
    DEFAULT_SOURCES_DIR_NAME,
    EXPORTED_DIR_NAME,
    INDEX_FILE,
    KNOX_DIR,
    LOG_FILE,
    QUALITY_FILE,
    SUMMARIES_SUBDIR,
    VaultPaths,
    attachment_slug,
    is_protected_source,
)
from knox.core.pipeline.assets import AssetStore
from knox.core.pipeline.references import ReferenceStore
from knox.core.utils.time import iso_local
from knox.logging import get_logger
from knox.output.base import BaseOutputWriter
from knox.output.models import Document, OkfSource, TrustEvent
from knox.output.reference_links import format_bibliography_entry, rewrite_source_references

# OKF spec version every vault writer targets.
OKF_VERSION = "0.2"

# OKF actor convention (§7) for the producer of generated content.
# ``<producer>/<version>`` for agents and tools.
_GENERATED_BY = f"knox/{_KNOX_VERSION or '0.0.0'}"

# Map KNOX node_type values to OKF ``type`` values. OKF types are free-form
# strings (§4.1); these are descriptive and self-explanatory per the spec.
KIND_TO_TYPE: dict[str, str] = {
    "concept": "Concept",
    "comparison": "Comparison",
    "entity": "Entity",
    "summary": "Summary",
    "overview": "Overview",
    "procedure": "Procedure",
    "section": "Section",
}
DEFAULT_TYPE = "Concept"

# Regex to find ``[[wikilinks]]`` or ``[[wikilinks|alias]]`` in body text.
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

logger = get_logger(__name__)


class _NoAliasDumper(yaml.Dumper):
    """YAML Dumper that disables anchor/alias emission."""

    def ignore_aliases(self, data: object) -> bool:  # noqa: PLR6301
        return True


class VaultWriter(BaseOutputWriter):
    """Base class for writers that emit a vault of ``.md`` files.

    Subclasses fill in the rendering hooks that differ between flavors; the
    shared vault algorithm lives here so it is written exactly once. The
    frontmatter is already uniform — see :meth:`okf_frontmatter`.
    """

    # node_type -> subfolder under the vault root. Kinds not listed here
    # (overview, procedure, section, unknown) stay at the root so the
    # layout never splits a type the compiler didn't anticipate.
    _KIND_FOLDERS: ClassVar[dict[str, str]] = {
        "concept": "concepts",
        "comparison": "concepts",
        "entity": "entities",
        "summary": "concepts",
    }

    def __init__(
        self,
        *,
        max_related_links: int = 15,
        references_file: Path | None = None,
        assets_file: Path | None = None,
        base_dir: Path | None = None,
        output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_related_links = max_related_links
        self._references_file = references_file
        self._assets_file = assets_file
        self._base_dir = base_dir
        self._output_dir_name = output_dir_name

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        docs = self._wiki_to_documents(wiki)
        return self._write_vault(docs, wiki, output_dir)

    def portablize_images(
        self,
        images: list[ImageRef],
        provenance: list[str],
        folder: str,
    ) -> list[ImageRef]:
        """Copy ``.knox/``-internal images into ``attachments/`` and rewrite paths.

        Markdown-only invariant: wiki pages must never reference ``.knox/``
        (the cache is derived state and may be absent). Any embedded image
        living under ``.knox/`` is copied to
        ``attachments/<source-slug>/<filename>`` (existing per-source layout)
        and the reference rewritten to a link relative to the page's
        *folder*. Images outside ``.knox/`` and data URIs pass through
        unchanged. Requires ``base_dir``; without it nothing is rewritten.
        """
        if self._base_dir is None or not images:
            return images
        # Provenance entries may be bare stems ("report") or source-relative
        # paths ("sources/report.pdf"); the attachment flow slugs the source
        # name without the sources/ prefix, so strip it first.
        prov = provenance[0] if provenance else ""
        prefix = f"{DEFAULT_SOURCES_DIR_NAME}/"
        if prov.startswith(prefix):
            prov = prov[len(prefix) :]
        slug = attachment_slug(prov) if prov else ""
        if not slug:
            return images
        out: list[ImageRef] = []
        for img in images:
            raw = img.get("path", "")
            rel = self._knox_relative(raw)
            if rel is None:
                out.append(img)
                continue
            src_file = self._base_dir / rel
            if not src_file.is_file():
                out.append(img)
                continue
            dest = self._base_dir / ATTACHMENTS_DIR_NAME / slug / src_file.name
            if not dest.exists() or dest.read_bytes() != src_file.read_bytes():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest)
            link = os.path.relpath(
                dest.relative_to(self._base_dir),
                f"{self._output_dir_name}/{folder}" if folder else self._output_dir_name,
            ).replace("\\", "/")
            new_img = img.copy()
            new_img["path"] = link
            out.append(new_img)
        return out

    # ── Canonical OKF frontmatter ────────────────────────────────────────
    # This is the single, unique frontmatter schema every vault writer
    # emits. Markdown flavors differ only in body link syntax, not here.

    @staticmethod
    def okf_type(node_type: str | None) -> str:
        """Map a KNOX node_type to an OKF type string."""
        if not node_type:
            return DEFAULT_TYPE
        return KIND_TO_TYPE.get(node_type, node_type.title() or DEFAULT_TYPE)

    @staticmethod
    def okf_description(node: WikiNode) -> str:
        """Extract a one-line description from a wiki node.

        Uses the first sentence of the first non-empty content line, with
        ``[[wikilinks]]`` reduced to plain text. Generated pages open with a
        full paragraph; without sentence truncation the index duplicates the
        whole page body (Karpathy's index spec is link + one-line summary).
        """
        content = (node.content or "").strip()
        if not content:
            return ""
        first_line = content.split("\n")[0].strip()
        text = WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), first_line)
        # ponytail: heuristic sentence split — a boundary needs ". " +
        # capitalized/paren continuation, and pseudo-boundaries inside
        # identifiers like "Rec. P.863" are skipped when the next word itself
        # contains a period. Swap for a tokenizer if this ever misfires.
        desc = text
        for m in re.finditer(r"^(.+?[.!?])\s+(?=[A-Z(])", text):
            next_word = text[m.end() :].split(" ", 1)[0]
            if "." in next_word.rstrip(".,;:"):
                continue
            desc = m.group(1)
            break
        else:
            tail = re.match(r"^(.+?[.!?])\s*$", text)
            desc = tail.group(1) if tail else text
        if len(desc) > 200:
            desc = desc[:200].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        return desc

    @staticmethod
    def okf_frontmatter(
        node: WikiNode,
        compiled_at: str,
        frontmatter_defaults: dict[str, Any] | None = None,
        *,
        sources: list[OkfSource] | None = None,
        verified: list[TrustEvent] | None = None,
    ) -> dict[str, Any]:
        """Build the canonical OKF v0.2 frontmatter for a wiki node.

        The result is OKF §11-conformant (parseable YAML, non-empty ``type``)
        and carries KNOX extensions permitted by OKF §4.1. Config
        ``frontmatter_defaults`` and per-node ``metadata['frontmatter']``
        overrides are merged in.

        OKF v0.2 fields emitted:
        - ``type``, ``title``, ``description``, ``tags``, ``status`` (§4.1, §5.4)
        - ``generated: {by, at}`` — supersedes the v0.1 ``timestamp`` (§5.2, §13.1)
        - ``verified`` — trust events, only when ``verified`` is provided (§5.2)
        - ``sources`` — structured provenance objects (§5.1), only when
          ``sources`` is provided; bare filename provenance is NOT emitted
          here under the reserved ``sources`` key (it would collide with the
          v0.2 object shape) — see the ``source_files`` extension below.

        KNOX extensions (OKF §4.1 permits additional keys):
        - ``kind`` — original KNOX node_type
        - ``source_files`` — provenance filename stems (distinct from OKF
          ``sources`` to avoid shape collision; freed ``tags`` from this overload)
        """
        provenance = node.metadata.get(MetadataField.PROVENANCE) or []
        source_files = sorted({Path(str(p)).stem for p in provenance if p}) if provenance else []
        node_tags = node.metadata.get(MetadataField.TAGS) or []

        frontmatter: dict[str, Any] = dict(frontmatter_defaults or {})
        # Per-node frontmatter overrides config defaults.
        frontmatter.update(node.metadata.get("frontmatter", {}))

        # OKF v0.2 consolidation: the node ``frontmatter`` override is a legacy
        # escape hatch and may carry v0.1 bare-string ``sources`` (filename
        # stems) or ``tags`` overloaded with provenance. Drop both so the only
        # ``sources`` that survives is the structured §5.1 list passed in below,
        # and ``tags`` stays topic-only (provenance stems live in ``source_files``).
        frontmatter.pop("sources", None)
        frontmatter.pop("compiled_at", None)
        frontmatter.pop("timestamp", None)

        # OKF-required fields (always present, spec-compliant).
        frontmatter["type"] = frontmatter.get("type") or VaultWriter.okf_type(node.node_type)
        frontmatter.setdefault("title", node.title or "Untitled")
        description = VaultWriter.okf_description(node)
        if description:
            frontmatter.setdefault("description", description)
        # OKF lifecycle (§5.4): default stable; absence ⇒ stable per spec.
        frontmatter.setdefault("status", "stable")
        # OKF tags (§4.1): cross-cutting topic tags from compilation.
        # Source-filename provenance no longer overloads ``tags``; it lives in
        # the ``source_files`` extension to match OKF v0.2 semantics.
        if node_tags:
            merged_tags = sorted(set(frontmatter.get("tags", [])) | {str(t) for t in node_tags})  # nocheck: OKF frontmatter key, not wiki metadata
            frontmatter["tags"] = merged_tags  # nocheck: OKF frontmatter key, not wiki metadata
        node_attrs = node.metadata.get(MetadataField.ATTRIBUTES) or {}
        if node_attrs and "attributes" not in frontmatter:
            serialized: dict[str, str | list[str]] = {}
            for k, v in node_attrs.items():
                serialized[k] = v[0] if len(v) == 1 else v
            frontmatter["attributes"] = serialized

        # OKF trust (§5.2): ``generated`` records how/when content was produced.
        # Supersedes the v0.1 ``timestamp`` field (§13.1).
        frontmatter["generated"] = {"by": _GENERATED_BY, "at": compiled_at}
        # ``verified`` only when trust events are known (absence ⇒ unverified tier).
        if verified:
            frontmatter["verified"] = verified

        # OKF structured provenance (§5.1): list of objects with at least
        # ``resource``. Only emitted when callers supply real structured sources
        # (from the ReferenceStore); never populated with bare filename strings.
        if sources:
            frontmatter.setdefault("sources", sources)

        # KNOX extensions (OKF §4.1 permits additional keys).
        frontmatter["kind"] = node.metadata.get("frontmatter", {}).get("kind", node.node_type or "page")
        if source_files:
            frontmatter["source_files"] = source_files

        return frontmatter

    @staticmethod
    def node_quality(verification: dict[str, Any], node_id: str) -> dict[str, Any]:
        """Aggregate verification claims for *node_id* into a stats dict.

        Returns ``{}`` when no claim was verified for the node.
        """
        claims = [c for c in verification.get("claims", []) if isinstance(c, dict) and c.get("node_id") == node_id]
        if not claims:
            return {}
        supported = sum(1 for c in claims if c.get("verdict") == "supported")
        confidences = [float(c.get("confidence", 0.0)) for c in claims]
        return {
            "claims": len(claims),
            "supported": supported,
            "quality_score": round(sum(confidences) / len(confidences), 2),
        }

    def _write_vault(self, docs: list[Document], wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write each document as a separate ``.md`` file, plus index/log pages."""
        # Clear stale compile output so the on-disk vault exactly reflects the
        # current CompiledWiki. Root ``*.md`` plus generated subfolders are
        # removed; user-authored subfolders (``_distilled/``, ``write/``) stay.
        # ``log.md`` is append-only history (Karpathy) and survives recompiles.
        for stale in output_dir.glob("*.md"):
            if stale.name != LOG_FILE:
                stale.unlink()
        for folder in set(self._KIND_FOLDERS.values()):
            sub = output_dir / folder
            if sub.is_dir():
                for stale in sub.glob("*.md"):
                    stale.unlink()

        # slug/title -> bundle-relative path mapping for link rendering.
        # Paths are relative (no leading slash) so links stay portable.
        slug_targets: dict[str, str] = {}
        for doc in docs:
            folder = self._folder_for_kind(doc.node_type)
            path = f"{folder}/{doc.page_slug or 'untitled'}.md" if folder else f"{doc.page_slug or 'untitled'}.md"
            slug_targets[doc.page_slug] = path
            slug_targets[doc.title] = path

        node_map = {doc.page_slug: doc.title for doc in docs if doc.page_slug}

        related_edges = self._build_related_edges(wiki.edges, node_map)

        # Load reference store once for citation link rewriting.
        ref_store: ReferenceStore | None = None
        if self._references_file and self._references_file.exists():
            ref_store = ReferenceStore(self._references_file)

        for doc in docs:
            folder = self._folder_for_kind(doc.node_type)
            target_dir = output_dir / folder if folder else output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{doc.page_slug or 'untitled'}.md"

            # Rewrite in-text citation markers to reference links (shared).
            body = doc.body
            if ref_store is not None:
                # Determine the source file from provenance metadata.
                source_files = doc.frontmatter.get("source_files") or []
                if source_files:
                    # Try each source stem against the store's keys.
                    for source_key in self._candidate_source_keys(source_files, ref_store):
                        rewritten = rewrite_source_references(body, source_key, ref_store)
                        if rewritten != body:
                            body = rewritten
                            break

            rendered = self._render_document(doc, body, related_edges.get(doc.page_slug, []), wiki, slug_targets)
            (target_dir / filename).write_text(rendered, encoding="utf-8")

        # References page is always generated — an empty bibliography is
        # information too (signals reference extraction found nothing).
        self._write_references_page(ref_store, output_dir, wiki)
        # Per-source summary pages (markdown-only invariant: per-source asset
        # inventory + citation mapping live in the wiki, not only in .knox/).
        self._write_source_summaries(wiki, output_dir)
        # Verification summary (WiCER) — per-page claim table from
        # .knox/verification.json; stays internal when no pass ran.
        self._write_quality_page(wiki, output_dir)

        self._write_index(docs, output_dir, wiki)
        self._write_log(output_dir, wiki)
        return output_dir / INDEX_FILE

    def _write_source_summaries(self, wiki: CompiledWiki, output_dir: Path) -> None:
        """Write ``summaries/<slug>.md`` — one page per ingested source.

        Backlinks to concepts the source feeds, asset inventory (ids +
        one-line summaries from ``assets.json``), citations extracted from
        this source. Cache-only reads; no LLM calls. Skipped without
        ``base_dir`` (non-vault writers).
        """
        if self._base_dir is None:
            return
        vp = VaultPaths(self._base_dir)
        sources_dir = vp.sources_dir
        if not sources_dir.is_dir():
            return
        source_keys = sorted(
            f"{DEFAULT_SOURCES_DIR_NAME}/{p.relative_to(sources_dir).as_posix()}"
            for p in sources_dir.rglob("*")
            if p.is_file() and not is_protected_source(p, sources_dir)
        )
        if not source_keys:
            return

        ref_store: ReferenceStore | None = None
        if self._references_file and self._references_file.exists():
            ref_store = ReferenceStore(self._references_file)
        asset_store = AssetStore(vp.assets_file) if vp.assets_file.exists() else None

        # Backlink index: stem -> concept titles fed by this source.
        feeds: dict[str, list[str]] = {}
        for node in wiki.nodes:
            for stem in node.metadata.get(MetadataField.PROVENANCE) or []:
                feeds.setdefault(Path(str(stem)).stem, []).append(node.title or node.id)

        # LLM extraction summary text per source (keyed by vault-relative
        # path), so each summary page carries the semantic extract instead of
        # frontmatter + asset counts alone. Skipped when unavailable (e.g.
        # non-extractive caches).
        summary_by_key = {
            str(s.get("source", "")): str(s.get("summary", "")).strip()
            for s in wiki.metadata.get("extraction_stats") or []
            if str(s.get("summary", "")).strip()
        }

        summaries_dir = output_dir / SUMMARIES_SUBDIR
        summaries_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.stem for p in summaries_dir.glob("*.md")}
        sources_prefix = f"{DEFAULT_SOURCES_DIR_NAME}/"

        def _source_name(key: str) -> str:
            return key[len(sources_prefix) :] if key.startswith(sources_prefix) else key

        # Disambiguate same-stem collisions (e.g. ``foo.doc`` vs ``foo.docx``
        # both slug to ``foo``) by appending the extension so each source gets
        # its own page instead of silently overwriting the previous one.
        slug_assignments: dict[str, str] = {}
        by_base: dict[str, list[str]] = {}
        for key in source_keys:
            by_base.setdefault(attachment_slug(_source_name(key)), []).append(key)
        for base, keys in by_base.items():
            if len(keys) == 1:
                slug_assignments[keys[0]] = base
                continue
            used: set[str] = set()
            for key in keys:
                ext = Path(_source_name(key)).suffix.lstrip(".").lower()
                slug = f"{base}-{ext}" if ext else base
                n = 2
                while slug in used:
                    slug = f"{base}-{ext}-{n}" if ext else f"{base}-{n}"
                    n += 1
                used.add(slug)
                slug_assignments[key] = slug

        for key in source_keys:
            # Slug from the source name without the sources/ prefix —
            # consistent with the attachments flow ("report", not "sources_report").
            _source_name(key)
            slug = slug_assignments[key]
            stem = Path(key).stem
            # Display title: collapse underscore runs so raw filenames like
            # ``ITU-T_P_863__03_2018_`` read as text; the slug stays stable.
            display = re.sub(r"[_\s]+", " ", stem).strip(" .-")
            title = f"Source: {display}"
            lines = [
                okf_frontmatter_yaml(
                    {
                        "type": "Source",
                        "title": title,
                        "description": f"Per-source overview of {key}",
                        "generated": {"by": _GENERATED_BY, "at": iso_local()},
                        "status": "stable",
                        "kind": "source",
                        "source_files": [key],
                    }
                ).rstrip(),
                "",
                f"# {title}",
                "",
                f"**File:** `{key}`",
                "",
            ]

            summary = summary_by_key.get(key)
            if summary:
                lines.append("## Summary\n")
                lines.append(summary)
                lines.append("")

            # Backlinks: concepts this source feeds.
            fed = sorted(set(feeds.get(stem, [])))
            if fed:
                lines.append("## Feeds concepts\n")
                lines.extend(f"- {t}" for t in fed)
                lines.append("")

            # Tables/equations render inline on concept pages; ids live in
            # .knox/assets.json and mean nothing to a wiki reader.
            if asset_store is not None:
                _asset_lines(lines, [rec for rec in asset_store if Path(rec.source).stem == stem])

            # Citations extracted from this source.
            if ref_store is not None:
                refs = ref_store.get_source(key) or ref_store.get_source(stem) or []
                if refs:
                    lines.append("## References extracted\n")
                    lines.extend(f"- {r.short_label()} — {r.title}" for r in refs)
                    lines.append("")

            # Link the full-source export when one exists (on-demand only —
            # exports are reference dumps, never generated by compile).
            export_file = self._base_dir / EXPORTED_DIR_NAME / f"{stem}.md"
            if export_file.is_file():
                lines.append("## Full export\n")
                lines.append(f"- [Full source as markdown](../../{EXPORTED_DIR_NAME}/{export_file.name})")
                lines.append("")

            existing.discard(slug)
            (summaries_dir / f"{slug}.md").write_text("\n".join(lines), encoding="utf-8")

        # Prune stale summaries (sources removed / reorganized).
        for stale_slug in existing:
            (summaries_dir / f"{stale_slug}.md").unlink(missing_ok=True)

    def _load_verification(self) -> dict[str, Any] | None:
        """Read ``.knox/verification.json`` if present (per-claim records)."""
        if self._base_dir is None:
            return None
        vf = VaultPaths(self._base_dir).verification_file
        if not vf.is_file():
            return None
        try:
            data = json.loads(vf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Verification report corrupt — skipping quality page: %s", vf)
            return None
        return data if isinstance(data, dict) else None

    def _write_quality_page(self, wiki: CompiledWiki, output_dir: Path) -> None:
        """Write ``quality.md`` — WiCER verification summary (per-page table)."""
        verification = self._load_verification()
        if verification is None:
            return

        # Node title lookup for readable table rows.
        titles = {n.id: (n.title or n.id) for n in wiki.nodes}
        per_node: dict[str, dict[str, Any]] = {}
        for claim in verification.get("claims", []):
            if isinstance(claim, dict) and claim.get("node_id"):
                nid = str(claim["node_id"])
                per_node.setdefault(nid, {"claims": 0, "supported": 0, "conf": []})
                per_node[nid]["claims"] += 1
                if claim.get("verdict") == "supported":
                    per_node[nid]["supported"] += 1
                per_node[nid]["conf"].append(float(claim.get("confidence", 0.0)))
        if not per_node:
            return

        b = PageBuilder()
        b.heading(1, "Quality")
        b.paragraph(
            "Claims checked: ",
            strong(str(verification.get("total", 0))),
            " — ",
            "supported: ",
            strong(str(verification.get("verified", 0))),
            " — ",
            "unsupported: ",
            strong(str(verification.get("unsupported", 0))),
            " — ",
            "avg confidence: ",
            strong(f"{verification.get('average_confidence', 0.0):.2f}"),
        )
        b.table(
            headers=["Page", "Claims", "Supported", "Score"],
            rows=[
                [
                    titles.get(nid, nid),
                    str(stats["claims"]),
                    str(stats["supported"]),
                    f"{sum(stats['conf']) / len(stats['conf']):.2f}",
                ]
                for nid, stats in sorted(per_node.items(), key=lambda kv: kv[1]["supported"] / kv[1]["claims"])
            ],
            aligns=[None, "right", "right", "right"],
        )
        (output_dir / QUALITY_FILE).write_text(
            okf_frontmatter_yaml(
                {
                    "type": "Quality",
                    "title": "Quality",
                    "description": "Claim verification summary (WiCER pass)",
                    "generated": {"by": _GENERATED_BY, "at": iso_local()},
                    "status": "stable",
                    "kind": "quality",
                }
            ).rstrip()
            + "\n\n"
            + b.render(),
            encoding="utf-8",
        )

    # --- Format-specific hooks (overridden by subclasses) -----------------

    @abstractmethod
    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]: ...

    @abstractmethod
    def _render_document(
        self,
        doc: Document,
        body: str,
        related: list[tuple[str, str, float | None]],
        wiki: CompiledWiki,
        slug_targets: dict[str, str],
    ) -> str: ...

    def _write_references_page(
        self,
        store: ReferenceStore | None,
        output_dir: Path,
        wiki: CompiledWiki,
    ) -> None:
        """Write ``references.md`` — always, even when empty (§4.1).

        Two sections: the corpus sources themselves (linked to their summary
        pages) and bibliographic entries extracted from the sources'
        reference sections. The latter are mostly works NOT in the corpus —
        reading that section is the gap list for sources worth adding.
        """
        all_refs = sorted(store.all_references(), key=lambda r: r.short_label().lower()) if store else []

        # Corpus sources: same enumeration as _write_source_summaries.
        corpus: list[tuple[str, str]] = []  # (display key, summary slug or "")
        summaries_dir = output_dir / SUMMARIES_SUBDIR
        existing_slugs = {p.stem for p in summaries_dir.glob("*.md")} if summaries_dir.is_dir() else set()
        vp = VaultPaths(self._base_dir) if self._base_dir else None
        if vp and vp.sources_dir.is_dir():
            sources_prefix = f"{DEFAULT_SOURCES_DIR_NAME}/"
            for p in sorted(vp.sources_dir.rglob("*")):
                if not p.is_file() or is_protected_source(p, vp.sources_dir):
                    continue
                key = f"{DEFAULT_SOURCES_DIR_NAME}/{p.relative_to(vp.sources_dir).as_posix()}"
                name = key[len(sources_prefix) :]
                slug = attachment_slug(name)
                corpus.append((key, slug if slug in existing_slugs else ""))
        else:
            # No vault context — derive from node provenance.
            seen: set[str] = set()
            for node in wiki.nodes:
                for stem in node.metadata.get(MetadataField.PROVENANCE) or []:
                    key = str(stem)
                    if key not in seen:
                        seen.add(key)
                        corpus.append((key, attachment_slug(key)))

        fm = okf_frontmatter_yaml(
            {
                "type": "Reference",
                "title": "References",
                "description": "Corpus sources and bibliographic entries cited by them",
                "generated": {"by": _GENERATED_BY, "at": iso_local()},
                "status": "stable",
                "kind": "reference",
            }
        ).rstrip()

        b = PageBuilder()
        b.heading(1, "References")
        b.heading(2, "Corpus sources")
        if corpus:
            b.bullet_list([link(f"{SUMMARIES_SUBDIR}/{slug}.md", key) if slug and slug in existing_slugs else key for key, slug in sorted(corpus)])
        else:
            b.bullet_list(["_(no corpus sources)_"])
        b.heading(2, "Cited works not in corpus")
        b.paragraph("Bibliographic entries extracted from the sources' reference sections.")
        b.paragraph("Candidates for additional sources to ingest.")

        parts = [fm, "", b.render()]
        if all_refs:
            parts.extend(format_bibliography_entry(ref) for ref in all_refs)
        else:
            parts.append("_(none extracted yet)_")

        (output_dir / "references.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")

    @abstractmethod
    def _write_index(self, docs: list[Document], output_dir: Path, wiki: CompiledWiki) -> None: ...

    @abstractmethod
    def _write_log(self, output_dir: Path, wiki: CompiledWiki) -> None:
        """Write the bundle log page (no-op for the wikilink variant)."""

    def _knox_relative(self, raw: str) -> Path | None:
        """Return *raw* as a base_dir-relative path if it lives under ``.knox/``."""
        if not raw or raw.startswith("data:") or self._base_dir is None:
            return None
        p = Path(raw)
        try:
            rel = p.relative_to(self._base_dir) if p.is_absolute() else p
        except ValueError:
            return None
        return rel if rel.parts and rel.parts[0] == KNOX_DIR else None

    @classmethod
    def _folder_for_kind(cls, node_type: str) -> str:
        """Return the subfolder for a page kind, or '' to stay at the root."""
        return cls._KIND_FOLDERS.get(node_type, "")

    @staticmethod
    def _build_related_edges(
        edges: list[GraphEdge],
        node_map: dict[str, str],
    ) -> dict[str, list[tuple[str, str, float | None]]]:
        """Build a map of page slug -> sorted list of related page entries.

        Each entry is ``(target_title, relation_type, confidence)``. Typed
        relation edges carry their type and confidence; plain entity-reference
        edges are rendered without an annotation.
        """
        related: dict[str, dict[str, tuple[str, float | None]]] = {}
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "")
            confidence = edge.get("confidence")
            confidence = float(confidence) if isinstance(confidence, (int, float)) else None

            for src, tgt in ((source, target), (target, source)):
                if not src or not tgt or src == tgt:
                    continue
                if src not in node_map or tgt not in node_map:
                    continue
                target_title = node_map[tgt]
                current = related.setdefault(src, {})
                # Prefer the highest-confidence annotation per target.
                prev = current.get(target_title)
                if prev is None or (confidence is not None and (prev[1] is None or confidence > prev[1])):
                    current[target_title] = (rel_type, confidence)

        result: dict[str, list[tuple[str, str, float | None]]] = {}
        for slug, targets in related.items():
            sorted_targets = sorted(
                targets.items(),
                key=lambda item: (item[1][1] if item[1][1] is not None else -1.0, item[0]),
                reverse=True,
            )
            result[slug] = [(title, rel_type, confidence) for title, (rel_type, confidence) in sorted_targets]
        return result

    @staticmethod
    def _candidate_source_keys(sources: list[str], store: ReferenceStore) -> list[str]:
        """Yield source keys likely to match store entries for the given sources.

        The store is keyed by full source paths (``sources/paper.pdf``); the
        frontmatter ``sources`` list contains stems (``paper``). We try
        multiple candidate formats against the store's actual keys.
        """
        store_keys = set(store.to_dict().keys())
        candidates: list[str] = []
        for src in sources:
            for candidate in (
                src,
                f"sources/{src}",
                f"sources/{src}.pdf",
                f"sources/{src}.md",
                f"sources/{src}.txt",
            ):
                if candidate in store_keys:
                    candidates.append(candidate)
            # Also check for partial matches (stem anywhere in key)
            for key in store_keys:
                if src and src in Path(key).stem:
                    candidates.append(key)
        return candidates

    @staticmethod
    def _embed_images(body: str, images: list[ImageRef]) -> str:
        """Embed image references into the document body.

        Each image dict should have:
        - ``path``: file path or base64 data URI
        - ``caption``: optional VLM-generated caption (used as alt text)
        - ``element_type``: "image" (for future extensibility)

        Images are appended after the body text with proper Markdown syntax.
        """
        lines: list[str] = [body.rstrip(), ""] if body.strip() else []

        for img in images:
            img_file = img.get("path", "")
            caption = img.get("caption", "")

            if not img_file:
                continue

            alt_text = caption or "image"
            lines.append(f"![{alt_text}]({img_file})")
            if caption:
                lines.append(f"*{caption}*")
            lines.append("")

        return "\n".join(lines)


def okf_frontmatter_yaml(frontmatter: dict[str, Any]) -> str:
    """Render frontmatter as YAML without anchors/aliases."""
    yaml_str = yaml.dump(
        frontmatter,
        Dumper=_NoAliasDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    ).strip()
    return f"---\n{yaml_str}\n---\n\n"


def _asset_lines(lines: list[str], items: list[AssetRecord]) -> None:
    """Append the per-source asset inventory grouped by kind."""
    if not items:
        return
    lines.append("## Assets\n")
    for kind in sorted({rec.kind for rec in items}, key=str):
        kind_items = [r for r in items if r.kind == kind]
        lines.append(f"- **{kind.name.lower().capitalize()}s:** {len(kind_items)}")
        if kind == AssetKind.FIGURE:
            lines.extend(f"  - {getattr(r.data, 'caption', '') or r.summary or r.id}".rstrip() for r in kind_items)
    lines.append("")
