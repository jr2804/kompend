"""Generated surface pages — the non-concept pages the bundle always/optionally emits.

Concept pages are rendered per node by the writer's format hooks. The surfaces
here are the bundle-level pages that exist once per vault:

- ``summaries/<slug>.md`` — one page per ingested source (asset inventory,
  backlinks, extracted citations).
- ``references.md`` — corpus sources plus the cited works that are not in the
  corpus.
- ``quality.md`` — the WiCER verification summary, written only when a
  verification pass ran.

The mixin keeps the documented ``VaultWriter._load_verification`` entry point
while moving the implementations out of the vault algorithm module.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from kompend.core.models import CompiledWiki, ExtractionStats, MetadataField
from kompend.core.models.assets import AssetRecord
from kompend.core.models.enums import AssetKind, NodeType, OkfField
from kompend.core.models.metadata import meta_dict_list, meta_str_list
from kompend.core.page_builder import PageBuilder, link, strong
from kompend.core.paths import (
    DEFAULT_SOURCES_DIR_NAME,
    EXPORTED_DIR_NAME,
    QUALITY_FILE,
    REFERENCES_MD,
    SUMMARIES_SUBDIR,
    VaultPaths,
    attachment_slug,
    is_protected_source,
    safe_stem,
)
from kompend.core.pipeline.assets import AssetStore
from kompend.core.pipeline.references import ReferenceStore
from kompend.core.store import open_store
from kompend.core.store.schema import VERIFICATION_TABLE
from kompend.core.text.pipes import escape_prose_pipes
from kompend.core.utils.time import iso_local
from kompend.logging import get_logger
from kompend.output.frontmatter import okf_frontmatter_yaml
from kompend.output.page import DEFAULT_STATUS, GENERATED_BY
from kompend.output.reference_links import format_bibliography_entry

__all__ = ["GeneratedSurfaceMixin", "source_keys"]

logger = get_logger(__name__)


def source_keys(sources_dir: Path) -> list[str]:
    """Vault-relative ``sources/<path>`` keys for every ingestable source file."""
    return sorted(
        f"{DEFAULT_SOURCES_DIR_NAME}/{p.relative_to(sources_dir).as_posix()}"
        for p in sources_dir.rglob("*")
        if p.is_file() and not is_protected_source(p, sources_dir)
    )


class GeneratedSurfaceMixin:
    """Bundle-level surface pages, written once per vault."""

    _base_dir: Path | None
    _references_file: Path | None
    _assets_file: Path | None

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
        keys = source_keys(sources_dir)
        if not keys:
            return

        ref_store: ReferenceStore | None = None
        if self._references_file is not None:
            ref_store = ReferenceStore(self._references_file)
        asset_store: AssetStore | None = AssetStore(vp.assets_file)

        # Backlink index: stem -> concept titles fed by this source.
        feeds: dict[str, list[str]] = {}
        for node in wiki.nodes:
            for stem in meta_str_list(node.metadata.get(MetadataField.PROVENANCE)):
                feeds.setdefault(Path(stem).stem, []).append(node.title or node.id)

        # LLM extraction summary text per source (keyed by vault-relative
        # path), so each summary page carries the semantic extract instead of
        # frontmatter + asset counts alone. Skipped when unavailable (e.g.
        # non-extractive caches).
        summary_by_key = {
            stats.source: stats.summary.strip()
            for stats in (ExtractionStats.from_mapping(item) for item in meta_dict_list(wiki.metadata.get("extraction_stats")))
            if stats.summary.strip()
        }

        summaries_dir = output_dir / SUMMARIES_SUBDIR
        summaries_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.stem for p in summaries_dir.glob("*.md")}

        # Disambiguate same-stem collisions (e.g. ``foo.doc`` vs ``foo.docx``
        # both slug to ``foo``) by appending the extension so each source gets
        # its own page instead of silently overwriting the previous one.
        slug_assignments: dict[str, str] = {}
        by_base: dict[str, list[str]] = {}
        for key in keys:
            by_base.setdefault(attachment_slug(_source_name(key)), []).append(key)
        for base, group in by_base.items():
            if len(group) == 1:
                slug_assignments[group[0]] = base
                continue
            used: set[str] = set()
            for key in group:
                ext = Path(_source_name(key)).suffix.lstrip(".").lower()
                slug = f"{base}-{ext}" if ext else base
                n = 2
                while slug in used:
                    slug = f"{base}-{ext}-{n}" if ext else f"{base}-{n}"
                    n += 1
                used.add(slug)
                slug_assignments[key] = slug

        for key in keys:
            # Slug from the source name without the sources/ prefix —
            # consistent with the attachments flow ("report", not "sources_report").
            slug = slug_assignments[key]
            stem = safe_stem(key)
            # Display title: collapse underscore runs so raw filenames like
            # ``ITU-T_P_863__03_2018_`` read as text; the slug stays stable.
            display = re.sub(r"[_\s]+", " ", stem).strip(" .-")
            title = f"Source: {display}"
            lines = [
                okf_frontmatter_yaml(
                    {
                        OkfField.TYPE: NodeType.SOURCE.okf_type,
                        OkfField.TITLE: title,
                        OkfField.DESCRIPTION: f"Per-source overview of {key}",
                        OkfField.GENERATED: {"by": GENERATED_BY, "at": iso_local()},
                        OkfField.STATUS: DEFAULT_STATUS,
                        OkfField.KIND: NodeType.SOURCE.value,
                        OkfField.SOURCE_FILES: [key],
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
                lines.append(escape_prose_pipes(summary))
                lines.append("")

            # Backlinks: concepts this source feeds.
            fed = sorted(set(feeds.get(stem, [])))
            if fed:
                lines.append("## Feeds concepts\n")
                lines.extend(f"- {t}" for t in fed)
                lines.append("")

            # Tables/equations render inline on concept pages; ids live in
            # .kompend/assets.json and mean nothing to a wiki reader.
            if asset_store is not None:
                _asset_lines(lines, [rec for rec in asset_store if safe_stem(rec.source) == stem])

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
        """Read the latest ``verification`` row (per-claim records) if present."""
        if self._base_dir is None:
            return None
        db_path = VaultPaths(self._base_dir).kompend_db_file
        if not db_path.is_file():
            return None
        try:
            with open_store(db_path) as conn:
                query = f"SELECT total, verified, unsupported, average_confidence, claims FROM {VERIFICATION_TABLE} ORDER BY id DESC LIMIT 1"  # noqa: S608
                row = conn.execute(query).fetchone()
        except sqlite3.Error as exc:
            logger.warning("Verification store unreadable — skipping quality page: %s", exc)
            return None
        if row is None:
            return None
        return {
            "total": row["total"],
            "verified": row["verified"],
            "unsupported": row["unsupported"],
            "average_confidence": row["average_confidence"],
            "claims": json.loads(row["claims"]),
        }

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
                    OkfField.TYPE: NodeType.QUALITY.okf_type,
                    OkfField.TITLE: "Quality",
                    OkfField.DESCRIPTION: "Claim verification summary (WiCER pass)",
                    OkfField.GENERATED: {"by": GENERATED_BY, "at": iso_local()},
                    OkfField.STATUS: DEFAULT_STATUS,
                    OkfField.KIND: NodeType.QUALITY.value,
                }
            ).rstrip()
            + "\n\n"
            + b.render(),
            encoding="utf-8",
        )

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
            for key in source_keys(vp.sources_dir):
                name = _source_name(key)
                slug = attachment_slug(name)
                corpus.append((key, slug if slug in existing_slugs else ""))
        else:
            # No vault context — derive from node provenance.
            seen: set[str] = set()
            for node in wiki.nodes:
                for stem in meta_str_list(node.metadata.get(MetadataField.PROVENANCE)):
                    key = stem
                    if key not in seen:
                        seen.add(key)
                        corpus.append((key, attachment_slug(key)))

        fm = okf_frontmatter_yaml(
            {
                OkfField.TYPE: NodeType.REFERENCE.okf_type,
                OkfField.TITLE: "References",
                OkfField.DESCRIPTION: "Corpus sources and bibliographic entries cited by them",
                OkfField.GENERATED: {"by": GENERATED_BY, "at": iso_local()},
                OkfField.STATUS: DEFAULT_STATUS,
                OkfField.KIND: NodeType.REFERENCE.value,
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

        (output_dir / REFERENCES_MD).write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def _source_name(key: str) -> str:
    """A ``sources/<path>`` key without its ``sources/`` prefix."""
    prefix = f"{DEFAULT_SOURCES_DIR_NAME}/"
    return key[len(prefix) :] if key.startswith(prefix) else key


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
