"""Structural lint rules — deterministic, zero LLM cost.

Each rule is a class that satisfies the :class:`LintRule` protocol.
Rules inspect a :class:`CompiledWiki` (and optionally the sources
directory on disk) and return :class:`LintResult` findings.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from knox.compilation.concept_keys import normalize_concept_key
from knox.core.models import CompiledWiki, MetadataField
from knox.core.models.attachments import AttachmentManifest
from knox.core.paths import KIND_FOLDERS, KNOX_DIR, REFERENCES_FILE, VaultPaths, attachment_slugs_for_source, is_protected_source
from knox.core.pipeline.references import ReferenceStore, consolidate_references
from knox.core.pipeline.state import TierStateStore
from knox.core.text.matching import best_match
from knox.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from knox.logging import get_logger

#: Max chars of a reference title used for display labels in lint findings.
_REF_LABEL_TITLE_MAX = 40

#: Leading argv entries of the rumdl command shown in a failure warning —
#: enough to identify the invocation without echoing every path.
_CMD_PREVIEW_ARGS = 4

# ── Helpers ──────────────────────────────────────────────────────────────

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_HEADING_RE = re.compile(r"^(#{1,6})\s", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

#: Image file extensions — ``[[img.png]]`` is an embed-syntax error, not a
#: broken wikilink; :class:`ImageEmbedRule` handles those.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".bmp",
        ".tiff",
    }
)

#: ``[[img.png]]`` without the ``!`` embed prefix.
_ATTACHMENT_LINK_RE = re.compile(r"\]\((?P<target>[^)]*attachments/[^)]+)\)")
_IMAGE_EMBED_RE = re.compile(
    r"(?<!\!)\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff))\]\]",
    re.IGNORECASE,
)


#: Minimum token-Jaccard score to flag a broken wikilink as an alias of an
#: existing page (vs. a genuinely missing topic). Conservative default:
#: 0.5 means at least half the target's tokens overlap with the candidate.

#: Threshold above which a broken wikilink is treated as a SHORT-FORM alias
#: (severity WARNING, the suggested alias is attached) instead of a hard
#: failure (severity ERROR). Lowered from the default best_match threshold
#: (0.85) because the p863 wiki roundup showed most "broken" wikilinks
#: were intentional short-form aliases emitted by small models — the lint
#: should surface these as guidance, not block the run.
_WIKILINK_ALIAS_WARN_THRESHOLD: float = 60.0

#: Lower detection bound — anything below this is treated as a genuine
#: missing topic (no alias suggestion, ERROR severity). Avoids noisy
#: ``alias_of=foo (similarity=9)`` markers when the target has no real
#: relationship to any page.
_WIKILINK_ALIAS_MIN_DETECT: float = 25.0

# ── Rule: knox-reference ───────────────────────────────────────────────
#: Markdown link or embed target pointing into the ``.knox/`` state dir
#: (``.knox/…``, ``./.knox/…``, ``../.knox/…``, ``/.knox/…``).
_KNOX_PATH_RE = re.compile(rf"\]\(((?:\.\./)*(?:\./)?/?{re.escape(KNOX_DIR)}/[^)\s]+)\)")

logger = get_logger(__name__)


# ── Rule 1: broken-wikilinks ──────────────────────────────────────────────


class BrokenWikilinksRule(BaseLintRule):
    """Detect ``[[wikilinks]]`` that point to non-existent pages."""

    name = "broken-wikilinks"
    description = "Wikilinks pointing to pages that do not exist"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        # Lookup index covers: page title, page slug, and acronym-stripped
        # short alias (e.g. ``[[P.863]]`` resolves to the page titled
        # ``POLQA (Perceptual Objective Listening Quality Analysis)``).
        alias_index, slug_to_id = _node_lookup_index(wiki)
        results: list[LintResult] = []
        for node in wiki.nodes:
            for target_title in _extract_wikilinks(node.content):
                if Path(target_title).suffix.lower() in _IMAGE_EXTENSIONS:
                    # Image embeds are handled by ImageEmbedRule — not page links.
                    continue
                # Source-file citations (sources/foo.pdf) are provenance, not page links.
                if target_title.startswith("sources/"):
                    continue
                # Alias syntax [[Page Title|display text]]: resolve the page ref.
                page_ref = target_title.split("|", 1)[0].strip()
                if page_ref.lower() not in alias_index:
                    target_norm = normalize_concept_key(page_ref)
                    alias_of: str | None = None
                    similarity_score: float = 0.0
                    if target_norm:
                        best = best_match(target_norm, alias_index, threshold=0.0)
                        if best and best[1] >= _WIKILINK_ALIAS_MIN_DETECT:
                            alias_of = alias_index[best[0]]
                            similarity_score = best[1]
                    details: dict[str, Any] = {"target": target_title}
                    if alias_of:
                        details["alias_of"] = alias_of
                        details["alias_similarity"] = round(similarity_score, 1)
                    severity = Severity.WARNING if similarity_score >= _WIKILINK_ALIAS_WARN_THRESHOLD else Severity.ERROR
                    message = (
                        f"Wikilink [[{target_title}]] in page '{node.title}' resolves to '{alias_of}' via short-form alias (similarity={similarity_score:.0f})"
                        if alias_of
                        else f"Broken wikilink [[{target_title}]] in page '{node.title}'"
                    )
                    results.append(
                        LintResult(
                            rule=self.name,
                            outcome=LintRuleOutcome.FAIL,
                            severity=severity,
                            message=message,
                            page_id=node.id,
                            details=details,
                        )
                    )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All wikilinks resolve"))
        return results

    def fix(self, wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        """Repair broken wikilinks in page content.

        Alias-resolvable links (the LLM used a short form of an existing
        page's title) are rewritten to real markdown links pointing at the
        resolved page — matching what the renderer's ``convert_wikilinks``
        produces. Unresolvable links are stripped to their text.
        """
        node_by_id = {n.id: n for n in wiki.nodes}
        page_edits: dict[str, list[tuple[str, str | None]]] = {}
        for r in results:
            if r.outcome != LintRuleOutcome.FAIL or not r.details.get("target"):
                continue
            page_edits.setdefault(r.page_id, []).append((r.details["target"], r.details.get("alias_of")))

        fixed = list(results)
        for node in wiki.nodes:
            edits = page_edits.get(node.id)
            if not edits:
                continue
            for target, alias_of in edits:
                inner = target
                if "|" in inner:
                    page_ref, alias_text = inner.split("|", 1)
                else:
                    alias_text = inner
                replacement: str
                target_node = node_by_id.get(alias_of or "")
                if alias_of and target_node is not None:
                    # Rewrite to the resolved page link (same path the
                    # renderer's convert_wikilinks produces).
                    folder = KIND_FOLDERS.get(target_node.node_type, "concepts")
                    replacement = f"[{alias_text.strip()}]({folder}/{alias_of}.md)"
                else:
                    # Genuinely missing target — keep the text, drop the brackets.
                    replacement = alias_text.strip()
                node.content = node.content.replace(f"[[{target}]]", replacement)
            for r in fixed:
                if r.page_id == node.id and r.outcome == LintRuleOutcome.FAIL:
                    r.fix_applied = True
        return fixed


# ── Rule 2: image-embed ───────────────────────────────────────────────────


class ImageEmbedRule(BaseLintRule):
    """Detect image wikilinks written without the ``!`` embed prefix."""

    name = "image-embed"
    description = "Image links that should use ![[...]] embed syntax"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        for node in wiki.nodes:
            for target in _IMAGE_EMBED_RE.findall(node.content or ""):
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Image link [[{target}]] in page '{node.title}' should be ![[{target}]]",
                        page_id=node.id,
                        details={"target": target},
                    )
                )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All image links use embed syntax"))
        return results

    def fix(self, wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        """Rewrite ``[[img.png]]`` → ``![[img.png]]`` in page content."""
        fixed = list(results)
        for node in wiki.nodes:
            content = node.content or ""
            if not _IMAGE_EMBED_RE.search(content):
                continue
            node.content = _IMAGE_EMBED_RE.sub(lambda m: f"![[{m.group(1)}]]", content)
            for r in fixed:
                if r.page_id == node.id and r.outcome == LintRuleOutcome.FAIL:
                    r.fix_applied = True
        return fixed


class KnoxReferenceRule(BaseLintRule):
    """Detect page links/embeds that resolve into ``.knox/``.

    Markdown-only invariant: ``.knox/`` is derived state and may be absent
    from a shared vault — a page referencing it breaks portability.
    """

    name = "knox-reference"
    description = "Links or embeds resolving into the .knox/ state directory"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        for node in wiki.nodes:
            content = node.content or ""
            for target in _KNOX_PATH_RE.findall(content):
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.ERROR,
                        message=(f"Page '{node.title}' references '{target}' inside {KNOX_DIR}/ — copy the file into attachments/ and link relatively"),
                        page_id=node.id,
                        details={"target": target},
                    )
                )
            for img in node.metadata.get("images") or []:
                path = str(img.get("path", "")) if isinstance(img, dict) else ""
                if path and KNOX_DIR in PurePosixPath(path).parts:
                    results.append(
                        LintResult(
                            rule=self.name,
                            outcome=LintRuleOutcome.FAIL,
                            severity=Severity.ERROR,
                            message=(
                                f"Page '{node.title}' embeds image '{path}' from inside {KNOX_DIR}/ — "
                                "the writer must copy it into attachments/ (portablize_images)"
                            ),
                            page_id=node.id,
                            details={"target": path},
                        )
                    )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message=f"No page references {KNOX_DIR}/"))
        return results


# ── Rule 3: missing-frontmatter ─────────────────────────────────────────


class MissingFrontmatterRule(BaseLintRule):
    """Detect pages missing required frontmatter fields.

    Required fields: ``title`` and ``date``.
    """

    name = "missing-frontmatter"
    description = "Pages missing required frontmatter fields"

    REQUIRED_FIELDS: tuple[str, ...] = ("title",)

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        for node in wiki.nodes:
            fm = node.metadata.get("frontmatter", {})
            if not isinstance(fm, dict):
                fm = {}
            missing = [f for f in self.REQUIRED_FIELDS if f not in fm]
            if missing:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Page '{node.title}' missing frontmatter: {', '.join(missing)}",
                        page_id=node.id,
                        details={"missing_fields": missing},
                    )
                )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All pages have frontmatter"))
        return results

    def fix(self, wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        """Add minimal frontmatter to pages that lack it."""
        fixed = list(results)
        for r in fixed:
            if r.outcome != LintRuleOutcome.FAIL:
                continue
            node = next((n for n in wiki.nodes if n.id == r.page_id), None)
            if node is None:
                continue
            fm = node.metadata.get("frontmatter", {})
            if not isinstance(fm, dict):
                fm = {}
            for field_name in r.details.get("missing_fields", []):
                if field_name == "title":
                    fm["title"] = node.title
            node.metadata["frontmatter"] = fm
            r.fix_applied = True
        return fixed


# ── Rule 3: orphan-page ─────────────────────────────────────────────────


class OrphanPageRule(BaseLintRule):
    """Detect pages with no connections to other pages.

    A page is considered linked if it has incoming ``[[wikilinks]]`` in other
    pages' content OR participates in any edge in ``wiki.edges``. Relationship
    edges (``entity_reference``, ``sibling``, ``related``, ``concept_reference``)
    are undirected connections — both endpoints are linked, not just the
    ``target``. Counting only ``target`` falsely orphaned every node that sorts
    alphabetically first and was stored as ``source``.
    """

    name = "orphan-page"
    description = "Pages with no connections to other pages"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        incoming_content = _incoming_link_counts(wiki)
        # Relationship edges connect both endpoints — count either role.
        linked_nodes: set[str] = set()
        for edge in wiki.edges:
            for endpoint in (edge.get("source"), edge.get("target")):
                if endpoint:
                    linked_nodes.add(endpoint)

        results: list[LintResult] = []
        for node in wiki.nodes:
            has_content_links = incoming_content.get(node.id, 0) > 0
            has_edge_links = node.id in linked_nodes
            if not has_content_links and not has_edge_links:
                # Skip single-page wikis — orphans are meaningless there
                if len(wiki.nodes) <= 1:
                    continue
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Orphan page '{node.title}' has no incoming links",
                        page_id=node.id,
                    )
                )
        if not results and len(wiki.nodes) > 1:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All pages are linked"))
        return results


# ── Rule 4: duplicate-title ─────────────────────────────────────────────


class DuplicateTitleRule(BaseLintRule):
    """Detect pages with duplicate titles (case-insensitive exact match)."""

    name = "duplicate-title"
    description = "Two or more pages with the exact same title (case-insensitive)"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        seen: dict[str, list[str]] = {}  # lower(title) -> [node_id, ...]
        for node in wiki.nodes:
            key = node.title.lower().strip()
            seen.setdefault(key, []).append(node.id)

        for title_key, node_ids in seen.items():
            if len(node_ids) > 1:
                for nid in node_ids:
                    results.append(
                        LintResult(
                            rule=self.name,
                            outcome=LintRuleOutcome.FAIL,
                            severity=Severity.ERROR,
                            message=f"Duplicate title '{title_key}' shared by {len(node_ids)} pages",
                            page_id=nid,
                            details={"duplicate_ids": node_ids},
                        )
                    )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No duplicate titles"))
        return results


# ── Rule 5: empty-page ──────────────────────────────────────────────────


class EmptyPageRule(BaseLintRule):
    """Detect pages with no body content."""

    name = "empty-page"
    description = "Pages with no body content"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        for node in wiki.nodes:
            content = node.content.strip() if node.content else ""
            if not content:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Empty page '{node.title}'",
                        page_id=node.id,
                    )
                )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No empty pages"))
        return results


# ── Rule 6: heading-skip ────────────────────────────────────────────────


class HeadingSkipRule(BaseLintRule):
    """Detect heading level jumps (e.g. h1 → h3 without h2)."""

    name = "heading-skip"
    description = "Heading level skips (h1 → h3 without h2)"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        for node in wiki.nodes:
            headings: list[int] = []
            for match in _HEADING_RE.finditer(node.content):
                headings.append(len(match.group(1)))
            if not headings:
                continue
            prev = headings[0]
            for level in headings[1:]:
                if level > prev + 1:
                    results.append(
                        LintResult(
                            rule=self.name,
                            outcome=LintRuleOutcome.FAIL,
                            severity=Severity.WARNING,
                            message=f"Heading skip h{prev} → h{level} in page '{node.title}'",
                            page_id=node.id,
                            details={"from": prev, "to": level},
                        )
                    )
                prev = level
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No heading skips"))
        return results


# ── Rule 7: stale-page ──────────────────────────────────────────────────


class StalePageRule(BaseLintRule):
    """Detect pages whose source files have been modified since last compile.

    Also checks the memory lifecycle store for superseded and decayed (stale)
    nodes when memory metadata is present in ``wiki.metadata``.
    """

    name = "stale-page"
    description = "Pages whose source files changed since last compile"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []

        # ── Source file existence check ──
        if sources_dir is not None:
            sources_root = Path(sources_dir)
            if sources_root.exists():
                for node in wiki.nodes:
                    source_file_str = node.metadata.get(MetadataField.SOURCE_FILE)
                    if not source_file_str:
                        continue
                    source_file = Path(source_file_str)
                    if not source_file.is_absolute():
                        # Try base_dir-relative first (sources/paper.pdf),
                        # then sources_dir-relative (paper.pdf)
                        for root in (sources_root.parent, Path(sources_dir)):
                            if (root / source_file).exists():
                                source_file = str(root / source_file)
                                break
                        else:
                            source_file = str(sources_root.parent / source_file)
                    if not Path(source_file).exists():
                        results.append(
                            LintResult(
                                rule=self.name,
                                outcome=LintRuleOutcome.FAIL,
                                severity=Severity.WARNING,
                                message=f"Source file not found for page '{node.title}': {source_file}",
                                page_id=node.id,
                                details={MetadataField.SOURCE_FILE: str(source_file)},
                            )
                        )

        # ── Memory lifecycle checks ──
        memory_meta = wiki.metadata.get("memory", {})
        superseded_ids = set(memory_meta.get("superseded", []))
        stale_ids = set(memory_meta.get("stale", []))

        for node in wiki.nodes:
            if node.id in superseded_ids:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Page '{node.title}' has been superseded by a newer node",
                        page_id=node.id,
                        details={"reason": "superseded"},
                    )
                )
            elif node.id in stale_ids:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.INFO,
                        message=f"Page '{node.title}' has decayed below retention threshold",
                        page_id=node.id,
                        details={"reason": "decayed"},
                    )
                )

        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All source files intact and memory fresh"))
        return results


# ── Rule 8: source-not-found ────────────────────────────────────────────


class UnreferencedAttachmentsRule(BaseLintRule):
    """Detect published media in a live source directory that no page references.

    The vault writer copies every referenced figure out of ``.knox/`` into
    ``attachments/<source-slug>/`` (markdown-only invariant). Nothing removes
    those copies when a page is rewritten to reference fewer images or when a
    page that referenced them is gone, so a live source directory keeps copies
    the wiki no longer shows. (A directory whose slug maps to no source at all
    is :class:`OrphanAttachmentsRule`'s territory — this rule skips those, so
    the two never report the same file.)

    A file is prunable only when it is byte-identical to an extracted image in
    ``.knox/cache/images/``, the copy source. That keeps the rule lossless: it
    deletes only what a later compile can recreate, and never a file a user
    placed there (whose bytes came from somewhere else — hand-added, or written
    by an earlier run whose cache entry is gone). Auto-adoption registers writer
    copies in the manifest too, so the manifest row is not consulted for the
    decision — only for tombstoning what gets deleted.
    """

    name = "unreferenced-attachments"
    description = "Published media in a live source directory that no page references"

    #: Deletes files on disk — same opt-in contract as
    #: :class:`OrphanAttachmentsRule`.
    destructive: ClassVar[bool] = True

    # Populated by check() so fix() does not need sources_dir threaded through.
    _last_sources_dir: str | None = None

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        self._last_sources_dir = sources_dir
        if sources_dir is None:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided")]

        base_dir = Path(sources_dir).parent
        vp = VaultPaths(base_dir)
        if not vp.attachments_dir.is_dir():
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No attachments directory")]

        stale = unreferenced_attachment_files(base_dir)
        if not stale:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No unreferenced attachments")]
        return [
            LintResult(
                rule=self.name,
                outcome=LintRuleOutcome.FAIL,
                severity=Severity.WARNING,
                message=(
                    f"Unreferenced attachment '{rel}' — no page links it and the manifest does not "
                    "register it (recompile after a page change, or delete it with --fix-destructive)"
                ),
                details={"path": rel},
            )
            for rel in stale
        ]

    def fix(self, wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:
        """Delete unreferenced, unregistered attachment files (destructive).

        Mirrors :meth:`OrphanAttachmentsRule.fix`: the lint runner's
        ``--fix-destructive`` opt-in is the only trigger, and a failure leaves
        the result FAIL with the error recorded. Empty slug directories are
        pruned afterwards.
        """
        sources_dir = self._last_sources_dir
        if not results or sources_dir is None:
            return list(results)

        base_dir = Path(sources_dir).parent
        vp = VaultPaths(base_dir)
        manifest = AttachmentManifest(vp.attachments_manifest_file)
        updated: list[LintResult] = []
        deleted = 0
        for r in results:
            if r.outcome != LintRuleOutcome.FAIL:
                updated.append(r)
                continue
            rel_path = (r.details or {}).get("path")
            if not rel_path:
                updated.append(r)
                continue
            abs_path = base_dir / rel_path
            try:
                if abs_path.is_file():
                    abs_path.unlink()
                    deleted += 1
                # Tombstone the manifest row (auto-adoption registered writer
                # copies too) so the manifest does not keep pointing at a file
                # that is gone — the row keeps its history, marked deleted.
                manifest.remove(rel_path)
                r.fix_applied = True
                r.message = f"Deleted unreferenced attachment '{rel_path}'"
            except OSError as exc:
                logger.warning("Could not delete unreferenced attachment %s: %s", abs_path, exc)
                r.message = f"Could not delete unreferenced attachment '{rel_path}': {exc}"
                r.details = {**(r.details or {}), "error": str(exc)}
            updated.append(r)

        if deleted:
            logger.info("Deleted %d unreferenced attachment file(s)", deleted)
            _prune_empty_dirs(vp.attachments_dir)
        return updated


class SourceNotFoundRule(BaseLintRule):
    """Detect pages that reference source files not present on disk."""

    name = "source-not-found"
    description = "Pages referencing source files that don't exist"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        if sources_dir is None:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided"))
            return results

        sources_root = Path(sources_dir)
        for node in wiki.nodes:
            source_chunks = node.source_chunks or []
            if not source_chunks:
                continue
            # source_chunks are chunk IDs, not file paths. Check if
            # metadata references source_file.
            source_file_str = node.metadata.get(MetadataField.SOURCE_FILE)
            if not source_file_str:
                continue
            source_file = Path(source_file_str)
            if not source_file.is_absolute():
                for root in (sources_root.parent, Path(sources_dir)):
                    if (root / source_file).exists():
                        source_file = str(root / source_file)
                        break
                else:
                    source_file = str(sources_root.parent / source_file)
            if not Path(source_file).exists():
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.ERROR,
                        message=f"Source file missing for page '{node.title}': {source_file}",
                        page_id=node.id,
                        details={MetadataField.SOURCE_FILE: str(source_file)},
                    )
                )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All referenced sources exist"))
        return results


class OrphanAttachmentsRule(BaseLintRule):
    """Detect attachment files whose slug matches no source in the vault.

    Files under ``attachments/<slug>/`` whose slug is derivable from a live
    source are auto-adopted at compile start (manifest entry created, flags
    derived from config). Files whose slug matches NO source cannot be
    recovered by that pass — the parent source is gone or was renamed — so
    they are flagged here for manual intervention. ``--fix`` deletes them
    on disk + tombstones them in the ``attachment_manifest`` table (in ``.knox/knox.db``); the caller
    must opt in via the lint ``--fix`` flag.
    """

    name = "orphan-attachments"
    description = "Attachment files with no matching source (unrecoverable by auto-adoption)"

    #: Deletes files on disk — never auto-run by the post-compile lint or
    #: default ``knox lint check``. Only an explicit ``fix_destructive``
    #: opt-in triggers deletion (p863 wiki roundup 2026-09-10).
    destructive: ClassVar[bool] = True

    # Stored on the instance by check() so fix() can locate the manifest
    # without the lint runner needing to thread ``sources_dir`` through.
    _last_sources_dir: str | None = None

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        self._last_sources_dir = sources_dir
        results: list[LintResult] = []
        if sources_dir is None:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided"))
            return results

        base_dir = Path(sources_dir).parent
        vp = VaultPaths(base_dir)
        attachments_root = vp.attachments_dir
        if not attachments_root.is_dir():
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No attachments directory"))
            return results

        # A compile must have run at least once — before that, orphan state
        # is meaningless (nothing was ever adoptable).
        if not TierStateStore(vp.compile_state_file).load().entries:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No compile state — run compile first"))
            return results

        sources_root = Path(sources_dir)
        orphans = orphan_attachment_dirs(base_dir, sources_root)

        if orphans:
            for rel in orphans:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=(
                            f"Orphaned attachment '{rel}' — slug matches no source in sources/ "
                            "(re-add the source, move the file under the correct slug, or delete it)"
                        ),
                        details={"path": rel},
                    )
                )
        else:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No orphaned attachments"))
        return results

    def fix(  # noqa: PLR6301 — signature on lint-rule contract
        self,
        wiki: CompiledWiki,
        results: list[LintResult],
    ) -> list[LintResult]:
        """Delete orphan attachment files from disk + tombstone in the manifest.

        Destructive. Each FAIL result becomes a removed file + a tombstone
        entry in the ``attachment_manifest`` table (in ``.knox/knox.db``). Empty slug directories are
        also removed after their last file is deleted. Failures (permission
        errors, missing manifest) leave the result outcome FAIL but record
        the error in ``details["error"]`` and log at WARNING.

        The lint runner's ``--fix`` flag is the user's opt-in to this
        destructive action. There is no dry-run.

        Reads ``sources_dir`` from the instance attribute populated by the
        most recent :meth:`check` call — the lint runner calls them in
        sequence (``check`` → ``fix``) on the same rule instance.
        """
        sources_dir = self._last_sources_dir
        if not results or sources_dir is None:
            return list(results)

        base_dir = Path(sources_dir).parent
        vp = VaultPaths(base_dir)
        manifest = AttachmentManifest(vp.attachments_manifest_file)

        updated: list[LintResult] = []
        for r in results:
            if r.outcome != LintRuleOutcome.FAIL:
                updated.append(r)
                continue
            rel_path = (r.details or {}).get("path")
            if not rel_path:
                updated.append(r)
                continue
            abs_path = base_dir / rel_path
            try:
                if abs_path.is_file():
                    abs_path.unlink()
                manifest.remove(rel_path)
                r.fix_applied = True
                r.message = f"Deleted orphan attachment '{rel_path}' (was unrecoverable)"
                logger.info("Deleted orphan attachment: %s", rel_path)
            except OSError as exc:
                logger.warning("Could not delete orphan attachment %s: %s", abs_path, exc)
                r.message = f"Could not delete orphan attachment '{rel_path}': {exc}"
                r.details = {**(r.details or {}), "error": str(exc)}
            updated.append(r)

        # Clean up empty slug directories.
        if vp.attachments_dir.is_dir():
            for slug_dir in sorted(vp.attachments_dir.iterdir()):
                if slug_dir.is_dir() and not any(slug_dir.iterdir()):
                    with contextlib.suppress(OSError):
                        slug_dir.rmdir()
        return updated


# ── Rule 9: duplicate-references ────────────────────────────────────────


class DuplicateReferencesRule(BaseLintRule):
    """Detect duplicate bibliographic references across sources.

    Loads ``.knox/references.json`` and checks for references that appear
    in multiple source files — same DOI, same title+first-author pair, or
    identical ``raw`` string. When ``embedding_model`` is set, also
    detects near-duplicates via cosine similarity on title embeddings
    (same pattern as concept merging in ``assembly.py``).

    Duplicates are informational (not errors) because they are expected
    in multi-source corpora.
    """

    name = "duplicate-references"
    description = "Bibliographic references cited by more than one source"

    def __init__(
        self,
        *,
        embedding_model: str = "",
        fuzzy_threshold: float = 0.85,
    ) -> None:
        """Initialize the rule.

        Parameters
        ----------
        embedding_model:
            When non-empty, enables fuzzy dedup via embedding cosine similarity.
        fuzzy_threshold:
            Similarity threshold for fuzzy matching (0.0–1.0).
        """
        self.embedding_model = embedding_model
        self.fuzzy_threshold = fuzzy_threshold

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:  # noqa: ARG002, PLR0915
        results: list[LintResult] = []
        if sources_dir is None:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided"))
            return results

        base_dir = Path(sources_dir).parent
        store = ReferenceStore(VaultPaths(base_dir).references_file)
        raw_data = store.to_dict()
        if not raw_data:
            msg = f"No {REFERENCES_FILE} found — run 'knox compile' with reference extraction"
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message=msg))
            return results

        duplicates = self._find_exact_duplicates(raw_data)
        near_dupes = self._find_fuzzy_duplicates(store)

        if duplicates or near_dupes:
            all_findings = duplicates + near_dupes
            msg = f"{len(all_findings)} reference finding(s) ({len(duplicates)} exact, {len(near_dupes)} fuzzy) across {store.source_count()} source(s)"
            results.append(
                LintResult(
                    rule=self.name,
                    outcome=LintRuleOutcome.FAIL,
                    severity=Severity.INFO,
                    message=msg,
                    details={"duplicates": duplicates, "near_duplicates": near_dupes},
                )
            )
        else:
            msg = f"No cross-source duplicate references ({store.total_count()} refs across {store.source_count()} source(s))"
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message=msg))
        return results

    def _find_fuzzy_duplicates(self, store: ReferenceStore) -> list[dict[str, Any]]:
        """Find near-duplicate references via embedding cosine similarity."""
        near_dupes: list[dict[str, Any]] = []
        if not self.embedding_model:
            return near_dupes

        all_refs = store.all_references()
        consolidated = consolidate_references(
            all_refs,
            self.embedding_model,
            threshold=self.fuzzy_threshold,
        )
        fuzzy_removed = len(all_refs) - len(consolidated)
        if fuzzy_removed > 0:
            near_dupes.append(
                {
                    "type": "fuzzy",
                    "removed": fuzzy_removed,
                    "threshold": self.fuzzy_threshold,
                    "remaining": len(consolidated),
                }
            )
        return near_dupes

    @staticmethod
    def _find_exact_duplicates(
        raw_data: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Find exact-duplicate references (DOI, title+author, raw)."""
        by_doi: dict[str, list[tuple[str, str]]] = {}
        by_title_author: dict[str, list[tuple[str, str]]] = {}
        by_raw: list[tuple[str, str, str]] = []

        for source, ref_list in raw_data.items():
            source_name = Path(source).name
            for ref_dict in ref_list:
                label = ref_dict.get("id") or ref_dict.get("title", "")[:_REF_LABEL_TITLE_MAX]
                doi = ref_dict.get("doi", "").strip().lower()
                title = ref_dict.get("title", "").strip().lower()
                raw = ref_dict.get("raw", "").strip()
                authors = ref_dict.get("author", [])
                first_family = ""
                if authors and isinstance(authors, list):
                    first_family = authors[0].get("family", "") if isinstance(authors[0], dict) else ""

                if doi:
                    by_doi.setdefault(doi, []).append((source_name, label))
                if title and first_family:
                    key = f"{first_family.lower()}|{title}"
                    by_title_author.setdefault(key, []).append((source_name, label))
                if raw:
                    by_raw.append((raw.lower(), source_name, label))

        duplicates: list[dict[str, Any]] = []

        for doi, entries in by_doi.items():
            sources_seen = {s for s, _ in entries}
            if len(sources_seen) > 1:
                refs = [lbl for _, lbl in entries]
                duplicates.append({"type": "doi", "doi": doi, "sources": sorted(sources_seen), "refs": refs})

        for key, entries in by_title_author.items():
            sources_seen = {s for s, _ in entries}
            if len(sources_seen) > 1:
                family, title = key.split("|", 1)
                refs = [lbl for _, lbl in entries]
                duplicates.append({"type": "title-author", "author": family, "title": title[:60], "sources": sorted(sources_seen), "refs": refs})

        raw_seen: dict[str, list[str]] = {}
        for raw_text, source_name, _ in by_raw:
            raw_seen.setdefault(raw_text, []).append(source_name)
        for raw_text, sources_for_raw in raw_seen.items():
            unique_sources = set(sources_for_raw)
            if len(unique_sources) > 1:
                duplicates.append({"type": "raw", "raw": raw_text[:80], "sources": sorted(unique_sources)})

        return duplicates


# ── Rule 10: rumdl ───────────────────────────────────────────────────────


class RumdlRule(BaseLintRule):
    """Markdown style via ``rumdl check`` on all vault folders except sources.

    Scans every ``*.md`` file under the vault root (derived from
    *sources_dir*'s parent), excluding ``sources/``, ``.knox/``, and
    ``attachments/``. Sources are never read or modified. Files are passed
    explicitly to bypass rumdl's ignore-file filtering — self-written notes
    that are gitignored still get checked. rumdl auto-discovers a
    ``.rumdl.toml`` in the vault.
    """

    name = "rumdl"
    description = "Markdown style issues found by rumdl (all vault folders except sources)"

    #: Explicit paths bypass rumdl's ignore-file filtering, but the Windows
    #: command line caps at ~32k chars — batch to stay well under it.
    _BATCH_SIZE = 100

    def __init__(self, rumdl_config: Path | None = None) -> None:
        self.rumdl_config = rumdl_config

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        exe = shutil.which("rumdl")
        if not exe:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="rumdl is not installed")]
        if not sources_dir:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="Sources directory unknown — cannot derive vault root")]
        files = self._md_files(sources_dir)
        if not files:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No markdown files to check")]

        findings: list[dict[str, Any]] = []
        for batch in _batched(files, self._BATCH_SIZE):
            findings.extend(self._run_rumdl(exe, ["check", *self._config_args(), "--output-format", "json", *batch]))
        if not findings:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All markdown files pass rumdl checks")]
        return [self._to_result(f) for f in findings]

    def fix(self, wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:
        """Apply ``rumdl check --fix`` to the files with fixable findings."""
        exe = shutil.which("rumdl")
        if not exe:
            return results
        fixable_files = sorted(
            {r.details.get("file", "") for r in results if r.outcome == LintRuleOutcome.FAIL and r.details.get("fixable") and r.details.get("file")}
        )
        for batch in _batched(fixable_files, self._BATCH_SIZE):
            self._run_rumdl(exe, ["check", "--fix", *self._config_args(), "--output-format", "json", *batch])
        for r in results:
            if r.outcome == LintRuleOutcome.FAIL and r.details.get("fixable"):
                r.fix_applied = True
        return results

    def _config_args(self) -> list[str]:
        """``--config`` args when a custom rumdl config is set (else auto-discovery)."""
        return ["--config", str(self.rumdl_config)] if self.rumdl_config else []

    def _to_result(self, finding: dict[str, Any]) -> LintResult:
        severity = Severity.ERROR if finding.get("severity") == "error" else Severity.WARNING
        return LintResult(
            rule=self.name,
            outcome=LintRuleOutcome.FAIL,
            severity=severity,
            message=f"{finding.get('file')}:{finding.get('line')} {finding.get('rule')}: {finding.get('message')}",
            page_id="",
            details={
                "file": finding.get("file", ""),
                "line": finding.get("line"),
                "column": finding.get("column"),
                "rumdl_rule": finding.get("rule"),
                "fixable": bool(finding.get("fixable")),
            },
        )

    @staticmethod
    def _md_files(sources_dir: str) -> list[str]:
        """All ``*.md`` files under the vault root, excluding sources/.knox/attachments."""
        base = Path(sources_dir).parent
        sources_path = Path(sources_dir)
        excluded_names = {".knox", "attachments"}
        files: list[str] = []
        for path in base.rglob("*.md"):
            rel = path.relative_to(base)
            if sources_path in path.parents:
                continue
            if any(part in excluded_names for part in rel.parts[:-1]):
                continue
            files.append(str(path))
        return sorted(files)

    @staticmethod
    def _run_rumdl(exe: str, cmd: list[str]) -> list[dict[str, Any]]:
        """Run rumdl and return its JSON diagnostics (empty list on failure)."""
        try:
            result = subprocess.run(  # noqa: S603 — internally constructed cmd list
                [exe, *cmd],
                capture_output=True,
                encoding="utf-8",
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("rumdl invocation failed: %s", cmd[:_CMD_PREVIEW_ARGS])
            return []
        try:
            return json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            return []


def unreferenced_attachment_files(base_dir: Path) -> list[str]:
    """Vault-relative regenerable media in live source dirs that no page links.

    Page links are the source of truth (markdown-first invariant): a file the
    written pages do not reference is not part of the wiki. Only directories
    whose slug matches a live source are considered — dead slugs are
    :class:`OrphanAttachmentsRule`'s job — and only files byte-identical to an
    extracted image (:func:`_is_cached_extraction`), so nothing the wiki cannot
    recreate is ever reported.
    """
    vp = VaultPaths(base_dir)
    if not vp.attachments_dir.is_dir():
        return []

    referenced = _referenced_attachment_paths(base_dir)
    live_slugs = _live_attachment_slugs(base_dir)
    cache_dir = vp.images_dir

    stale: list[str] = []
    for path in sorted(vp.attachments_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base_dir).as_posix()
        slug = path.relative_to(vp.attachments_dir).as_posix().split("/", 1)[0]
        if slug not in live_slugs:
            continue  # orphan-attachments owns directories with no live source
        if rel in referenced:
            continue
        if not _is_cached_extraction(path, cache_dir):
            continue  # not a regenerable copy — leave it alone
        stale.append(rel)
    return stale


def _live_attachment_slugs(base_dir: Path) -> set[str]:
    """Slug forms of every live source (both full-path and bare-name forms)."""
    vp = VaultPaths(base_dir)
    if not vp.sources_dir.is_dir():
        return set()
    slugs: set[str] = set()
    for source in vp.sources_dir.rglob("*"):
        if source.is_file():
            slugs |= attachment_slugs_for_source(source.relative_to(vp.sources_dir))
    return slugs


def _is_cached_extraction(path: Path, cache_dir: Path) -> bool:
    """Is *path* a byte-identical copy of an image the parser extracted?

    Compares content, not just the name: same-named files diverge when a source
    is re-parsed, and a re-parse invalidates the copy anyway. Cheap enough —
    the caller only asks about files no page references.
    """
    original = cache_dir / path.name
    try:
        if not original.is_file() or original.stat().st_size != path.stat().st_size:
            return False
        return original.read_bytes() == path.read_bytes()
    except OSError:
        return False


def _referenced_attachment_paths(base_dir: Path) -> set[str]:
    """Every vault-relative ``attachments/...`` path linked from a markdown file.

    Reads the written pages, not the in-memory graph: the writer rewrites image
    paths to relative links at write time, so only the files on disk show what
    the wiki actually references. Links resolve page-relative first (what the
    wiki writer emits) and vault-relative as a fallback (what an export uses).
    """
    referenced: set[str] = set()
    for md in base_dir.rglob("*.md"):
        parts = md.relative_to(base_dir).parts[:-1]
        if any(part in {".knox", "sources", "attachments"} for part in parts):
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for raw in _ATTACHMENT_LINK_RE.findall(text):
            link = raw.strip().strip("<>").split("#", 1)[0]
            for candidate in (md.parent / link, base_dir / link):
                rel = _vault_relative(base_dir, candidate)
                if rel is not None:
                    referenced.add(rel)
    return referenced


def _vault_relative(base_dir: Path, candidate: Path) -> str | None:
    """``candidate`` as a vault-relative posix path, or ``None`` when outside."""
    try:
        return Path(os.path.normpath(candidate)).relative_to(base_dir).as_posix()
    except ValueError:
        return None


def _prune_empty_dirs(attachments_root: Path) -> None:
    """Remove directories left empty under ``attachments/`` (deepest first)."""
    for d in sorted((p for p in attachments_root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        with contextlib.suppress(OSError):
            d.rmdir()


# ── Rule: orphan-attachments ─────────────────────────────────────────────


def orphan_attachment_dirs(base_dir: Path, sources_dir: Path) -> list[str]:
    """Attachment files whose directory slug matches no live source.

    Returns **vault-relative** paths (``attachments/<slug>/<file>``). Shared by
    :class:`OrphanAttachmentsRule` and ``knox attachments verify`` so the slug
    logic has exactly one implementation. Matches both slug forms (full
    relative path and bare filename) — see :func:`attachment_slugs_for_source`.
    """
    attachments_root = VaultPaths(base_dir).attachments_dir
    if not attachments_root.is_dir():
        return []

    sources_root = Path(sources_dir)
    live_slugs: set[str] = set()
    if sources_root.is_dir():
        for p in sources_root.rglob("*"):
            if p.is_file() and not is_protected_source(p, sources_root):
                live_slugs.update(attachment_slugs_for_source(p.relative_to(sources_root).as_posix()))

    orphans: list[str] = []
    for slug_dir in sorted(attachments_root.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name in live_slugs:
            continue
        for f in sorted(slug_dir.rglob("*")):
            if f.is_file():
                orphans.append(f.relative_to(base_dir).as_posix())
    return orphans


def _incoming_link_counts(wiki: CompiledWiki) -> dict[str, int]:
    """Count incoming ``[[wikilinks]]`` per node id."""
    counts: dict[str, int] = {}
    title_to_id = _node_titles_index(wiki)
    for node in wiki.nodes:
        for target_title in _extract_wikilinks(node.content):
            target_id = title_to_id.get(target_title.lower(), target_title)
            counts[target_id] = counts.get(target_id, 0) + 1
    return counts


def _extract_wikilinks(content: str) -> list[str]:
    """Return all ``[[target]]`` strings from *content*."""
    return _WIKILINK_RE.findall(content)


def _node_titles_index(wiki: CompiledWiki) -> dict[str, str]:
    """Map ``node.title.lower()`` → ``node.id`` (legacy alias for callers)."""
    return {n.title.lower(): n.id for n in wiki.nodes}


def _node_lookup_index(wiki: CompiledWiki) -> tuple[dict[str, str], dict[str, str]]:
    """Build a wikilink lookup that also accepts slugs and acronym aliases.

    Returns a tuple of:
    - ``alias_index``: maps any of ``node.title.lower()``, ``node.id``,
      or the acronym-stripped title (e.g. ``"polqa"`` extracted from
      ``"POLQA (Perceptual Objective Listening Quality Analysis)"``)
      to ``node.id``.
    - ``slug_to_id``: just ``node.id`` → ``node.id``, used by alias detection.

    Also indexes the filesystem-path forms ``concepts/<id>`` and
    ``entities/<id>`` (entities get a sub-folder per type, e.g.
    ``entities/standard/foo``), so a wikilink emitted by the model that
    includes the subdirectory prefix (``[[concepts/foo]]``) still
    resolves.

    The acronym extraction lives in :mod:`knox.compilation.concept_keys`
    so the same normalization is used for concept-merge and wikilink
    resolution — one source of truth.
    """
    alias_index: dict[str, str] = {}
    slug_to_id: dict[str, str] = {}
    for n in wiki.nodes:
        # Lower-case first entry wins; existing index already preferred title
        alias_index.setdefault(n.title.lower(), n.id)
        alias_index.setdefault(n.id, n.id)
        slug_to_id[n.id] = n.id
        stripped = normalize_concept_key(n.title)
        if stripped:
            alias_index.setdefault(stripped, n.id)
        # Also index the bare-acronym form so wikilinks like ``[[MOS-LQO]]``
        # resolve to the page titled ``MOS-LQO (Mean Opinion Score - ...)``.
        # ``normalize_concept_key`` strips the parenthetical, so ``stripped``
        # already gives us this form for hyphen-having prefixes.
        title_before_paren = n.title.split("(", 1)[0].strip().lower()
        if title_before_paren and title_before_paren != n.title.lower():
            alias_index.setdefault(title_before_paren, n.id)
        # Index filesystem-path forms so wikilinks with subdirectory
        # prefix (``[[concepts/foo]]``) resolve. The vault writer places
        # nodes under KIND_FOLDERS[<node_type>]/<id>.md; entities get an
        # extra <type>/ level per _folder_for_kind. We can't know the
        # entity_subtype here without inspecting metadata, so we register
        # both forms (``entities/<id>`` for unknown-type lookup and let
        # nearest_alias pick the closer one).
        folder = KIND_FOLDERS.get(n.node_type, "concepts")
        alias_index.setdefault(f"{folder}/{n.id}", n.id)
    return alias_index, slug_to_id


def _batched(items: list[str], size: int) -> list[list[str]]:
    """Split *items* into consecutive batches of at most *size*."""
    return [items[i : i + size] for i in range(0, len(items), size)]


# ── Registry ────────────────────────────────────────────────────────────

STRUCTURAL_RULES: list[BaseLintRule] = [
    ImageEmbedRule(),
    BrokenWikilinksRule(),
    KnoxReferenceRule(),
    MissingFrontmatterRule(),
    OrphanPageRule(),
    DuplicateTitleRule(),
    EmptyPageRule(),
    HeadingSkipRule(),
    StalePageRule(),
    SourceNotFoundRule(),
    OrphanAttachmentsRule(),
    UnreferencedAttachmentsRule(),
]

RULE_BY_NAME: dict[str, Any] = {r.name: r for r in STRUCTURAL_RULES}
# RumdlRule and DuplicateReferencesRule are not singletons — register factory
# references so they can be looked up by name and instantiated with config
# values in LintRunner._resolve_rules.
RULE_BY_NAME["rumdl"] = RumdlRule
RULE_BY_NAME["duplicate-references"] = DuplicateReferencesRule
