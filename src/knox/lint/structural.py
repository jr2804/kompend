"""Structural lint rules — deterministic, zero LLM cost.

Each rule is a class that satisfies the :class:`LintRule` protocol.
Rules inspect a :class:`CompiledWiki` (and optionally the sources
directory on disk) and return :class:`LintResult` findings.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki, MetadataField
from knox.core.paths import REFERENCES_FILE, VaultPaths
from knox.core.pipeline.references import ReferenceStore, consolidate_references
from knox.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from knox.logging import get_logger

#: Max chars of a reference title used for display labels in lint findings.
_REF_LABEL_TITLE_MAX = 40

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
_IMAGE_EMBED_RE = re.compile(
    r"(?<!\!)\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff))\]\]",
    re.IGNORECASE,
)

logger = get_logger(__name__)


# ── Rule 1: broken-wikilinks ──────────────────────────────────────────────


class BrokenWikilinksRule(BaseLintRule):
    """Detect ``[[wikilinks]]`` that point to non-existent pages."""

    name = "broken-wikilinks"
    description = "Wikilinks pointing to pages that do not exist"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        title_to_id = _node_titles_index(wiki)
        results: list[LintResult] = []
        for node in wiki.nodes:
            for target_title in _extract_wikilinks(node.content):
                if Path(target_title).suffix.lower() in _IMAGE_EXTENSIONS:
                    # Image embeds are handled by ImageEmbedRule — not page links.
                    continue
                if target_title.lower() not in title_to_id:
                    results.append(
                        LintResult(
                            rule=self.name,
                            outcome=LintRuleOutcome.FAIL,
                            severity=Severity.ERROR,
                            message=f"Broken wikilink [[{target_title}]] in page '{node.title}'",
                            page_id=node.id,
                            details={"target": target_title},
                        )
                    )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All wikilinks resolve"))
        return results

    def fix(self, wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        """Remove broken wikilinks from page content."""
        page_edits: dict[str, list[str]] = {}
        for r in results:
            if r.outcome == LintRuleOutcome.FAIL and r.details.get("target"):
                page_edits.setdefault(r.page_id, []).append(r.details["target"])

        fixed = list(results)
        for node in wiki.nodes:
            targets = page_edits.get(node.id)
            if not targets:
                continue
            for target in targets:
                # Remove the broken [[target]] but keep surrounding text
                node.content = node.content.replace(f"[[{target}]]", target)
            # Mark results as fixed
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
        ref_file = VaultPaths(base_dir).references_file
        if not ref_file.exists():
            msg = f"No {REFERENCES_FILE} found — run 'knox compile' with reference extraction"
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message=msg))
            return results

        store = ReferenceStore(ref_file)
        raw_data = store.to_dict()

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
            logger.warning("rumdl invocation failed: %s", cmd[:4])
            return []
        try:
            return json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            return []


def _node_ids_set(wiki: CompiledWiki) -> set[str]:
    return {n.id for n in wiki.nodes}


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
    """Map ``node.title.lower()`` → ``node.id``."""
    return {n.title.lower(): n.id for n in wiki.nodes}


def _batched(items: list[str], size: int) -> list[list[str]]:
    """Split *items* into consecutive batches of at most *size*."""
    return [items[i : i + size] for i in range(0, len(items), size)]


# ── Registry ────────────────────────────────────────────────────────────

STRUCTURAL_RULES: list[BaseLintRule] = [
    ImageEmbedRule(),
    BrokenWikilinksRule(),
    MissingFrontmatterRule(),
    OrphanPageRule(),
    DuplicateTitleRule(),
    EmptyPageRule(),
    HeadingSkipRule(),
    StalePageRule(),
    SourceNotFoundRule(),
]

RULE_BY_NAME: dict[str, Any] = {r.name: r for r in STRUCTURAL_RULES}
# RumdlRule and DuplicateReferencesRule are not singletons — register factory
# references so they can be looked up by name and instantiated with config
# values in LintRunner._resolve_rules.
RULE_BY_NAME["rumdl"] = RumdlRule
RULE_BY_NAME["duplicate-references"] = DuplicateReferencesRule
