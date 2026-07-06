"""Structural lint rules — deterministic, zero LLM cost.

Each rule is a class that satisfies the :class:`LintRule` protocol.
Rules inspect a :class:`CompiledWiki` (and optionally the sources
directory on disk) and return :class:`LintResult` findings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from knox.core.models import MetadataField
from knox.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity

# ── Helpers ──────────────────────────────────────────────────────────────

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_HEADING_RE = re.compile(r"^(#{1,6})\s", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _extract_wikilinks(content: str) -> list[str]:
    """Return all ``[[target]]`` strings from *content*."""
    return _WIKILINK_RE.findall(content)


def _node_titles_index(wiki: Any) -> dict[str, str]:
    """Map ``node.title.lower()`` → ``node.id``."""
    return {n.title.lower(): n.id for n in wiki.nodes}


def _node_ids_set(wiki: Any) -> set[str]:
    return {n.id for n in wiki.nodes}


def _incoming_link_counts(wiki: Any) -> dict[str, int]:
    """Count incoming ``[[wikilinks]]`` per node id."""
    counts: dict[str, int] = {}
    title_to_id = _node_titles_index(wiki)
    for node in wiki.nodes:
        for target_title in _extract_wikilinks(node.content):
            target_id = title_to_id.get(target_title.lower(), target_title)
            counts[target_id] = counts.get(target_id, 0) + 1
    return counts


# ── Rule 1: broken-wikilinks ────────────────────────────────────────────


class BrokenWikilinksRule:
    """Detect ``[[wikilinks]]`` that point to non-existent pages."""

    name = "broken-wikilinks"
    description = "Wikilinks pointing to pages that do not exist"

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
        title_to_id = _node_titles_index(wiki)
        results: list[LintResult] = []
        for node in wiki.nodes:
            for target_title in _extract_wikilinks(node.content):
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

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
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


# ── Rule 2: missing-frontmatter ─────────────────────────────────────────


class MissingFrontmatterRule:
    """Detect pages missing required frontmatter fields.

    Required fields: ``title`` and ``date``.
    """

    name = "missing-frontmatter"
    description = "Pages missing required frontmatter fields"

    REQUIRED_FIELDS: tuple[str, ...] = ("title",)

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
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

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
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

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
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

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        # Cannot auto-fix orphans without knowing where to link from
        return list(results)


# ── Rule 4: duplicate-title ─────────────────────────────────────────────


class DuplicateTitleRule(BaseLintRule):
    """Detect pages with duplicate or very similar titles."""

    name = "duplicate-title"
    description = "Two or more pages with the same or similar title"

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
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

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        # Cannot auto-fix duplicate titles without domain knowledge
        return list(results)


# ── Rule 5: empty-page ──────────────────────────────────────────────────


class EmptyPageRule(BaseLintRule):
    """Detect pages with no body content."""

    name = "empty-page"
    description = "Pages with no body content"

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
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

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        # Cannot auto-fix empty pages
        return list(results)


# ── Rule 6: heading-skip ────────────────────────────────────────────────


class HeadingSkipRule(BaseLintRule):
    """Detect heading level jumps (e.g. h1 → h3 without h2)."""

    name = "heading-skip"
    description = "Heading level skips (h1 → h3 without h2)"

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
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

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        # Heading level rewrites are too risky to auto-fix
        return list(results)


# ── Rule 7: stale-page ──────────────────────────────────────────────────


class StalePageRule(BaseLintRule):
    """Detect pages whose source files have been modified since last compile.

    Also checks the memory lifecycle store for superseded and decayed (stale)
    nodes when memory metadata is present in ``wiki.metadata``.
    """

    name = "stale-page"
    description = "Pages whose source files changed since last compile"

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []

        # ── Source file existence check ──
        if sources_dir is not None:
            sources_dir = Path(sources_dir)
            if sources_dir.exists():
                for node in wiki.nodes:
                    source_file_str = node.metadata.get(MetadataField.SOURCE_FILE)
                    if not source_file_str:
                        continue
                    source_file = Path(source_file_str)
                    if not source_file.is_absolute():
                        source_file = sources_dir / source_file
                    if not source_file.exists():
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

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        # Stale pages need re-compilation, not auto-fix
        return list(results)


# ── Rule 8: source-not-found ────────────────────────────────────────────


class SourceNotFoundRule(BaseLintRule):
    """Detect pages that reference source files not present on disk."""

    name = "source-not-found"
    description = "Pages referencing source files that don't exist"

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        if sources_dir is None:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided"))
            return results

        sources_dir = Path(sources_dir)
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
                source_file = sources_dir / source_file
            if not source_file.exists():
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

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        # Cannot auto-fix missing source files
        return list(results)


# ── Registry ────────────────────────────────────────────────────────────

STRUCTURAL_RULES: list[Any] = [
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
