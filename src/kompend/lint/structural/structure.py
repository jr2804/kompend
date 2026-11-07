"""Structural lint family: the shape of the compiled wiki.

Owns the rules that look at a page set rather than at one link target or
one file on disk — pages nobody links to (``orphan-page``), pages that
share a title (``duplicate-title``), pages with no body (``empty-page``)
and heading levels that skip a step (``heading-skip``).
"""

from __future__ import annotations

import re

from kompend.core.models import CompiledWiki
from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from kompend.lint.structural.links import _extract_wikilinks, _node_titles_index

_HEADING_RE = re.compile(r"^(#{1,6})\s", re.MULTILINE)


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


def _incoming_link_counts(wiki: CompiledWiki) -> dict[str, int]:
    """Count incoming ``[[wikilinks]]`` per node id."""
    counts: dict[str, int] = {}
    title_to_id = _node_titles_index(wiki)
    for node in wiki.nodes:
        for target_title in _extract_wikilinks(node.content):
            target_id = title_to_id.get(target_title.lower(), target_title)
            counts[target_id] = counts.get(target_id, 0) + 1
    return counts
