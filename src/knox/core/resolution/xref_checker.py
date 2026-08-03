"""Programmatic cross-reference checker for KNOX Tier 3.

Runs consistency checks at DATA level — no LLM needed. Results feed into
Phase 5 compilation as consistency notes.

Checks:
    - **Definitions**: entities defined differently across locations
    - **References**: unresolved cross-references (Section/Table/Figure)
    - **Table consistency**: contradicting values for the same entity
"""

from __future__ import annotations

import re

from knox.core.models import (
    ChunkTree,
    CrossReferenceIssue,
    CrossReferenceResult,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)
from knox.core.resolution.entity_registry import EntityRegistry
from knox.logging import get_logger

# ── Reference patterns ──────────────────────────────────────────────────

# ponytail: numbered references only ("see Section 3.2"). Unnumbered refs
# ("see Methodology section") are not resolved here. Phase 5 compilation
# handles those via LLM-based checking when needed.
_REF_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("section", re.compile(r"\b(?:see\s+)?(?:Section|Sec\.?|§)\s*([\d.]+)", re.IGNORECASE)),
    ("table", re.compile(r"\b(?:see\s+)?(?:Table|Tab\.?)\s*(\d+[a-zA-Z]?)", re.IGNORECASE)),
    ("figure", re.compile(r"\b(?:see\s+)?(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)", re.IGNORECASE)),
    ("equation", re.compile(r"\b(?:see\s+)?(?:Equation|Eq\.?)\s*(\d+[a-zA-Z]?)", re.IGNORECASE)),
    ("chapter", re.compile(r"\b(?:see\s+)?Chapter\s+(\d+[a-zA-Z]?)", re.IGNORECASE)),
]

logger = get_logger(__name__)


class CrossReferenceChecker:
    """Run programmatic consistency checks over a ChunkTree set.

    All checks are deterministic and require no LLM.  Instantiate with
    an :class:`EntityRegistry` (already populated via ``scan_tree``),
    then call :meth:`check_all` with the trees.
    """

    def __init__(self, registry: EntityRegistry) -> None:
        self._registry = registry

    # ── Public API ──────────────────────────────────────────────────

    def check_all(self, trees: list[ChunkTree]) -> CrossReferenceResult:
        """Run all checks and return the combined result."""
        issues: list[CrossReferenceIssue] = []
        issues.extend(self.check_definitions())
        issues.extend(self.check_references(trees))
        issues.extend(self.check_table_consistency(trees))

        entities_with_issues: set[str] = set()
        for issue in issues:
            if issue.entity:
                entities_with_issues.add(issue.entity)
        resolved = self._registry.entity_count - len(entities_with_issues)
        return CrossReferenceResult(
            issues=issues,
            entity_count=self._registry.entity_count,
            resolved_count=max(0, resolved),
        )

    def check_definitions(self) -> list[CrossReferenceIssue]:
        """Flag entities with divergent definitions."""
        return self._registry.find_conflicts()

    def check_references(self, trees: list[ChunkTree]) -> list[CrossReferenceIssue]:
        """Flag unresolved Section/Table/Figure/Equation references.

        Collects all valid targets from the trees (heading numbers,
        table/figure/equation captions and indices), then scans every
        text chunk for reference patterns that don't resolve.
        """
        valid_targets = self._collect_targets(trees)
        issues: list[CrossReferenceIssue] = []

        for tree in trees:
            source_file = str(tree.source_file) if tree.source_file else "unknown"
            for node in tree.walk():
                if not node.chunk or not node.chunk.text:
                    continue
                heading_path = "/".join(node.heading_path)
                for ref_type, pattern in _REF_PATTERNS:
                    for match in pattern.finditer(node.chunk.text):
                        ref_id = match.group(1)
                        if ref_id not in valid_targets.get(ref_type, set()):
                            issues.append(
                                CrossReferenceIssue(
                                    issue_type="unresolved",
                                    description=(f"Reference to {ref_type} {ref_id} does not match any {ref_type} in the document"),
                                    entity=f"{ref_type}:{ref_id}",
                                    locations=[f"{source_file}:{heading_path}"],
                                )
                            )
        return issues

    def check_table_consistency(self, trees: list[ChunkTree]) -> list[CrossReferenceIssue]:
        """Flag contradicting values for the same entity across tables.

        Collects all StructuredTables, groups by shared column headers,
        and flags rows where the same key column has different values
        in the comparison column.
        """
        all_tables = self._collect_tables(trees)
        if len(all_tables) < 2:
            return []

        issues: list[CrossReferenceIssue] = []

        for i, (meta_a, table_a) in enumerate(all_tables):
            for meta_b, table_b in all_tables[i + 1 :]:
                shared_cols = set(table_a.columns) & set(table_b.columns)
                if len(shared_cols) < 2:
                    continue
                key_col = sorted(shared_cols)[0]
                val_cols = sorted(shared_cols)[1:]

                map_a = {row.get(key_col, ""): row for row in table_a.rows}
                map_b = {row.get(key_col, ""): row for row in table_b.rows}

                for key_val, row_a in map_a.items():
                    if not key_val or key_val not in map_b:
                        continue
                    row_b = map_b[key_val]
                    for val_col in val_cols:
                        val_a = str(row_a.get(val_col, "")).strip()
                        val_b = str(row_b.get(val_col, "")).strip()
                        if val_a and val_b and val_a.lower() != val_b.lower():
                            issues.append(
                                CrossReferenceIssue(
                                    issue_type="contradiction",
                                    description=(f'Entity "{key_val}" has "{val_col}"="{val_a}" in one table but "{val_b}" in another'),
                                    entity=key_val,
                                    locations=[meta_a, meta_b],
                                )
                            )
        return issues

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _collect_targets(trees: list[ChunkTree]) -> dict[str, set[str]]:
        """Collect all valid reference targets from trees."""
        targets: dict[str, set[str]] = {
            "section": set(),
            "table": set(),
            "figure": set(),
            "equation": set(),
            "chapter": set(),
        }
        table_idx = 0
        figure_idx = 0
        equation_idx = 0

        def _add_numbered(key: str, idx: int, caption: str | None) -> None:
            targets[key].add(str(idx))
            if caption:
                cap_num = _extract_number(caption)
                if cap_num:
                    targets[key].add(cap_num)

        for tree in trees:
            for node in tree.walk():
                if node.heading_path:
                    section_num = _extract_heading_number(node.heading_path[-1])
                    if section_num:
                        targets["section"].add(section_num)

                if node.structured:
                    if isinstance(node.structured, StructuredTable):
                        table_idx += 1
                        _add_numbered("table", table_idx, node.structured.caption)
                    elif isinstance(node.structured, StructuredFigure):
                        figure_idx += 1
                        _add_numbered("figure", figure_idx, node.structured.caption)
                    elif isinstance(node.structured, StructuredEquation):
                        equation_idx += 1
                        _add_numbered("equation", equation_idx, node.structured.caption)

        return targets

    @staticmethod
    def _collect_tables(trees: list[ChunkTree]) -> list[tuple[str, StructuredTable]]:
        """Collect all structured tables with their location metadata."""
        result: list[tuple[str, StructuredTable]] = []
        for tree in trees:
            source_file = str(tree.source_file) if tree.source_file else "unknown"
            for node in tree.walk():
                if node.structured and isinstance(node.structured, StructuredTable):
                    heading_path = "/".join(node.heading_path)
                    result.append((f"{source_file}:{heading_path}", node.structured))
        return result


def _extract_heading_number(heading: str) -> str:
    """Extract a section number like ``3.2`` from a heading string."""
    match = re.match(r"^[\s]*(\d+(?:\.\d+)*)", heading)
    return match.group(1) if match else ""


def _extract_number(text: str) -> str:
    """Extract the first integer from *text*."""
    match = re.search(r"\b(\d+)\b", text)
    return match.group(1) if match else ""
