"""Programmatic entity registry for kompend Tier 3.

Accumulates entities across ChunkTrees using pattern-based extraction (no LLM).
Deduplicates acronym/expanded-form pairs (e.g. ``SNR`` and
``Signal-to-Noise Ratio`` resolve to a single record).

Extraction sources:
    - Acronym definitions: ``Full Name (ACRONYM)`` or ``ACRONYM (Full Name)``
    - Definition sentences: ``X is defined as Y``, ``X refers to Y``
    - Structured data: equation variables, table column headers
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kompend.core.models import ChunkTree, CrossReferenceIssue, StructuredData, StructuredEquation, StructuredTable
from kompend.logging import get_logger

# ── Regex patterns ──────────────────────────────────────────────────────

# "Signal-to-Noise Ratio (SNR)" or "signal to noise ratio (SNR)"
_ACRONYM_IN_PARENS = re.compile(
    r"([A-Z][\w]+(?:[\s\-][A-Z][\w]+)+)\s*\(([A-Z]{2,8})\)",
)
# "SNR (Signal-to-Noise Ratio)"
_REVERSE_ACRONYM = re.compile(
    r"\b([A-Z]{2,8})\s*\(([A-Z][\w]+(?:[\s\-][A-Za-z\w]+)+)\)",
)
# "X is defined as Y", "X is referred to as Y", "X is known as Y"
_DEFINITION_VERB = re.compile(
    r"\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)"
    r"\s+(?:is|are)\s+"
    r"(?:defined\s+as|referred\s+to\s+as|known\s+as|called)\s+"
    r"(.{5,300}?)[\.;\n]",
)
# "X means Y"
_MEANS_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)\s+means\s+(.{5,300}?)[\.;\n]",
)
# "X (also known as Y)" or "X also known as Y."
_ALSO_KNOWN_AS = re.compile(
    r"([A-Z][A-Za-z\s\-]{2,50}?)\s*\(?(?:also\s+known\s+as|aka|a\.k\.a\.)\s+"
    r"([A-Z][A-Za-z\s\-]+?)(?:\)|[.;!?\n]|$)",
)

logger = get_logger(__name__)


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class EntityLocation:
    """Where an entity was found in the document tree."""

    source_file: str
    heading_chain: tuple[str, ...]


@dataclass
class EntityRecord:
    """A single entity accumulated across the document tree."""

    name: str
    canonical_name: str
    definition: str = ""
    locations: list[EntityLocation] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    entity_type: str = ""
    all_definitions: list[str] = field(default_factory=list)

    @property
    def all_names(self) -> list[str]:
        """Every name (canonical, original, aliases) for this entity."""
        seen: set[str] = set()
        result: list[str] = []
        for n in (self.canonical_name, self.name, *self.aliases):
            key = n.lower()
            if key not in seen:
                seen.add(key)
                result.append(n)
        return result


# ── Registry ────────────────────────────────────────────────────────────


class EntityRegistry:
    """Accumulates entities across ChunkTrees with deduplication.

    All name lookups are case-insensitive. When an acronym and its
    expanded form are both registered, they resolve to the same record.
    """

    def __init__(self) -> None:
        self._records: list[EntityRecord] = []
        self._lookup: dict[str, int] = {}

    # ── Public API ──────────────────────────────────────────────────

    @property
    def entities(self) -> list[EntityRecord]:
        return list(self._records)

    @property
    def entity_count(self) -> int:
        return len(self._records)

    def query(self, name: str) -> EntityRecord | None:
        """Look up an entity by any of its names (case-insensitive)."""
        idx = self._lookup.get(name.lower().strip())
        return self._records[idx] if idx is not None else None

    def register(
        self,
        name: str,
        *,
        canonical_name: str = "",
        definition: str = "",
        location: EntityLocation | None = None,
        entity_type: str = "",
        aliases: list[str] | None = None,
    ) -> EntityRecord:
        """Register or update an entity.

        If the entity (by canonical name, name, or any alias) already
        exists, the existing record is updated with new information.
        """
        canon = (canonical_name or name).strip()
        key = self._key(canon)

        existing_idx = self._lookup.get(key)
        if existing_idx is None:
            existing_idx = self._lookup.get(self._key(name))

        if existing_idx is not None:
            return self._merge(existing_idx, definition, location, aliases)

        record = EntityRecord(
            name=name.strip(),
            canonical_name=canon,
            entity_type=entity_type,
        )
        if definition:
            record.definition = definition.strip()
            record.all_definitions.append(definition.strip())
        if location:
            record.locations.append(location)
        if aliases:
            for a in aliases:
                a_stripped = a.strip()
                if a_stripped and a_stripped.lower() not in {al.lower() for al in record.aliases}:
                    record.aliases.append(a_stripped)

        idx = len(self._records)
        self._records.append(record)
        self._index_record(idx, record)
        return record

    def find_conflicts(self) -> list[CrossReferenceIssue]:
        """Flag entities with same name but divergent definitions."""
        issues: list[CrossReferenceIssue] = []
        for record in self._records:
            unique_defs = {d.lower().strip() for d in record.all_definitions if d}
            if len(unique_defs) > 1:
                locations = [f"{loc.source_file}:{'/'.join(loc.heading_chain)}" for loc in record.locations]
                issues.append(
                    CrossReferenceIssue(
                        issue_type="conflict",
                        description=(f'"{record.canonical_name}" has {len(record.all_definitions)} divergent definitions'),
                        entity=record.canonical_name,
                        locations=locations,
                    )
                )
        return issues

    def find_duplicates(self) -> list[CrossReferenceIssue]:
        """Flag records that are likely the same entity under different names.

        Detects when two records share an alias or when one record's
        canonical name is another's alias.
        """
        issues: list[CrossReferenceIssue] = []
        seen_pairs: set[frozenset[str]] = set()

        for i, record_a in enumerate(self._records):
            names_a = {n.lower() for n in record_a.all_names}
            for record_b in self._records[i + 1 :]:
                names_b = {n.lower() for n in record_b.all_names}
                overlap = names_a & names_b
                if overlap and record_a.canonical_name.lower() != record_b.canonical_name.lower():
                    pair_key = frozenset({record_a.canonical_name.lower(), record_b.canonical_name.lower()})
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        issues.append(
                            CrossReferenceIssue(
                                issue_type="duplicate",
                                description=(
                                    f'"{record_a.canonical_name}" and "{record_b.canonical_name}" '
                                    f"may refer to the same entity (shared: {', '.join(sorted(overlap))})"
                                ),
                                entity=record_a.canonical_name,
                                locations=[f"{loc.source_file}:{'/'.join(loc.heading_chain)}" for loc in record_a.locations + record_b.locations],
                            )
                        )
        return issues

    def scan_tree(self, tree: ChunkTree) -> None:
        """Extract entities from all nodes in a ChunkTree."""
        source_file = str(tree.source_file) if tree.source_file else "unknown"
        for node in tree.walk():
            heading_chain = tuple(node.heading_chain)
            location = EntityLocation(source_file=source_file, heading_chain=heading_chain)

            if node.chunk and node.chunk.text:
                self._extract_from_text(node.chunk.text, location)

            if node.structured:
                self._extract_from_structured(node.structured, location)

    def _index_record(self, idx: int, record: EntityRecord) -> None:
        """Populate lookup dict for all names of a record."""
        for name in record.all_names:
            self._lookup.setdefault(self._key(name), idx)

    def _merge(
        self,
        idx: int,
        definition: str,
        location: EntityLocation | None,
        aliases: list[str] | None,
    ) -> EntityRecord:
        record = self._records[idx]
        if definition:
            def_stripped = definition.strip()
            if def_stripped and def_stripped.lower() not in {d.lower() for d in record.all_definitions}:
                record.all_definitions.append(def_stripped)
                if not record.definition:
                    record.definition = def_stripped
        if location and location not in record.locations:
            record.locations.append(location)
        if aliases:
            for a in aliases:
                a_stripped = a.strip()
                if a_stripped and self._key(a_stripped) not in self._lookup:
                    record.aliases.append(a_stripped)
                    self._lookup[self._key(a_stripped)] = idx
        return record

    def _extract_from_text(self, text: str, location: EntityLocation) -> None:
        """Run all pattern extractors on a text block."""
        self._extract_acronyms(text, location)
        self._extract_definitions(text, location)
        self._extract_also_known_as(text, location)

    def _extract_acronyms(self, text: str, location: EntityLocation) -> None:
        """Extract ``Full Name (ACRONYM)`` and reverse patterns."""
        for match in _ACRONYM_IN_PARENS.finditer(text):
            full_name = match.group(1).strip()
            acronym = match.group(2).strip()
            self.register(
                full_name,
                canonical_name=full_name,
                aliases=[acronym],
                location=location,
                entity_type="term",
            )

        for match in _REVERSE_ACRONYM.finditer(text):
            acronym = match.group(1).strip()
            full_name = match.group(2).strip()
            self.register(
                full_name,
                canonical_name=full_name,
                aliases=[acronym],
                location=location,
                entity_type="term",
            )

    def _extract_definitions(self, text: str, location: EntityLocation) -> None:
        """Extract ``X is defined as Y`` and ``X means Y`` patterns."""
        for match in _DEFINITION_VERB.finditer(text):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            self.register(
                term,
                canonical_name=term,
                definition=definition,
                location=location,
                entity_type="term",
            )

        for match in _MEANS_PATTERN.finditer(text):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            self.register(
                term,
                canonical_name=term,
                definition=definition,
                location=location,
                entity_type="term",
            )

    def _extract_also_known_as(self, text: str, location: EntityLocation) -> None:
        """Extract ``X (also known as Y)`` patterns."""
        for match in _ALSO_KNOWN_AS.finditer(text):
            name_a = match.group(1).strip()
            name_b = match.group(2).strip()
            self.register(
                name_a,
                canonical_name=name_a,
                aliases=[name_b],
                location=location,
                entity_type="term",
            )

    def _extract_from_structured(self, structured: StructuredData, location: EntityLocation) -> None:
        """Extract entities from StructuredData (equation variables, table columns)."""
        if isinstance(structured, StructuredEquation):
            for var in structured.variables:
                self.register(
                    var.name,
                    canonical_name=var.name,
                    definition=var.definition,
                    location=location,
                    entity_type="variable",
                )

        elif isinstance(structured, StructuredTable):
            for col in structured.columns:
                col_clean = col.strip()
                if col_clean and len(col_clean) > 1:
                    self.register(
                        col_clean,
                        canonical_name=col_clean,
                        location=location,
                        entity_type="column",
                    )

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _key(name: str) -> str:
        return name.lower().strip()
