from __future__ import annotations

from kompend.compilation._two_phase_core import Extraction
from kompend.compilation.schema import Concept, Entity

# ── Memory digests ─────────────────────────────────────────────────────


class ConceptDigest:
    """One concept in extraction memory, with ref_count for compaction."""

    __slots__ = ("title", "definition", "ref_count")

    def __init__(self, title: str, definition: str = "") -> None:
        self.title = title
        self.definition = definition
        self.ref_count = 1

    def __repr__(self) -> str:
        return f"ConceptDigest(title={self.title!r}, ref_count={self.ref_count})"


class EntityDigest:
    """One entity in extraction memory, with ref_count for compaction."""

    __slots__ = ("name", "entity_type", "context", "ref_count")

    def __init__(self, name: str, entity_type: str = "", context: str = "") -> None:
        self.name = name
        self.entity_type = entity_type
        self.context = context
        self.ref_count = 1

    def __repr__(self) -> str:
        return f"EntityDigest(name={self.name!r}, ref_count={self.ref_count})"


# ── Memory core ────────────────────────────────────────────────────────


class ExtractionMemory:
    """Structured cross-section memory for agentic extraction.

    Accumulates concepts, entities, and tags as an agent reads a document
    section by section. The digest is handed to the model in full; there is no
    pruning heuristic.
    """

    def __init__(self) -> None:
        self.concepts: dict[str, ConceptDigest] = {}
        self.entities: dict[str, EntityDigest] = {}
        self.tags: set[str] = set()
        self.section_summaries: list[str] = []
        self._last_section_titles: list[str] = []

    # ── Ingestion ──────────────────────────────────────────────────

    def ingest(self, extraction: Extraction) -> None:
        """Merge a section's extraction into memory, dedup by title/name.

        Parameters
        ----------
        extraction: Extraction
            Per-section extraction result with ``concepts``, ``entities``,
            ``tags``, and ``summary`` attributes.
        """
        for c in extraction.concepts:
            key = self._norm(c.name)
            if not key:
                continue
            if key in self.concepts:
                self.concepts[key].ref_count += 1
                # overwrite definition with latest occurrence
                if c.definition:
                    self.concepts[key].definition = c.definition
            else:
                self.concepts[key] = ConceptDigest(
                    title=c.name,
                    definition=c.definition,
                )

        for e in extraction.entities:
            key = self._norm(e.name)
            if not key:
                continue
            if key in self.entities:
                self.entities[key].ref_count += 1
                # update context if richer
                ctx = e.context
                if ctx and len(ctx) > len(self.entities[key].context):
                    self.entities[key].context = ctx
            else:
                self.entities[key] = EntityDigest(
                    name=e.name,
                    entity_type=e.type,
                    context=e.context,
                )

        for t in getattr(extraction, "tags", ()):
            if t:
                self.tags.add(t)

        sec_summary = getattr(extraction, "summary", "")
        if sec_summary:
            self.section_summaries.append(sec_summary)

    # ── Summarization ──────────────────────────────────────────────

    def summarize(self) -> str:
        """Render current memory as a PydanticAI-suitable context string.

        The digest is a structured text block that
        the agent can reason about — sorted by ref_count descending so
        that recurring concepts appear first.
        """
        lines: list[str] = []

        if self.section_summaries:
            line = "Section summary: " + " | ".join(self.section_summaries)
            lines.append(line)

        if self.concepts:
            concepts_sorted = sorted(self.concepts.values(), key=lambda d: d.ref_count, reverse=True)
            parts: list[str] = []
            for d in concepts_sorted:
                label = d.title
                if d.definition:
                    label += f": {d.definition}"
                parts.append(label)
            lines.append("Concepts: " + "; ".join(parts))

        if self.entities:
            entities_sorted = sorted(self.entities.values(), key=lambda d: d.ref_count, reverse=True)
            eparts: list[str] = []
            for d in entities_sorted:
                label = d.name
                if d.entity_type:
                    label += f" ({d.entity_type})"
                if d.context:
                    label += f" — {d.context}"
                eparts.append(label)
            lines.append("Entities: " + "; ".join(eparts))

        if self.tags:
            lines.append("Tags: " + ", ".join(sorted(self.tags)))

        return "\n".join(lines)

    # ── Finalization ───────────────────────────────────────────────

    def to_extraction(self, source: str) -> Extraction:
        """Derive a final ``Extraction`` from accumulated memory.

        Concepts/entities are ordered by ref_count descending (most
        referenced first). Tags are the full tag set. Summary is the
        first section summary (if any), or empty.
        """
        return Extraction(
            source=source,
            summary=self.section_summaries[0] if self.section_summaries else "",
            concepts=[
                Concept(name=d.title, definition=d.definition)
                for d in sorted(
                    self.concepts.values(),
                    key=lambda d: d.ref_count,
                    reverse=True,
                )
            ],
            entities=[
                Entity(name=d.name, type=d.entity_type, context=d.context)
                for d in sorted(
                    self.entities.values(),
                    key=lambda d: d.ref_count,
                    reverse=True,
                )
            ],
            tags=sorted(self.tags),
        )

    def __len__(self) -> int:
        return len(self.concepts) + len(self.entities)

    def __bool__(self) -> bool:
        return bool(self.concepts or self.entities or self.tags)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _norm(title: str) -> str:
        """Normalize a title/name for dedup keys."""
        return title.strip().lower()
