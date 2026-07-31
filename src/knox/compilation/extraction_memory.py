from __future__ import annotations

from knox.compilation._two_phase_core import Extraction

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
    section by section. Supports deterministic compaction when the
    estimated token budget is tight.
    """

    def __init__(self) -> None:
        self.concepts: dict[str, ConceptDigest] = {}
        self.entities: dict[str, EntityDigest] = {}
        self.tags: set[str] = set()
        self.section_summaries: list[str] = []
        self._last_section_titles: list[str] = []
        self._total_additions = 0

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
            key = self._norm(c.get("name", c.get("title", "")))
            if not key:
                continue
            if key in self.concepts:
                self.concepts[key].ref_count += 1
                # overwrite definition with latest occurrence
                new_def = c.get("definition", "")
                if new_def:
                    self.concepts[key].definition = new_def
            else:
                self.concepts[key] = ConceptDigest(
                    title=c.get("name", c.get("title", "")),
                    definition=c.get("definition", ""),
                )

        for e in extraction.entities:
            key = self._norm(e.get("name", ""))
            if not key:
                continue
            if key in self.entities:
                self.entities[key].ref_count += 1
                # update context if richer
                ctx = e.get("context", "")
                if ctx and len(ctx) > len(self.entities[key].context):
                    self.entities[key].context = ctx
            else:
                self.entities[key] = EntityDigest(
                    name=e.get("name", ""),
                    entity_type=e.get("type", ""),
                    context=e.get("context", ""),
                )

        for t in getattr(extraction, "tags", ()):
            if t:
                self.tags.add(t)

        sec_summary = getattr(extraction, "summary", "")
        if sec_summary:
            self.section_summaries.append(sec_summary)

        self._total_additions += 1

    # ── Summarization ──────────────────────────────────────────────

    def summarize(self, max_tokens: int = 4096) -> str:
        """Render current memory as a PydanticAI-suitable context string.

        The digest is formatted as a compact, structured text block that
        the agent can reason about — sorted by ref_count descending so
        that recurring concepts appear first.
        """
        lines: list[str] = []

        if self.section_summaries:
            line = "Section summary: " + " | ".join(s[:120] for s in self.section_summaries)
            lines.append(line)

        if self.concepts:
            concepts_sorted = sorted(self.concepts.values(), key=lambda d: d.ref_count, reverse=True)
            parts: list[str] = []
            for d in concepts_sorted:
                label = d.title
                if d.definition:
                    label += f": {d.definition[:100]}"
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
                    label += f" — {d.context[:80]}"
                eparts.append(label)
            lines.append("Entities: " + "; ".join(eparts))

        if self.tags:
            lines.append("Tags: " + ", ".join(sorted(self.tags)))

        result = "\n".join(lines)

        # Rough token estimate: ~4 chars per token
        if len(result) <= max_tokens * 4:
            return result
        # Truncate from the end (tag list is cheapest to lose)
        return result[: max_tokens * 4] + "\n[memory truncated]"

    # ── Compaction ─────────────────────────────────────────────────

    def compact(self) -> None:
        """Deterministic prune of low-value entries.

        - Keeps only the last 3 section summaries.
        - Drops concepts/entities with ref_count == 1.
        - For remaining digests, keeps only the title/name
          (definition/context dropped) to save tokens.
        """
        # Trim section summaries
        if len(self.section_summaries) > 3:
            self.section_summaries = self.section_summaries[-3:]

        # Drop low-ref concepts
        self.concepts = {k: v for k, v in self.concepts.items() if v.ref_count > 1}
        # Strip definitions
        for d in self.concepts.values():
            d.definition = ""

        # Drop low-ref entities
        self.entities = {k: v for k, v in self.entities.items() if v.ref_count > 1}
        # Strip context
        for d in self.entities.values():
            d.context = ""

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
                {"name": d.title, "definition": d.definition}
                for d in sorted(
                    self.concepts.values(),
                    key=lambda d: d.ref_count,
                    reverse=True,
                )
            ],
            entities=[
                {
                    "name": d.name,
                    "type": d.entity_type,
                    "context": d.context,
                }
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

    def _estimate_tokens(self) -> int:
        """Rough token estimate based on accumulated content.

        Used by the agentic flow to decide when compaction is needed.
        This is intentionally a fast approximation (~4 chars/token).
        """
        total_chars = 0
        for d in self.concepts.values():
            total_chars += len(d.title) + len(d.definition)
        for d in self.entities.values():
            total_chars += len(d.name) + len(d.entity_type) + len(d.context)
        total_chars += sum(len(s) for s in self.section_summaries)
        total_chars += len(" ".join(self.tags))
        if not total_chars:
            return 0
        return total_chars // 4

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _norm(title: str) -> str:
        """Normalize a title/name for dedup keys."""
        return title.strip().lower()
