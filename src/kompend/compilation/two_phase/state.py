"""Extraction state for the two-phase compiler.

``Extraction`` and ``ExtractionSet`` are the phase-1 product,
``BatchFailureTracker`` owns the batch-failure policy, and
``TwoPhaseRuntime`` is the structural interface the flows call instead of
importing the compiler.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from kompend.compilation.concept_keys import is_structural_concept, normalize_concept_key
from kompend.compilation.schema import Concept, Entity
from kompend.core.cache.store import CacheStore
from kompend.core.config import BudgetConfig
from kompend.core.models import ExtractionStats, ResolvedChunk, WikiNode
from kompend.core.pipeline.assets import AssetStore
from kompend.core.pipeline.config import PipelineConfig
from kompend.llm.context_window import ContextWindow
from kompend.llm.usage import UsageTracker

# Default failure thresholds. Configurable via ``compile.llm.batch_failure_threshold_ratio``
# and ``compile.llm.batch_failure_threshold_min``.
_FAILURE_THRESHOLD_RATIO: float = 0.5
_FAILURE_THRESHOLD_MIN: int = 5

# Internal names injected by VLMEnricher for bookkeeping — must never
# become concept pages.
_RESERVED_CONCEPT_NAMES: frozenset[str] = frozenset({"vlm_filtered", "vlm_skipped", "vlm_error"})

#: Progress callback: ``(units done, units total, activity)``.
#:
#: The two-phase flows report in the units they actually work in - the
#: sections of a document, the batches of a source - so a long document
#: shows movement instead of a frozen bar. *activity* names what the
#: running call is doing ("" when there is nothing to add).
ProgressFn = Callable[[int, int, str], None]


@dataclass
class Extraction:
    """Phase-1 extraction result for a single source."""

    source: str
    summary: str = ""
    entities: list[Entity] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary,
            "entities": [e.model_dump() for e in self.entities],
            "concepts": [c.model_dump() for c in self.concepts],
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Extraction:
        return cls(
            source=data.get("source", ""),
            summary=data.get("summary", ""),
            entities=[Entity.model_validate(e) for e in data.get("entities", [])],
            concepts=[Concept.model_validate(c) for c in data.get("concepts", [])],
            tags=list(data.get("tags", [])),
        )


@dataclass
class ExtractionSet:
    """All extractions across the corpus."""

    extractions: list[Extraction] = field(default_factory=list)

    def add(self, extraction: Extraction) -> None:
        self.extractions.append(extraction)

    def concept_pages(self) -> list[dict[str, Any]]:
        """Return merged concept dicts with provenance per source.

        Output schema: ``{name, definition, sources: [str], definitions: [str]}``

        Concepts whose ``normalize_concept_key`` collides are merged into one.
        The canonical name is the longest variant (most descriptive).
        Concept names are salvaged tolerantly: the extraction prompt asks for
        ``name``, but LLMs sometimes return ``concept``/``title``/``term``.
        Without salvage, an otherwise-good extraction yields zero concept pages
        and the whole generation phase silently falls back.
        """
        merged: dict[str, dict[str, Any]] = {}
        for ext in self.extractions:
            for concept in ext.concepts:
                name = concept.name.strip()
                if not name or name in _RESERVED_CONCEPT_NAMES or name.startswith("vlm_"):
                    continue
                if is_structural_concept(name):
                    continue
                key = normalize_concept_key(name)
                if not key:
                    continue
                if key not in merged:
                    merged[key] = {"name": name, "sources": [], "definitions": []}
                # Keep most descriptive canonical name (longest).
                elif len(name) > len(merged[key]["name"]):
                    merged[key]["name"] = name
                bucket = merged[key]
                bucket["sources"].append(ext.source)
                definition = concept.definition or concept.importance
                if definition:
                    bucket["definitions"].append(definition)
        return list(merged.values())

    def entity_index(self) -> dict[str, list[str]]:
        """Return ``entity_name -> [source, ...]``."""
        out: dict[str, list[str]] = defaultdict(list)
        for ext in self.extractions:
            for ent in ext.entities:
                name = ent.name.strip()
                if name:
                    out[name].append(ext.source)
        return dict(out)

    def entity_pages(self) -> list[dict[str, Any]]:
        """Return merged entity page specs, keyed like :meth:`concept_pages`.

        Output schema: ``{name, kind: "entity", type, sources, definitions}``.
        Entities whose ``normalize_concept_key`` collides are merged; the
        canonical name is the longest variant. These become mandatory
        generated pages (Karpathy-style entity pages) unless the same key
        already exists as a concept.
        """
        merged: dict[str, dict[str, Any]] = {}
        for ext in self.extractions:
            for ent in ext.entities:
                name = ent.name.strip()
                if not name or name.startswith("vlm_"):
                    continue
                key = normalize_concept_key(name)
                if not key:
                    continue
                if key not in merged:
                    merged[key] = {
                        "name": name,
                        "kind": "entity",
                        "type": ent.type,
                        "sources": [],
                        "definitions": [],
                    }
                else:
                    bucket = merged[key]
                    if ent.type and not bucket["type"]:
                        bucket["type"] = ent.type
                    if len(name) > len(bucket["name"]):
                        bucket["name"] = name
                entry = merged[key]
                if ext.source not in entry["sources"]:
                    entry["sources"].append(ext.source)
                if ent.context and ent.context not in entry["definitions"]:
                    entry["definitions"].append(ent.context)
        return list(merged.values())


class BatchFailureTracker:
    """Counts successful/failed batches; signals an unsuitable model.

    A model that fails too high a fraction of batches — even after per-batch
    retries — is treated as not suitable for structured JSON compilation.
    :meth:`should_stop` returns True once the failure rate exceeds
    :data:`_FAILURE_THRESHOLD_RATIO` *and* enough batches have run for the
    rate to be meaningful.
    """

    __slots__ = ("total", "failed", "_ratio", "_min_batches")

    def __init__(
        self,
        *,
        ratio: float | None = None,
        min_batches: int | None = None,
    ) -> None:
        self.total = 0
        self.failed = 0
        # Fall back to module defaults when no config value is passed.
        # Callers should pass ``compile.llm.batch_failure_threshold_ratio``
        # and ``compile.llm.batch_failure_threshold_min``.
        self._ratio = _FAILURE_THRESHOLD_RATIO if ratio is None else ratio
        self._min_batches = _FAILURE_THRESHOLD_MIN if min_batches is None else min_batches

    def record(self, *, failed: bool) -> None:
        self.total += 1
        if failed:
            self.failed += 1

    def should_stop(self) -> bool:
        """True once the model's failure rate clears the stop threshold."""
        return self.total >= self._min_batches and self.failed / self.total > self._ratio


class TwoPhaseRuntime(Protocol):
    """Structural interface for the compiler passed to the two-phase flows.

    Flow modules (``flows/extraction/``, ``flows/generation/``,
    ``flows/agentic.py``, ``flows/references.py``) receive the compiler as
    a parameter but cannot import :class:`TwoPhaseCompiler` directly
    (circular import — ``two_phase/compiler.py`` imports the flows). This Protocol
    breaks the cycle: flows type-hint against the abstraction, and
    :class:`TwoPhaseCompiler` satisfies it structurally.

    Only the attributes and methods that the flows actually access are
    declared.
    """

    # ── Compiler state accessed by the flows ─────────────────────────
    _config: PipelineConfig
    _budget: BudgetConfig
    _timeout: float
    _model: str
    _cache_store: CacheStore
    _context_window: ContextWindow
    _batch_tracker: BatchFailureTracker
    _extraction_stats: list[ExtractionStats]
    _asset_store: AssetStore | None
    _active_gen_deps: dict[str, str]
    _extract_system_tokens: int
    _gen_system_tokens: int
    _llm_stats: list[tuple[int, int]]
    _usage_tracker: UsageTracker

    def merge_duplicate_nodes(
        self,
        nodes: list[WikiNode],
        embedding_model: str,
        *,
        threshold: float,
    ) -> list[WikiNode]: ...

    def filter_empty_nodes(self, nodes: list[WikiNode], *, min_content_length: int) -> list[WikiNode]: ...

    def dedup_content_duplicate_nodes(self, nodes: list[WikiNode]) -> list[WikiNode]: ...

    def disambiguate_duplicate_titles(self, nodes: list[WikiNode], chunks: list[ResolvedChunk]) -> None: ...

    @staticmethod
    def merge_by_concept_key(nodes: list[WikiNode]) -> list[WikiNode]: ...

    def _clamp_window(self, max_input: int, *, reason: str) -> bool: ...

    # ── Methods the flows call ───────────────────────────────────────
    def _reduce_extractions(self, source: str, parts: list[Extraction]) -> Extraction: ...

    def _report(self, current: int, total: int, description: str) -> None: ...


def merge_extractions(source: str, parts: list[Extraction]) -> Extraction:
    """Merge multiple batch extractions for the same source into one."""
    if not parts:
        return Extraction(source=source)
    if len(parts) == 1:
        return parts[0]

    summary = " ".join(p.summary for p in parts if p.summary)
    entities: list[Entity] = []
    concepts: list[Concept] = []
    tags: list[str] = []
    seen_entities: set[str] = set()
    seen_concepts: set[str] = set()
    seen_tags: set[str] = set()
    for p in parts:
        for e in p.entities:
            key = e.name.strip().lower()
            if key and key not in seen_entities:
                entities.append(e)
                seen_entities.add(key)
        for c in p.concepts:
            key = c.name.strip().lower()
            if key and key not in seen_concepts:
                concepts.append(c)
                seen_concepts.add(key)
        for t in p.tags:
            if t not in seen_tags:
                tags.append(t)
                seen_tags.add(t)
    return Extraction(source=source, summary=summary, entities=entities, concepts=concepts, tags=tags)
