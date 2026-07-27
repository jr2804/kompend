"""Internal helpers for the N3 two-phase compiler.

This module is private API. External callers should import through
``knox.compilation.two_phase`` for backward compatibility.
"""

from __future__ import annotations

import functools
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from slugify import slugify as _slugify_impl

from knox.agents.deps import GenerationDeps, PipelineAgentDeps
from knox.compilation.concept_keys import (
    is_structural_concept,
    normalize_concept_key,
)
from knox.compilation.schema import LLMOutputError
from knox.core.models import ElementType, ResolvedChunk, WikiNode
from knox.core.templates import get_template_loader
from knox.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()

# Maximum slug length. Windows limits a single path component to 255 chars;
# the LLM sometimes emits a full sentence as a page title, which slugifies to
# 300+ chars and crashes ``write_text`` with ``OSError: [Errno 22]``. Cap the
# slug and suffix a short hash of the full slug so distinct long titles stay
# distinct (avoids _upsert_node merging two different concepts into one).
# Default slug max length. Configurable via ``compile.llm.slug_max_length``.
_SLUG_MAX_LENGTH: int = 80

# Default failure thresholds. Configurable via ``compile.llm.batch_failure_threshold_ratio``
# and ``compile.llm.batch_failure_threshold_min``.
_FAILURE_THRESHOLD_RATIO: float = 0.5
_FAILURE_THRESHOLD_MIN: int = 5

# Default minimum input tokens per generation segment.
# Configurable via ``compile.llm.min_segment_tokens``.
MIN_SEGMENT_TOKENS: int = 512

# Default chars-per-token heuristic. Configurable via ``compile.llm.default_chars_per_token``
# and the ``KNOX_CHARS_PER_TOKEN`` env var.
_DEFAULT_CHARS_PER_TOKEN: float = 3.85


@functools.cache
def _get_chars_per_token(*, default: float | None = None) -> float:
    """Return the characters-per-token heuristic used for budget estimation.

    Priority: ``KNOX_CHARS_PER_TOKEN`` env var > *default* argument >
    ``_DEFAULT_CHARS_PER_TOKEN`` module constant > ``compile.llm.default_chars_per_token`` config.
    """
    env_raw = os.environ.get("KNOX_CHARS_PER_TOKEN")
    if env_raw is not None:
        try:
            return float(env_raw)
        except ValueError:
            logger.warning("Invalid KNOX_CHARS_PER_TOKEN value %r; using fallback", env_raw)
    return default if default is not None else _DEFAULT_CHARS_PER_TOKEN


def tokens_to_chars(tokens: int, *, chars_per_token: float | None = None) -> int:
    """Convert token count to approximate character count.

    Args:
        tokens: Token count to convert.
        chars_per_token: Optional override (e.g. from ``compile.llm.default_chars_per_token``).
    """
    return int(tokens * _get_chars_per_token(default=chars_per_token))


def chars_to_tokens(chars: int, *, chars_per_token: float | None = None) -> int:
    """Convert character count to approximate token count.

    Args:
        chars: Character count to convert.
        chars_per_token: Optional override (e.g. from ``compile.llm.default_chars_per_token``).
    """
    return max(1, int(chars / _get_chars_per_token(default=chars_per_token)))


def slugify(text: str, max_length: int | None = None) -> str:
    """Convert a title to a lowercase hyphenated slug (capped for filesystem safety).

    When the slug exceeds *max_length* (defaults to ``compile.llm.slug_max_length``),
    it is truncated at a word boundary and a short hash of the *full* slug is
    appended so two distinct long titles don't collide (which would otherwise be
    merged by :func:`_upsert_node`).

    Args:
        text: The title to slugify.
        max_length: Maximum slug length. Defaults to ``_SLUG_MAX_LENGTH`` (80),
            which is the fallback for ``compile.llm.slug_max_length``.
    """
    cap = max_length if max_length is not None else _SLUG_MAX_LENGTH
    slug = _slugify_impl(text) or "concept"
    if len(slug) <= cap:
        return slug
    truncated = slug[: cap - 5].rstrip("-")
    return f"{truncated}-{hashlib.sha256(slug.encode()).hexdigest()[:4]}"


def _dedup_id(node_id: str, seen: set[str]) -> str:
    """Append a numeric suffix if node_id already exists in *seen*, then add."""
    candidate = node_id
    counter = 2
    while candidate in seen:
        candidate = f"{node_id}-{counter}"
        counter += 1
    seen.add(candidate)
    return candidate


def extract_images_from_chunks(chunks: list[ResolvedChunk]) -> list[dict[str, Any]]:
    """Extract image metadata from IMAGE chunks.

    Returns a list of dicts with keys: path, caption, element_type.
    Images are collected from:
    - chunk.metadata["image_path"] (file mode)
    - chunk.metadata["image_base64"] (base64 mode)
    - VLM captions from resolved.entities with type="image_description"
    """
    images: list[dict[str, Any]] = []

    for rc in chunks:
        if ElementType.IMAGE not in rc.chunk.element_types:
            continue

        img_ref = rc.chunk.metadata.get("image_path") or rc.chunk.metadata.get("image_base64")
        if not img_ref:
            # Try resolved metadata
            img_ref = rc.metadata.get("image_path") or rc.metadata.get("image_base64")

        if not img_ref:
            continue

        caption = ""
        for entity in rc.entities:
            if entity.get("type") == "image_description" and entity.get("context"):
                caption = entity["context"]
                break

        images.append(
            {
                "path": img_ref,
                "caption": caption,
                "element_type": "image",
            }
        )

    return images


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown text into sections keyed by heading.

    Content before the first heading is stored under the empty-string key.
    """
    sections: dict[str, str] = {}
    current_heading = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("## "):
            if current_lines:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = stripped[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _merge_sections(existing: dict[str, str], incoming: dict[str, str]) -> str:
    """Merge two section dicts, keeping the richer version per heading."""
    merged: dict[str, str] = {}
    for heading in dict.fromkeys(list(existing) + list(incoming)):
        ex = existing.get(heading, "")
        inc = incoming.get(heading, "")
        merged[heading] = ex if len(ex) >= len(inc) else inc

    result = ""
    for heading, body in merged.items():
        if heading:
            result += f"## {heading}\n{body}\n\n"
        elif body:
            result += f"{body}\n\n"
    return result.strip()


def compute_quality_score(nodes: list[WikiNode], edges: list[dict[str, Any]]) -> float:
    """Compute a 0–100 quality score for a compiled wiki.

    Weighted average of three sub-scores:
    - **Completeness** (40 %): fraction of nodes with non-trivial content (>50 chars)
    - **Connectivity** (30 %): edge density relative to a target of ~3 edges/node
    - **Richness** (30 %): average content length normalised against a 500-char target
    """
    n = len(nodes)
    if n == 0:
        return 0.0

    non_empty = sum(1 for node in nodes if len((node.content or "").strip()) > 50)
    completeness = non_empty / n

    target_edges = n * 3
    connectivity = min(len(edges) / target_edges, 1.0) if target_edges > 0 else 0.0

    avg_len = sum(len(node.content or "") for node in nodes) / n
    richness = min(avg_len / 500.0, 1.0)

    return round((completeness * 0.4 + connectivity * 0.3 + richness * 0.3) * 100, 1)


def _merge_node(
    existing: WikiNode,
    incoming: WikiNode,
) -> WikiNode:
    """Merge *incoming* node into *existing* using section-aware dedup.

    Sections with the same heading are de-duplicated, keeping the longer
    (richer) version.  Sections unique to either source are appended.
    Provenance lists are unioned.
    """
    existing_text = existing.content or ""
    incoming_text = incoming.content or ""

    if not existing_text.strip():
        merged_content = incoming_text
    elif not incoming_text.strip():
        merged_content = existing_text
    else:
        ex_sections = _split_sections(existing_text)
        inc_sections = _split_sections(incoming_text)
        merged_content = _merge_sections(ex_sections, inc_sections)

    merged_title = existing.title or incoming.title

    # Merge provenance
    # ponytail: LLM output may return dicts/non-strings in provenance/related
    # (e.g. {"chunk": ...} under timeout stress); coerce to str so the set
    # dedup never hits "unhashable type: 'dict'".
    existing_prov = {str(x) for x in (existing.metadata.get("provenance") or [])}
    incoming_prov = {str(x) for x in (incoming.metadata.get("provenance") or [])}
    merged_prov = sorted(existing_prov | incoming_prov)

    # Merge related
    existing_rel = {str(x) for x in (existing.metadata.get("related") or [])}
    incoming_rel = {str(x) for x in (incoming.metadata.get("related") or [])}
    merged_rel = sorted(existing_rel | incoming_rel)

    return WikiNode(
        id=existing.id,
        title=merged_title,
        content=merged_content,
        node_type=existing.node_type,
        source_chunks=list(
            dict.fromkeys(
                (existing.source_chunks or []) + (incoming.source_chunks or []),
            ),
        ),
        metadata={
            **existing.metadata,
            "provenance": merged_prov,
            "related": merged_rel,
        },
    )


@dataclass
class Extraction:
    """Phase-1 extraction result for a single source."""

    source: str
    summary: str = ""
    entities: list[dict[str, str]] = field(default_factory=list)
    concepts: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary,
            "entities": list(self.entities),
            "concepts": list(self.concepts),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Extraction:
        return cls(
            source=data.get("source", ""),
            summary=data.get("summary", ""),
            entities=list(data.get("entities", [])),
            concepts=list(data.get("concepts", [])),
            tags=list(data.get("tags", [])),
        )


_CONCEPT_NAME_KEYS = ("name", "concept", "title", "term", "label")

# Internal names injected by VLMEnricher for bookkeeping — must never
# become concept pages.
_RESERVED_CONCEPT_NAMES: frozenset[str] = frozenset({"vlm_filtered", "vlm_skipped", "vlm_error"})

# Structural-concept filtering lives in :mod:`knox.compilation.concept_keys`
# so the ``wicer`` (GraphCompiler) and TwoPhase backends share one
# implementation. Re-exported here for callers that already import it
# from this module.


def _concept_name(concept: dict[str, Any]) -> str:
    """Salvage a concept name, tolerating schema drift.

    The extraction prompt asks for ``name``; LLMs sometimes return
    ``concept``/``title``/``term`` instead. Without salvage, concept_pages()
    returns empty and generation silently falls back.
    """
    for key in _CONCEPT_NAME_KEYS:
        val = str(concept.get(key, "")).strip()
        if val:
            return val
    return ""


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
                name = _concept_name(concept)
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
                definition = concept.get("definition") or concept.get("importance") or concept.get("description")
                if definition:
                    bucket["definitions"].append(definition)
        return list(merged.values())

    def entity_index(self) -> dict[str, list[str]]:
        """Return ``entity_name -> [source, ...]``."""
        out: dict[str, list[str]] = defaultdict(list)
        for ext in self.extractions:
            for ent in ext.entities:
                name = ent.get("name", "").strip()
                if name:
                    out[name].append(ext.source)
        return dict(out)


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


def try_pydanticai_extraction(
    agent: Any,
    *,
    text: str,
    source_id: str,
    timeout: float = 120.0,
) -> Any | None:
    """Run a PydanticAI :class:`ExtractionAgent` on a chunk of source text.

    This is the agent-based extraction entry point. Callers invoke it
    for every extraction batch (it is the only path, not opt-in).

    The agent's ``run`` method already does JSON extraction and Pydantic
    validation internally; auto-retry on schema mismatch is built in.
    We catch and log any exception here so the caller's failure
    tracker can record the attempt uniformly with the pydantic-ai path.

    Args:
        agent: An instantiated :class:`~knox.agents.ExtractionAgent`.
        text: The combined chunk text to extract from.
        source_id: Provenance tag (file stem / URL).
        timeout: Per-call timeout in seconds; forwarded to the
            agent's model settings.

    Returns:
        The validated :class:`~knox.compilation.schema.ExtractionOutput`
        instance, or ``None`` on failure.
    """
    try:
        deps = PipelineAgentDeps(text=text, source_id=source_id, timeout=timeout)
        return agent.run(deps)
    except LLMOutputError as exc:
        logger.warning("PydanticAI extraction failed for '%s': %s", source_id, exc)
        return None
    except Exception as exc:  # ponytail: PydanticAI raises many subclassed errors; treat uniformly as agent failure
        logger.warning("PydanticAI extraction unexpected error for '%s': %s", source_id, exc)
        return None


def try_pydanticai_generation(
    agent: Any,
    *,
    concepts_json: str,
    content: str,
    source_id: str,
    pinned_facts: list[str] | None = None,
    timeout: float = 120.0,
) -> Any | None:
    """Run a PydanticAI :class:`GenerationAgent` on a concept batch.

    Agent-based generation entry point (the only path).

    Args:
        agent: An instantiated :class:`~knox.agents.GenerationAgent`.
        concepts_json: The concept batch encoded via
            :func:`knox.llm.format.encode_for_llm`.
        content: Combined source-content block.
        source_id: Provenance / log label for the batch.
        pinned_facts: Facts that must appear in the output.
        timeout: Per-call timeout in seconds.

    Returns:
        The validated :class:`~knox.compilation.schema.GenerationOutput`
        instance, or ``None`` on failure.
    """
    try:
        deps = GenerationDeps(
            concepts_json=concepts_json,
            content=content,
            source_id=source_id,
            pinned_facts=pinned_facts or [],
            timeout=timeout,
        )
        return agent.run(deps)
    except LLMOutputError as exc:
        logger.warning("PydanticAI generation failed for '%s': %s", source_id, exc)
        return None
    except Exception as exc:  # ponytail: PydanticAI raises many subclassed errors; treat uniformly as agent failure
        logger.warning("PydanticAI generation unexpected error for '%s': %s", source_id, exc)
        return None


def merge_extractions(source: str, parts: list[Extraction]) -> Extraction:
    """Merge multiple batch extractions for the same source into one."""
    if not parts:
        return Extraction(source=source)
    if len(parts) == 1:
        return parts[0]

    summary = " ".join(p.summary for p in parts if p.summary)
    entities: list[dict[str, str]] = []
    concepts: list[dict[str, str]] = []
    tags: list[str] = []
    seen_entities: set[str] = set()
    seen_concepts: set[str] = set()
    seen_tags: set[str] = set()
    for p in parts:
        for e in p.entities:
            key = e.get("name", "").strip().lower()
            if key and key not in seen_entities:
                entities.append(e)
                seen_entities.add(key)
        for c in p.concepts:
            key = c.get("name", "").strip().lower()
            if key and key not in seen_concepts:
                concepts.append(c)
                seen_concepts.add(key)
        for t in p.tags:
            if t not in seen_tags:
                tags.append(t)
                seen_tags.add(t)
    return Extraction(source=source, summary=summary, entities=entities, concepts=concepts, tags=tags)


def segment_sources(
    sources: list[tuple[str, str]],
    segment_size: int,
) -> list[list[tuple[str, str]]]:
    """Split source content into segments of at most *segment_size* chars.

    Returns a list of segments, where each segment is a list of
    ``(label, content)`` pairs.  No data is dropped — sources that
    exceed *segment_size* are split across multiple segments.
    """
    if not sources:
        return []

    segments: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_len = 0

    for label, content in sources:
        remaining = content
        while remaining:
            space = segment_size - current_len
            if space <= 0:
                segments.append(current)
                current = []
                current_len = 0
                space = segment_size
            if len(remaining) <= space:
                current.append((label, remaining))
                current_len += len(remaining)
                remaining = ""
            else:
                # Split at the last newline within budget, or hard-cut.
                cut = remaining.rfind("\n", 0, space)
                if cut <= 0:
                    cut = space
                current.append((label, remaining[:cut]))
                current_len += cut
                remaining = remaining[cut:]

    if current:
        segments.append(current)
    return segments


def upsert_node(
    new_node: WikiNode,
    nodes: list[WikiNode],
    slug_to_idx: dict[str, int],
) -> None:
    """Insert *new_node* into *nodes*, or merge with an existing same-slug node."""
    slug = new_node.id
    if slug in slug_to_idx:
        idx = slug_to_idx[slug]
        nodes[idx] = _merge_node(nodes[idx], new_node)
    else:
        slug_to_idx[slug] = len(nodes)
        nodes.append(new_node)
