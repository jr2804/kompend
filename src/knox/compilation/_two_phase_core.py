"""Internal helpers for the N3 two-phase compiler.

This module is private API. External callers should import through
``knox.compilation.two_phase`` for backward compatibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from slugify import slugify as _slugify_impl

from knox.agents.deps import GenerationDeps, PipelineAgentDeps
from knox.compilation.schema import LLMOutputError
from knox.core.cache_store import _decode_json, _encode_json, make_cache_key
from knox.core.exception_format import shorten_exception
from knox.core.models import ElementType, ResolvedChunk, WikiNode
from knox.core.template_loader import get_template_loader
from knox.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()

_CHARS_PER_TOKEN = 4  # Approximate for English text.

# Maximum slug length. Windows limits a single path component to 255 chars;
# the LLM sometimes emits a full sentence as a page title, which slugifies to
# 300+ chars and crashes ``write_text`` with ``OSError: [Errno 22]``. Cap the
# slug and suffix a short hash of the full slug so distinct long titles stay
# distinct (avoids _upsert_node merging two different concepts into one).
_SLUG_MAX_LENGTH: Final[int] = 80

# Bounded retry per batch before giving up — captures the non-determinism of
# small models (the same prompt often succeeds on a fresh attempt). The
# global failure tracker (_BatchFailureTracker) is what makes a
# systematically broken model fail hard.
_BATCH_RETRY_ATTEMPTS: Final[int] = 3
# A model failing more than this fraction of batches (after retries) is
# treated as not suitable for structured JSON compilation — compile stops
# early and the partial result is preserved (flagged incomplete).
_FAILURE_THRESHOLD_RATIO: Final[float] = 0.5
# Minimum batches before the threshold can fire — one failure in two batches
# shouldn't abort a small compile.
_FAILURE_THRESHOLD_MIN: Final[int] = 5

# Minimum input tokens per generation segment.  Prevents absurdly tiny segments
# when the model's context window is unknown or the fallback max_input is small
# enough that ``max_input - system - gen_output_max`` rounds to 1 token.
_MIN_SEGMENT_TOKENS = 512


def _tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * _CHARS_PER_TOKEN


def _chars_to_tokens(chars: int) -> int:
    """Convert character count to approximate token count."""
    return max(1, chars // _CHARS_PER_TOKEN)


_ACRONYM_RE = re.compile(r"\s*\([^)\d][^)]*\)\s*$")


def _normalize_concept_key(name: str) -> str:
    """Normalize a concept name for merge-key deduplication.

    Strips parenthetical acronyms, normalizes case/hyphens/whitespace,
    and collapses English plurals on the last word (heuristic).
    ``name`` is the output of :func:`_concept_name` so it is never empty.

    Examples::

        "Deep Neural Networks (DNNs)"  → "deep neural network"
        "Short-Time Objective Intelligibility (STOI)" → "short time objective intelligibility"
        "Deep neural network"  → "deep neural network"
        "Phoneme posteriorgrams" → "phoneme posteriorgram"
    """
    text = _ACRONYM_RE.sub("", name)
    text = text.replace("-", " ").lower().strip()
    if not text:
        return ""
    words = text.split()
    last = words[-1]
    if len(last) > 3:
        if last.endswith("ies"):
            words[-1] = last[:-3] + "y"
        elif last.endswith(("ches", "shes", "xes", "zes", "oes")):
            words[-1] = last[:-2]
        elif last.endswith("s") and not last.endswith("ss"):
            words[-1] = last[:-1]
    return " ".join(words).strip()


def _slugify(text: str) -> str:
    """Convert a title to a lowercase hyphenated slug (capped for filesystem safety).

    When the slug exceeds :data:`_SLUG_MAX_LENGTH` (e.g. the LLM emitted a full
    sentence as a concept title), it is truncated at a word boundary and a short
    hash of the *full* slug is appended so two distinct long titles don't collide
    (which would otherwise be merged by :func:`_upsert_node`).
    """
    slug = _slugify_impl(text) or "concept"
    if len(slug) <= _SLUG_MAX_LENGTH:
        return slug
    truncated = slug[: _SLUG_MAX_LENGTH - 5].rstrip("-")
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


def _extract_images_from_chunks(chunks: list[ResolvedChunk]) -> list[dict[str, Any]]:
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


def _compute_quality_score(nodes: list[WikiNode], edges: list[dict[str, Any]]) -> float:
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

        Concepts whose ``_normalize_concept_key`` collides are merged into one.
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
                key = _normalize_concept_key(name)
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


class _BatchFailureTracker:
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
        ratio: float = _FAILURE_THRESHOLD_RATIO,
        min_batches: int = _FAILURE_THRESHOLD_MIN,
    ) -> None:
        self.total = 0
        self.failed = 0
        self._ratio = ratio
        self._min_batches = min_batches

    def record(self, *, failed: bool) -> None:
        self.total += 1
        if failed:
            self.failed += 1

    def should_stop(self) -> bool:
        """True once the model's failure rate clears the stop threshold."""
        return self.total >= self._min_batches and self.failed / self.total > self._ratio


def _try_pydanticai_extraction(
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
    tracker can record the attempt uniformly with the litellm path.

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
        logger.warning("PydanticAI extraction failed for '%s': %s", source_id, shorten_exception(exc))
        return None
    except Exception as exc:  # ponytail: PydanticAI raises many subclassed errors; treat uniformly as agent failure
        logger.warning("PydanticAI extraction unexpected error for '%s': %s", source_id, shorten_exception(exc))
        return None


def _try_pydanticai_generation(
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
        logger.warning("PydanticAI generation failed for '%s': %s", source_id, shorten_exception(exc))
        return None
    except Exception as exc:  # ponytail: PydanticAI raises many subclassed errors; treat uniformly as agent failure
        logger.warning("PydanticAI generation unexpected error for '%s': %s", source_id, shorten_exception(exc))
        return None


def _merge_extractions(source: str, parts: list[Extraction]) -> Extraction:
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


def _segment_sources(
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


def _upsert_node(
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


def _gen_checkpoint_dir(config: object) -> Path:
    """Directory for per-batch generation checkpoint files."""
    return config.paths.cache_dir / "gen_checkpoint"


def _gen_checkpoint_path(config: object, gen_deps: dict[str, str]) -> Path:
    """On-disk path for the generation checkpoint."""
    return _gen_checkpoint_dir(config) / f"{make_cache_key(gen_deps)}.json"


def _load_gen_checkpoint(
    config: object,
    gen_deps: dict[str, str],
) -> tuple[list[WikiNode], dict[str, int], set[int]] | None:
    """Load a generation checkpoint if it exists and deps match.

    Returns ``(nodes, slug_to_idx, completed_batches)`` or ``None``
    when the checkpoint is absent, stale, or malformed.
    """
    if getattr(config, "no_cache", False):
        return None
    path = _gen_checkpoint_path(config, gen_deps)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Generation checkpoint read error -- ignoring", exc_info=True)
        return None
    stored_deps = raw.get("deps", {})
    if stored_deps != gen_deps:
        logger.debug("Generation checkpoint deps mismatch -- ignoring")
        return None
    try:
        nodes: list[WikiNode] = [_decode_json(n) for n in raw.get("nodes", [])]
    except Exception:
        logger.debug("Generation checkpoint node deserialization error -- ignoring", exc_info=True)
        return None
    slug_to_idx: dict[str, int] = dict(raw.get("slug_to_idx", {}))
    completed: set[int] = set(raw.get("completed_batches", []))
    return nodes, slug_to_idx, completed


def _save_gen_checkpoint(
    config: object,
    gen_deps: dict[str, str],
    nodes: list[WikiNode],
    slug_to_idx: dict[str, int],
    completed_batches: set[int],
) -> None:
    """Write generation checkpoint to disk (atomic write-to-temp + rename)."""
    if getattr(config, "no_cache", False):
        return
    path = _gen_checkpoint_path(config, gen_deps)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "deps": gen_deps,
        "nodes": [_encode_json(n) for n in nodes],
        "slug_to_idx": slug_to_idx,
        "completed_batches": list(completed_batches),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    try:
        os.replace(tmp, path)
        logger.debug("Generation checkpoint saved: %s", path)
    except OSError:
        logger.debug("Generation checkpoint write failed -- ignoring", exc_info=True)


def _delete_gen_checkpoint(config: object, gen_deps: dict[str, str]) -> None:
    """Remove a generation checkpoint from disk."""
    if getattr(config, "no_cache", False):
        return
    path = _gen_checkpoint_path(config, gen_deps)
    try:
        path.unlink()
        logger.debug("Generation checkpoint deleted: %s", path)
    except OSError:
        logger.debug("Generation checkpoint delete failed -- ignoring", exc_info=True)
