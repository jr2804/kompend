"""Internal helpers for the N3 two-phase compiler.

This module is private API. External callers should import through
``knox.compilation.two_phase`` for backward compatibility.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import litellm
from slugify import slugify as _slugify_impl

from knox.core.models import ElementType, ResolvedChunk, WikiNode
from knox.core.template_loader import get_template_loader
from knox.llm.client import JSON_RESPONSE_FORMAT, _strip_reasoning, salvage_json
from knox.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()

_CHARS_PER_TOKEN = 4  # Approximate for English text.

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


_ACRONYM_RE = re.compile(r"\s*\([^)]*[A-Z]{2,}[^)]*\)\s*$")


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
    """Convert a title to a lowercase hyphenated slug."""
    return _slugify_impl(text) or "concept"


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


def _try_litellm_json(
    prompt: str,
    model: str,
    system: str | None = None,
    timeout: float | None = None,
    call_stats: list[tuple[int, int]] | None = None,
) -> Any | None:
    """Call litellm completion and return parsed JSON, or None on failure.

    If *call_stats* is provided, ``(prompt_tokens, completion_tokens)`` is
    appended to it on each successful LLM response.
    """
    start = time.monotonic()
    try:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": JSON_RESPONSE_FORMAT,
            "temperature": 0.0,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = litellm.completion(**kwargs)
        elapsed = time.monotonic() - start
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            except (TypeError, ValueError):
                pt = 0
            try:
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
            except (TypeError, ValueError):
                ct = 0
            tt = pt + ct
            tps = f"{tt / elapsed:.1f}" if elapsed > 0 else "?"
            if call_stats is not None:
                call_stats.append((pt, ct))
            logger.info(
                "LLM ok model=%s in=%.1fs in=%d out=%d total=%d tok/s=%s",
                model,
                elapsed,
                pt,
                ct,
                tt,
                tps,
            )
        else:
            logger.info("LLM ok model=%s in=%.1fs (no token info)", model, elapsed)
        content = response.choices[0].message.content
        content = _strip_reasoning(content)
        if not content:
            return None
        parsed = salvage_json(content)
        if parsed is None:
            elapsed = time.monotonic() - start
            # ponytail: JSON parse failure is non-fatal — the caller falls
            # back gracefully. A full traceback box looks like a crash.
            logger.warning(
                "LLM returned unparseable JSON for model=%s in=%.1fs — using fallback",
                model,
                elapsed,
            )
        return parsed
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.warning("LLM call failed for model=%s in=%.1fs — %s", model, elapsed, exc)
        return None


def _fallback_extraction(source: str, text: str) -> Extraction:
    """Heuristic extraction when the LLM is unavailable."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    summary = sentences[0] if sentences else ""
    return Extraction(
        source=source,
        summary=summary,
        entities=[],
        concepts=[{"name": "auto_concept", "definition": summary}] if summary else [],
        tags=[],
    )


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
