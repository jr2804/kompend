"""Planning and measurement helpers for the two-phase compiler.

Phase 1 splits sources into windows, turns model output into extractions, and
measures the result. The module also re-exports the shared budget arithmetic
from :mod:`kompend.llm.request_budget`.
"""

from __future__ import annotations

import hashlib

from kompend.agents.base import KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.failure import UnitFailureLog, classify_agent_failure
from kompend.compilation.schema import ExtractionOutput
from kompend.core.models import ElementType, GraphEdge, ImageRef, MetadataField, ResolvedChunk, WikiNode
from kompend.core.text.slug import slugify as _canonical_slugify
from kompend.core.utils.exceptions import LLMContextOverflowError
from kompend.llm.request_budget import (
    CONTEXT_SAFETY_FRACTION,
    CONTEXT_SAFETY_TOKENS,
    chars_to_tokens,
    context_safety_tokens,
    tokens_to_chars,
)
from kompend.llm.usage import UsageTracker
from kompend.logging import get_logger

# The budget helpers are re-exported here: the compilation layer's budget
# arithmetic has always been importable from the two-phase writer, and the
# planner now owns the implementation.
__all__ = [
    "CONTEXT_SAFETY_FRACTION",
    "CONTEXT_SAFETY_TOKENS",
    "chars_to_tokens",
    "compute_quality_score",
    "context_safety_tokens",
    "extract_images_from_chunks",
    "rate_suffix",
    "segment_sources",
    "slugify",
    "tokens_to_chars",
    "try_pydanticai_extraction",
]

# Maximum slug length. Windows limits a single path component to 255 chars;
# the LLM sometimes emits a full sentence as a page title, which slugifies to
# 300+ chars and crashes ``write_text`` with ``OSError: [Errno 22]``. Cap the
# slug and suffix a short hash of the full slug so distinct long titles stay
# distinct (avoids upsert_node merging two different concepts into one).
# Default slug max length. Configurable via ``compile.llm.slug_max_length``.
_SLUG_MAX_LENGTH: int = 80

#: The same names, used to keep a *status* out of a figure caption. An
#: entity with one of these names says what happened to the figure, not
#: what it shows.
_STATUS_ENTITY_NAMES: frozenset[str] = frozenset({"vlm_filtered", "vlm_skipped", "vlm_error", "vlm_placeholder"})

#: Re-exported from :mod:`kompend.llm.request_budget` — the planner owns the
#: safety margin and the budget arithmetic; this module keeps the historical
#: import path working. See that module for the rationale.
#:
#: Fraction of a model's context window held back as headroom beyond the
#: measured per-request overhead.  Two reasons: a server's *effective* usable
#: context is smaller than the nominal window (KV cache slotting, prompt-cache
#: rollover), and the chars-per-token heuristic undercounts token-dense content
#: (formulas, tables, identifiers).  Without it a nominally exact-fit request
#: still overflows — observed as ``request (65762 tokens) exceeds the available
#: context size (65536 tokens)`` on a formula-heavy standard.
#:
#: Content is never cropped because of this margin: the content budget shrinks,
#: so the work splits across more batches/segments instead.

logger = get_logger(__name__)


def rate_suffix(tracker: UsageTracker | None) -> str:
    """`` · 12.3 tok/s (avg 9.8)`` from the shared tracker, or ``""``.

    Uses the tracker's real completion-token event log (short-term window
    + overall average) — the single source of truth for throughput.
    """
    if tracker is None:
        return ""
    overall = tracker.overall_rate()
    recent = tracker.recent_rate()
    if not overall and not recent:
        return ""
    return f" · {recent:.1f} tok/s" if recent else f" · avg {overall:.1f} tok/s"


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
    slug = _canonical_slugify(text, default="concept")
    if len(slug) <= cap:
        return slug
    truncated = slug[: cap - 5].rstrip("-")
    return f"{truncated}-{hashlib.sha256(slug.encode()).hexdigest()[:4]}"


def extract_images_from_chunks(chunks: list[ResolvedChunk]) -> list[ImageRef]:
    """Extract image metadata from IMAGE chunks.

    Returns a list of dicts with keys: path, caption, element_type, source_file.
    Images are collected from:
    - chunk.metadata[MetadataField.IMAGE_FILE] (file mode)
    - chunk.metadata[MetadataField.IMAGE_BASE64] (base64 mode)
    - VLM captions from resolved.entities with type="image_description"

    ``source_file`` is the document the figure came from — callers select the
    subset belonging to a page with :func:`images_for_provenance`.

    Only a real caption may be published: an entity of type
    ``image_description`` whose name is not one of the status names
    (``vlm_error``, ``vlm_filtered``, ``vlm_placeholder``, ``vlm_skipped``).
    A status entity says what happened to the figure, not what it shows.
    """
    images: list[ImageRef] = []

    for rc in chunks:
        if ElementType.IMAGE not in rc.chunk.element_types:
            continue

        img_ref = rc.chunk.metadata.get(MetadataField.IMAGE_FILE) or rc.chunk.metadata.get(MetadataField.IMAGE_BASE64)
        if not img_ref:
            # Try resolved metadata
            img_ref = rc.metadata.get(MetadataField.IMAGE_FILE) or rc.metadata.get(MetadataField.IMAGE_BASE64)

        if not img_ref:
            continue

        caption = ""
        for entity in rc.entities:
            if entity.get("type") != "image_description" or not entity.get("context"):
                continue
            name = str(entity.get("name") or "")
            if name in _STATUS_ENTITY_NAMES:
                continue
            caption = str(entity["context"])
            break

        images.append(
            {
                "path": img_ref,
                "caption": caption,
                "element_type": "image",
                "source_file": str(rc.chunk.source_file or ""),
            }
        )  # ImageRef

    return images


def compute_quality_score(nodes: list[WikiNode], edges: list[GraphEdge]) -> float:
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


def try_pydanticai_extraction(
    agent: KompendAgentBase[ExtractionOutput, PipelineAgentDeps],
    *,
    text: str,
    source_id: str,
    failures: UnitFailureLog | None = None,
) -> ExtractionOutput | None:
    """Run a PydanticAI :class:`ExtractionAgent` on a chunk of source text.

    This is the agent-based extraction entry point. Callers invoke it
    for every extraction batch (it is the only path, not opt-in).

    The agent's ``run`` method already does JSON extraction and Pydantic
    validation internally; auto-retry on schema mismatch is built in.
    A failure is classified by :func:`kompend.agents.failure.classify_agent_failure`
    and returned as ``None`` so the caller's batch-retry loop can decide
    whether repeating it is worth anything — an empty answer is not (the
    model already proved it had nothing to say), a schema mismatch is.

    Args:
        agent: An instantiated :class:`~kompend.agents.ExtractionAgent`.
        text: The combined chunk text to extract from.
        source_id: Provenance tag (file stem / URL).
        failures: Optional per-source failure log; receives the classified
            record so the caller can read the kind back.

    Returns:
        The validated :class:`~kompend.compilation.schema.ExtractionOutput`
        instance, or ``None`` on failure.
    """
    try:
        deps = PipelineAgentDeps(text=text, source_id=source_id)
        return agent.run(deps)
    except LLMContextOverflowError:
        # The request did not fit the provider's window. Re-planning (a
        # smaller batch) is the caller's job - the error carries the limit
        # the provider named, so swallowing it here would lose the only
        # number that can fix the request.
        raise
    except Exception as exc:  # noqa: BLE001 — the classifier below decides what kind this is
        failure = failures.record(source_id, exc) if failures is not None else None
        kind = failure.kind.value if failure is not None else classify_agent_failure(exc).value
        logger.warning("PydanticAI extraction failed for '%s' (%s): %s", source_id, kind, exc)
        return None


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
