"""Assembly flow — wiki node assembly, edge building, and concept merging.

Extracted from :class:`TwoPhaseCompiler` — pure functions that operate
on extracted data without depending on compiler instance state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from knox.chunking.summarizer import _structured_to_text
from knox.compilation._two_phase_core import ExtractionSet
from knox.core.models import ChunkTree, WikiNode
from knox.core.models.enums import MetadataField
from knox.logging import get_logger
from knox.rag.zvec_store import _make_provider

logger = get_logger(__name__)

# ── Concept name-embedding merge (Option C) ─────────────────────────────

_CONCEPT_MERGE_THRESHOLD = 0.85


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (no numpy needed)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def merge_similar_concepts(
    concepts: list[dict[str, Any]],
    embedding_model: str,
    *,
    threshold: float = _CONCEPT_MERGE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Merge concept pages whose name embeddings are above *threshold*.

    Uses the configured embedding provider (SentenceTransformer either
    locally or via LiteLLM). Groups concepts greedily: for every pair
    with cosine similarity >= threshold, sources and definitions are
    unioned and the longest name is kept as canonical.

    When the embedding provider fails (e.g. model not found), concepts
    are returned unchanged and a warning is logged.
    """
    if not embedding_model or embedding_model == "hash":
        return concepts

    try:
        provider = _make_provider(embedding_model, dim=None)
    except Exception as exc:
        logger.debug(
            "Embedding merge disabled: could not create provider for %s: %s",
            embedding_model,
            exc,
        )
        return concepts

    names = [c["name"] for c in concepts]
    try:
        vecs = [provider.embed(n) for n in names]
    except Exception as exc:
        logger.debug("Embedding merge disabled: embedding call failed: %s", exc)
        return concepts

    # Union-Find to group similar concepts.
    n = len(concepts)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx < ry:
            parent[ry] = rx
        else:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_similarity(vecs[i], vecs[j]) >= threshold:
                _union(i, j)

    # Group by leader.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        leader = _find(i)
        groups.setdefault(leader, []).append(i)

    # Merge each multi-member group.
    result: list[dict[str, Any]] = []
    for leader, members in groups.items():
        if len(members) == 1:
            result.append(concepts[leader])
            continue
        merged: dict[str, Any] = {
            "name": max((concepts[m]["name"] for m in members), key=len),
            "sources": [],
            "definitions": [],
        }
        seen_sources: set[str] = set()
        seen_defs: set[str] = set()
        for m in members:
            for s in concepts[m].get("sources", []):
                if s not in seen_sources:
                    merged["sources"].append(s)
                    seen_sources.add(s)
            for d in concepts[m].get("definitions", []):
                if d not in seen_defs:
                    merged["definitions"].append(d)
                    seen_defs.add(d)
        result.append(merged)

    merged_count = n - len(result)
    if merged_count:
        logger.info(
            "Embedding merge: %d groups combined (%d → %d concepts, threshold=%.2f)",
            merged_count,
            n,
            len(result),
            threshold,
        )
    return result


# ── Tree text building ──────────────────────────────────────────────────


def build_tree_texts(tree: ChunkTree) -> list[str]:
    """Build hierarchical text blocks from a ChunkTree.

    Internal nodes produce heading + summary. Leaf nodes produce
    full text. The result is a list of text blocks that preserve
    the document's heading hierarchy.
    """
    texts: list[str] = []
    for node in tree.walk():
        if node.chunk is None and not node.structured and not node.summary:
            continue
        heading_prefix = ""
        if node.heading_path:
            heading_prefix = f"{'#' * min(node.depth + 1, 6)} {' / '.join(node.heading_path)}\n\n"
        if node.chunk is not None:
            # Leaf prose node — full text.
            texts.append(f"{heading_prefix}{node.chunk.text}")
        elif node.structured is not None:
            # Leaf structured node — text representation.
            texts.append(f"{heading_prefix}{_structured_to_text(node.structured)}")
        elif node.summary:
            # Internal node — summary only.
            texts.append(f"{heading_prefix}[summary: {node.summary}]")
    return texts


# ── Tag attachment ──────────────────────────────────────────────────────


def attach_tags_to_nodes(nodes: list[WikiNode], chunks: list[Any]) -> None:
    """Attach tags from chunk metadata to wiki nodes."""
    all_tags: list[str] = []
    for rc in chunks:
        chunk_tags = rc.chunk.metadata.get(MetadataField.TAGS) if hasattr(rc, "chunk") else None
        if chunk_tags:
            all_tags.extend(chunk_tags)
    unique_tags = list(dict.fromkeys(all_tags))
    if unique_tags:
        for node in nodes:
            node.metadata[MetadataField.TAGS] = unique_tags


# ── Edge building ───────────────────────────────────────────────────────


def build_edges(
    nodes: list[WikiNode],
    extractions: ExtractionSet,
) -> list[dict[str, Any]]:
    """Build cross-source entity and explicit related edges."""
    edges: list[dict[str, Any]] = []
    entity_index = extractions.entity_index()
    entity_to_node: dict[str, set[str]] = defaultdict(set)
    for ent_name in entity_index:
        ent_lower = ent_name.lower()
        for node in nodes:
            # Scan FULL content — cropping to the first N chars (the old
            # [:200]) silently dropped every match past the cutoff, falsely
            # orphaning pages whose entity mentions appear later.
            match_text = (f"{node.title} {node.content or ''}").lower()
            if ent_lower in match_text:
                entity_to_node[ent_name].add(node.id)

    # Deduplicate edges — same pair may appear from both entity and related links
    seen_edges: set[tuple[str, str]] = set()
    for ent_name, node_ids in entity_to_node.items():
        sorted_ids = sorted(node_ids)
        for i in range(len(sorted_ids)):
            for j in range(i + 1, len(sorted_ids)):
                pair = (sorted_ids[i], sorted_ids[j])
                if pair in seen_edges:
                    continue
                seen_edges.add(pair)
                edges.append(
                    {
                        "source": sorted_ids[i],
                        "target": sorted_ids[j],
                        "type": "entity_reference",
                        "entity": ent_name,
                    }
                )

    # Also wire up explicit "related" links from the LLM response.
    title_to_id = {n.title: n.id for n in nodes}
    for node in nodes:
        for related_title in node.metadata.get(MetadataField.RELATED, []):
            target_id = title_to_id.get(related_title)
            if target_id and target_id != node.id:
                pair = (min(node.id, target_id), max(node.id, target_id))
                if pair in seen_edges:
                    continue
                seen_edges.add(pair)
                edges.append(
                    {"source": node.id, "target": target_id, "type": "related", "entity": related_title},
                )

    # Title-based edges: if a page's title appears as plaintext in another
    # page's content, that's an implicit cross-reference.
    for node in nodes:
        content_lower = (node.content or "").lower()
        for other in nodes:
            if other.id == node.id:
                continue
            other_title = other.title.lower().strip()
            if len(other_title) < 4:  # ponytail: skip very short generic titles
                continue
            if other_title in content_lower:
                pair = (min(node.id, other.id), max(node.id, other.id))
                if pair in seen_edges:
                    continue
                seen_edges.add(pair)
                edges.append(
                    {
                        "source": node.id,
                        "target": other.id,
                        "type": "title_reference",
                        "entity": other.title,
                    }
                )

    return edges
