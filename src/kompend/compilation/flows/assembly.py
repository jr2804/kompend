"""Assembly flow — wiki node assembly, edge building, and concept merging.

Extracted from :class:`TwoPhaseCompiler` — pure functions that operate
on extracted data without depending on compiler instance state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from kompend.compilation._two_phase_core import ExtractionSet
from kompend.compilation.asset_registry import AssetIdMap
from kompend.compilation.asset_summarizer import summarize_asset
from kompend.core.models import ChunkTree, GraphEdge, ResolvedChunk, WikiNode
from kompend.core.models.enums import MetadataField
from kompend.core.models.structured import StructuredEquation, structured_to_text
from kompend.core.text.matching import cosine, similarity
from kompend.logging import get_logger
from kompend.rag.zvec_store import make_provider

logger = get_logger(__name__)


def merge_similar_concepts(
    concepts: list[dict[str, Any]],
    embedding_model: str,
    *,
    threshold: float,
    string_threshold: float,
    jaccard_min_token_len: int,
) -> list[dict[str, Any]]:
    """Merge concept pages whose name+definition embeddings exceed *threshold*.

    Uses the configured embedding provider. Groups concepts greedily:
    for every pair with cosine similarity >= threshold, sources and
    definitions are unioned and the longest name is kept as canonical.

    When embeddings are unavailable (``embedding_model == "hash"`` or
    provider creation/embedding fails), falls back to rapidfuzz string
    similarity (:func:`kompend.core.text.matching.similarity`) at
    *string_threshold* so dedup still works offline. *string_threshold*
    stays on the legacy 0–1 Jaccard scale for config compatibility and
    is scaled to the 0–100 rapidfuzz scale internally.
    """
    if len(concepts) < 2:
        return concepts

    embed_texts = [_concept_embed_text(c) for c in concepts]
    vecs: list[list[float]] | None = None

    if embedding_model and embedding_model != "hash":
        try:
            provider = make_provider(embedding_model)
            vecs = [provider.embed(t) for t in embed_texts]
        except Exception as exc:
            logger.warning(
                "Embedding merge disabled (%s); falling back to string similarity: %s",
                embedding_model,
                exc,
            )
            vecs = None

    if vecs is not None:
        n = len(concepts)
        similar = [(i, j) for i in range(n) for j in range(i + 1, n) if cosine(vecs[i], vecs[j]) >= threshold]
        groups = _union_find_groups(n, similar)
        result = _merge_groups(concepts, groups)
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

    # Fallback: rapidfuzz string-similarity merge (scaled 0–1 config → 0–100).
    n = len(concepts)
    names = [c["name"] for c in concepts]
    fuzzy_threshold = string_threshold * 100.0
    similar = [(i, j) for i in range(n) for j in range(i + 1, n) if similarity(names[i], names[j]) >= fuzzy_threshold]
    groups = _union_find_groups(n, similar)
    result = _merge_groups(concepts, groups)
    merged_count = n - len(result)
    if merged_count:
        logger.info(
            "String merge: %d groups combined (%d → %d concepts, threshold=%.2f)",
            merged_count,
            n,
            len(result),
            string_threshold,
        )
    return result


# ── Concept name-embedding merge (Option C) ─────────────────────────────
# All thresholds/limits live in BudgetConfig — the single source of truth.
# Callers must pass them explicitly; no defaults are assumed here.


def _concept_embed_text(concept: dict[str, Any]) -> str:
    """Build the text to embed for a concept: name + first definition.

    Embedding the definition alongside the name catches surface-variant
    duplicates (e.g. ``"MTD"`` ↔ ``"Mean Temporal Distance"``) whose
    definitions reveal they are the same concept.
    """
    name = concept.get("name", "")
    defs = concept.get("definitions") or concept.get("definition")
    first_def = ""
    if isinstance(defs, list) and defs:
        first_def = defs[0]
    elif isinstance(defs, str):
        first_def = defs
    combined = f"{name}. {first_def}".strip(". ")
    return combined or name


def _union_find_groups(n: int, similar: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Union-Find clustering: return leader → member-index mapping."""
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

    for i, j in similar:
        _union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)
    return groups


def _merge_groups(
    concepts: list[dict[str, Any]],
    groups: dict[int, list[int]],
) -> list[dict[str, Any]]:
    """Merge concept groups: longest name wins, sources/definitions unioned."""
    result: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) == 1:
            result.append(concepts[members[0]])
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
    return result


# ── Tree text building ──────────────────────────────────────────────────


def build_tree_texts(
    tree: ChunkTree,
    *,
    asset_mode: str = "inline",
    asset_id_map: AssetIdMap | None = None,
) -> list[str]:
    """Build hierarchical text blocks from a ChunkTree.

    Internal nodes produce heading + summary. Leaf nodes produce
    full text. The result is a list of text blocks that preserve
    the document's heading hierarchy.

    Parameters
    ----------
    asset_mode:
        How to render structured leaves (tables/figures):

        - ``"inline"`` (default): full text via :func:`structured_to_text`
          — the legacy behavior, kept for parity / A-B testing.
        - ``"ref"``: emit ``¶ <asset-id> (summary: …)`` and elide full
          content, so extraction input is not polluted by large tables /
          figures. Requires ``asset_id_map`` to resolve node ids to asset ids.

        Equations are always inlined (Phase 1 defers them as assets).
    asset_id_map:
        Mapping from :class:`ChunkNode` id to asset id (the output of
        :func:`assign_asset_ids`). Required when ``asset_mode == "ref"``;
        ignored otherwise.
    """
    if asset_mode == "ref" and asset_id_map is None:
        asset_mode = "inline"
    texts: list[str] = []
    for node in tree.walk():
        if node.chunk is None and not node.structured and not node.summary:
            continue
        heading_prefix = ""
        if node.heading_chain:
            heading_prefix = f"{'#' * min(node.depth + 1, 6)} {' / '.join(node.heading_chain)}\n\n"
        if node.chunk is not None:
            # Leaf prose node — full text.
            texts.append(f"{heading_prefix}{node.chunk.text}")
        elif node.structured is not None:
            # Leaf structured node — equation always inlined; table/figure
            # depends on asset_mode.
            if (
                asset_mode == "ref"
                and asset_id_map is not None
                and not isinstance(node.structured, StructuredEquation)
                and asset_id_map.get(node.id) is not None
            ):
                asset_id = asset_id_map.get(node.id)
                if asset_id is None:
                    texts.append(f"{heading_prefix}{structured_to_text(node.structured)}")
                    continue
                summary = summarize_asset(node.structured)
                texts.append(f"{heading_prefix}¶ {asset_id} (summary: {summary})")
            else:
                texts.append(f"{heading_prefix}{structured_to_text(node.structured)}")
        elif node.summary:
            # Internal node — summary only.
            texts.append(f"{heading_prefix}[summary: {node.summary}]")
    return texts


# ── Tag attachment ──────────────────────────────────────────────────────


def attach_tags_to_nodes(nodes: list[WikiNode], chunks: list[ResolvedChunk]) -> None:
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


def attach_attributes_to_nodes(nodes: list[WikiNode], chunks: list[ResolvedChunk]) -> None:
    """Attach key=value attributes from source chunks to wiki nodes.

    Each node gets a dict mapping attribute key → list of distinct values
    seen across all chunks that contributed to that node. This preserves
    per-source semantics: if source A has ``version=1`` and source B has
    ``version=2``, a node built from both has ``{"version": ["1", "2"]}``.
    """
    chunk_by_id: dict[str, ResolvedChunk] = {}
    for rc in chunks:
        if hasattr(rc, "chunk") and rc.chunk and rc.chunk.id:
            chunk_by_id[rc.chunk.id] = rc

    for node in nodes:
        attr_values: dict[str, list[str]] = {}
        if not node.source_chunks:
            continue
        for cid in node.source_chunks:
            rc = chunk_by_id.get(cid)
            if rc is None:
                continue
            attrs = rc.chunk.metadata.get(MetadataField.ATTRIBUTES) if hasattr(rc, "chunk") else None
            if not attrs:
                continue
            for key, val in attrs.items():
                seen = attr_values.setdefault(key, [])
                if val not in seen:
                    seen.append(val)
        if attr_values:
            node.metadata[MetadataField.ATTRIBUTES] = attr_values


# ── Edge building ───────────────────────────────────────────────────────


def build_edges(
    nodes: list[WikiNode],
    extractions: ExtractionSet,
    *,
    min_title_len: int,
) -> list[GraphEdge]:
    """Build cross-source entity and explicit related edges."""
    edges: list[GraphEdge] = []
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
            if len(other_title) < min_title_len:  # skip very short generic titles
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
