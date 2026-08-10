"""Assembly flow — wiki node assembly, edge building, and concept merging.

Extracted from :class:`TwoPhaseCompiler` — pure functions that operate
on extracted data without depending on compiler instance state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from knox.chunking.summarizer import structured_to_text
from knox.compilation._two_phase_core import ExtractionSet
from knox.compilation.asset_registry import AssetIdMap
from knox.compilation.asset_summarizer import summarize_asset
from knox.core.models import ChunkTree, ResolvedChunk, WikiNode
from knox.core.models.enums import MetadataField
from knox.core.models.structured import StructuredEquation
from knox.logging import get_logger
from knox.rag.zvec_store import make_provider

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
    provider creation/embedding fails), falls back to a pure-Python
    token-overlap (Jaccard) merge at *string_threshold* so dedup still
    works offline.
    """
    if len(concepts) < 2:
        return concepts

    embed_texts = [_concept_embed_text(c) for c in concepts]
    vecs: list[list[float]] | None = None

    if embedding_model and embedding_model != "hash":
        try:
            provider = make_provider(embedding_model, dim=None)
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
        similar = [(i, j) for i in range(n) for j in range(i + 1, n) if cosine_similarity(vecs[i], vecs[j]) >= threshold]
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

    # Fallback: string-similarity (token Jaccard) merge.
    n = len(concepts)
    names = [c["name"] for c in concepts]
    similar = [(i, j) for i in range(n) for j in range(i + 1, n) if _token_jaccard(names[i], names[j], min_token_len=jaccard_min_token_len) >= string_threshold]
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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (no numpy needed)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


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


def _token_jaccard(a: str, b: str, *, min_token_len: int) -> float:
    """Token-overlap ratio (Jaccard) between two strings.

    Pure-Python fallback for when no embedding model is available.
    Catches near-duplicate concept names like ``"Mean Temporal Distance
    Measure"`` ↔ ``"Mean Temporal Distance"`` whose token sets overlap
    heavily. Tokens shorter than *min_token_len* are ignored.
    """
    ta = {t for t in a.lower().split() if len(t) > min_token_len}
    tb = {t for t in b.lower().split() if len(t) > min_token_len}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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
        if node.heading_path:
            heading_prefix = f"{'#' * min(node.depth + 1, 6)} {' / '.join(node.heading_path)}\n\n"
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


# ── Edge building ───────────────────────────────────────────────────────


def build_edges(
    nodes: list[WikiNode],
    extractions: ExtractionSet,
    *,
    min_title_len: int,
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
