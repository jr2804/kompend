"""Assembly flow — wiki node assembly, edge building, and concept merging.

Extracted from :class:`TwoPhaseCompiler` — pure functions that operate
on extracted data without depending on compiler instance state.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from kompend.compilation.asset_registry import AssetIdMap
from kompend.compilation.asset_summarizer import summarize_asset
from kompend.compilation.two_phase import ExtractionSet
from kompend.core.models import ChunkTree, GraphEdge, ResolvedChunk, StructuredFigure, WikiNode
from kompend.core.models.enums import EdgeType, MetadataField
from kompend.core.models.structured import StructuredEquation, structured_to_text
from kompend.core.text.matching import cosine, similarity
from kompend.logging import get_logger
from kompend.rag.embedding_pydantic import make_provider

logger = get_logger(__name__)


def merge_similar_concepts(
    concepts: list[dict[str, Any]],
    embedding_model: str,
    *,
    threshold: float,
    string_threshold: float,
    jaccard_min_token_len: int,
    max_reduction: float = 1.0,
    max_sources: int = 0,
    similarity_margin: float = 1.0,
) -> list[dict[str, Any]]:
    """Merge concept pages whose name+definition embeddings exceed *threshold*.

    Uses the configured embedding provider. Concepts are grouped by
    **centroid linkage**: a concept joins the cluster whose centroid it is
    closest to, and the cluster centroid moves to the mean of its members.
    Centroid linkage, not transitive closure: two concepts that are each
    similar to a third are not thereby similar to each other, and a chain
    of such pairs cannot absorb the whole corpus through a pairwise rule.

    When embeddings are unavailable (``embedding_model == "hash"`` or
    provider creation/embedding fails), falls back to rapidfuzz string
    similarity (:func:`kompend.core.text.matching.similarity`) at
    *string_threshold* so dedup still works offline. *string_threshold*
    stays on the legacy 0–1 Jaccard scale for config compatibility and
    is scaled to the 0–100 rapidfuzz scale internally.

    Two guards bound the damage a merge can do, both disabled by default
    (``max_reduction=1.0``, ``max_sources=0``):
    *max_reduction* is the largest fraction of the pool one merge may
    remove, and *max_sources* the largest number of sources one merged
    concept may hold. A merge that would exceed either is refused and
    logged - the concepts stay separate rather than silently collapsing.

    A cluster too large for those guards is not refused wholesale. It is
    merged pairwise: the most similar pairs first, every accepted pair
    within *similarity_margin* of the cluster's own best pair, and only
    as many pairs as the pool's remaining budget allows
    (:func:`_top_k_merge`). The margin is relative to the cluster, so no
    absolute similarity constant decides what merges.
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
        groups = _centroid_groups(n, vecs, threshold)
        result = _merge_groups(
            concepts,
            groups,
            max_reduction=max_reduction,
            max_sources=max_sources,
            similarity_margin=similarity_margin,
            pair_similarity=_embedding_similarity(vecs),
        )
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
    result = _merge_groups(
        concepts,
        groups,
        max_reduction=max_reduction,
        max_sources=max_sources,
        similarity_margin=similarity_margin,
        pair_similarity=_string_similarity(concepts),
    )
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


def _centroid_groups(n: int, vecs: list[list[float]], threshold: float) -> dict[int, list[int]]:
    """Group concepts by centroid linkage: leader → member-index mapping.

    Each concept joins the cluster whose centroid it is closest to, and that
    cluster's centroid becomes the mean of its members. Unlike transitive
    closure, a pair that is only *indirectly* connected never ends up in the
    same cluster: "PESQ" and "POLQA" can each be close to "speech quality"
    without being close to each other, and a chain of such pairs cannot
    absorb the corpus through a pairwise rule.

    The first concept seeds the first cluster; every later concept either
    joins the closest centroid within *threshold* or seeds a new cluster.
    """
    members: list[list[int]] = []
    for index in range(n):
        vec = vecs[index]
        best = -1
        best_score = threshold
        for c_index, cluster in enumerate(members):
            score = cosine(vec, _centroid(vecs, cluster))
            if score >= best_score:
                best = c_index
                best_score = score
        if best < 0:
            members.append([index])
            continue
        members[best].append(index)
    return dict(enumerate(members))


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
    *,
    max_reduction: float = 1.0,
    max_sources: int = 0,
    similarity_margin: float = 1.0,
    pair_similarity: Callable[[list[int], list[int]], float] | None = None,
) -> list[dict[str, Any]]:
    """Merge concept groups: longest name wins, sources/definitions unioned.

    The guards bound what a merge can do. Two limits apply:

    * *max_sources* — one merged concept may hold at most this many
      sources. A cluster spanning dozens of sources is a cluster of distinct
      concepts, not one concept seen from many angles.
    * *max_reduction* — the whole merge may remove at most this fraction of
      the pool. A collapse of that size means the threshold is matching a
      domain rather than a concept.

    A cluster that breaks either limit is not merged in one piece: it is
    merged pairwise, the most similar pairs first and only as far as the
    pool's own budget allows (:func:`_top_k_merge`). A cluster whose budget
    runs out before a single pair is admissible keeps its members separate,
    and the refusal is logged. Refusing keeps the concepts visible;
    merging them silently is what turned 340 candidates into 31 pages.
    """
    total = len(concepts)
    budget = max_reduction * total if max_reduction < 1.0 else math.inf
    entries = [(key, members) for key, members in groups.items() if len(members) > 1]
    clusters = [members for _, members in entries]

    whole: list[tuple[int, list[int]]] = []
    oversized: list[tuple[int, list[int]]] = []
    for key, members in entries:
        if _breaks_guard(members, concepts, budget, max_sources):
            oversized.append((key, members))
        else:
            whole.append((key, members))

    # The biggest absorber goes first, and a cluster that does not fit what
    # is left of the pool-wide budget is refused instead of merged.
    absorbed = 0
    refused: set[int] = set()
    for key, members in sorted(whole, key=lambda entry: len(entry[1]), reverse=True):
        if absorbed + len(members) - 1 > budget:
            refused.add(key)
            continue
        absorbed += len(members) - 1

    # What is left of the budget is shared by the oversized clusters by size.
    remaining = max(0.0, budget - absorbed)
    demand = sum(len(members) - 1 for _, members in oversized)

    plan: dict[int, list[list[int]]] = {key: [members] for key, members in entries}
    for key, members in oversized:
        cap = _cluster_cap(members, concepts, remaining, demand, max_sources)
        parts = _top_k_merge(
            concepts,
            members,
            cap,
            max_sources=max_sources,
            similarity_margin=similarity_margin,
            pair_similarity=pair_similarity,
        )
        plan[key] = parts
        merged_pairs = sum(len(part) - 1 for part in parts)
        if merged_pairs:
            logger.info(
                "Concept merge limited: %d of %d pairs merged in a cluster of %d concepts",
                merged_pairs,
                len(members) - 1,
                len(members),
            )
        if merged_pairs < len(members) - 1:
            _log_refusal(
                clusters,
                members,
                concepts,
                total,
                f"{merged_pairs} of {len(members) - 1} pairs merged, the rest stay separate",
            )

    for key, members in entries:
        if key in refused:
            _log_refusal(clusters, members, concepts, total, "keeping them separate")

    result: list[dict[str, Any]] = []
    for leader, members in groups.items():
        if len(members) == 1:
            result.append(concepts[members[0]])
            continue
        for part in plan[leader]:
            if len(part) == 1:
                result.append(concepts[part[0]])
            else:
                result.append(_merged_concept(concepts, part))
    return result


def _breaks_guard(
    members: list[int],
    concepts: list[dict[str, Any]],
    budget: float,
    max_sources: int,
) -> bool:
    """Whether one cluster is too large to merge in a single piece."""
    if max_sources > 0 and len(_group_sources(concepts, members)) > max_sources:
        return True
    return len(members) - 1 > budget


def _cluster_cap(
    members: list[int],
    concepts: list[dict[str, Any]],
    remaining: float,
    demand: int,
    max_sources: int,
) -> int:
    """How many pairs one oversized cluster may merge.

    The pool's own budget bounds the merge: the cluster's share of what is
    left of it, by size — nothing absolute decides the count. The source
    cap bounds it a second time: a merged concept may hold at most
    *max_sources* sources, so a cluster holding S sources cannot collapse
    below ceil(S / max_sources) concepts.
    """
    share = len(members) - 1
    if demand <= 0:
        return 0
    cap = share if math.isinf(remaining) else int(remaining * share / demand)
    if max_sources > 0:
        sources = len(_group_sources(concepts, members))
        cap = min(cap, len(members) - math.ceil(sources / max_sources))
    return max(0, min(cap, share))


def _top_k_merge(
    concepts: list[dict[str, Any]],
    members: list[int],
    cap: int,
    *,
    max_sources: int,
    similarity_margin: float,
    pair_similarity: Callable[[list[int], list[int]], float] | None,
) -> list[list[int]]:
    """Merge the *cap* most-similar pairs inside one oversized cluster.

    The pairs are ranked by similarity and taken from the top, so the
    near-duplicates go first and the distinct sub-topics stay separate. A
    pair is admissible when its similarity is at least *similarity_margin*
    of the cluster's own best pair — a margin relative to the cluster, not
    an absolute similarity constant. The merged pairs of a round are
    disjoint, and a later round may merge the results again, so a cluster
    of revisions of one concept collapses into that concept.

    The rounds are bounded by the cluster's own size (a group at most
    doubles per round, so ceil(log2(m)) rounds suffice) and stop as soon as
    a round merges fewer than its corpus-relative share of the budget
    (ceil(cap / rounds)) — a round that no longer pays for itself ends the
    collapse.
    """
    groups = [[member] for member in members]
    if cap <= 0 or pair_similarity is None:
        return groups

    rounds = max(1, math.ceil(math.log2(len(members))))
    min_merges = math.ceil(cap / rounds)
    budget = cap
    floor: float | None = None
    for _ in range(rounds):
        pairs = [(pair_similarity(groups[i], groups[j]), i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
        if not pairs:
            break
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        if floor is None:
            floor = pairs[0][0] * similarity_margin
        partner: dict[int, int] = {}
        merged = 0
        for score, left, right in pairs:
            if score < floor:
                break
            if merged >= budget:
                break
            if left in partner or right in partner:
                continue
            if not _may_merge(concepts, groups[left], groups[right], max_sources):
                continue
            partner[left] = right
            partner[right] = left
            merged += 1
        if merged == 0:
            break
        groups = _apply_pairs(groups, partner)
        budget -= merged
        if merged < min_merges or budget <= 0:
            break
    return groups


def _apply_pairs(groups: list[list[int]], partner: dict[int, int]) -> list[list[int]]:
    """Merge the paired groups, leaving the unpaired ones as they are."""
    merged: list[list[int]] = []
    consumed: set[int] = set()
    for index, group in enumerate(groups):
        if index in consumed:
            continue
        mate = partner.get(index)
        if mate is None:
            merged.append(group)
            continue
        consumed.add(index)
        consumed.add(mate)
        merged.append(group + groups[mate])
    return merged


def _may_merge(
    concepts: list[dict[str, Any]],
    left: list[int],
    right: list[int],
    max_sources: int,
) -> bool:
    """Whether the merged concept may hold its sources without breaking the cap."""
    if max_sources <= 0:
        return True
    sources = set(_group_sources(concepts, left)) | set(_group_sources(concepts, right))
    return len(sources) <= max_sources


def _merged_concept(concepts: list[dict[str, Any]], members: list[int]) -> dict[str, Any]:
    """One concept out of a group: longest name wins, sources and definitions unioned."""
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
    return merged


def _embedding_similarity(vecs: list[list[float]]) -> Callable[[list[int], list[int]], float]:
    """Score two groups by the cosine of their member centroids."""

    def _score(left: list[int], right: list[int]) -> float:
        return cosine(_centroid(vecs, left), _centroid(vecs, right))

    return _score


def _centroid(vecs: list[list[float]], cluster: list[int]) -> list[float]:
    """The centroid of a cluster: the component-wise mean of its members."""
    if len(cluster) == 1:
        return vecs[cluster[0]]
    size = len(cluster)
    return [sum(vecs[index][axis] for index in cluster) / size for axis in range(len(vecs[cluster[0]]))]


def _string_similarity(concepts: list[dict[str, Any]]) -> Callable[[list[int], list[int]], float]:
    """Score two groups by the fuzzy similarity of their canonical names."""

    def _score(left: list[int], right: list[int]) -> float:
        return similarity(_group_name(concepts, left), _group_name(concepts, right))

    return _score


def _group_name(concepts: list[dict[str, Any]], members: list[int]) -> str:
    """The canonical name of a group: the longest name among its members."""
    return max((concepts[m].get("name", "") for m in members), key=len)


def _log_refusal(
    clusters: list[list[int]],
    members: list[int],
    concepts: list[dict[str, Any]],
    pool_size: int,
    outcome: str,
) -> None:
    """Log a refused merge, what was done instead, and the shape around it.

    The cluster sizes are the shape of the redundancy: a refusal that names
    only the refused cluster leaves the next run without the picture of what
    the merge wanted to do.
    """
    logger.warning(
        "Concept merge refused: merging %d concepts holding %d sources would exceed the budget; %s — pool %d concepts in %d clusters, largest: %s",
        len(members),
        len(_group_sources(concepts, members)),
        outcome,
        pool_size,
        len(clusters),
        _cluster_distribution(clusters, concepts, pool_size),
    )


def _cluster_distribution(clusters: list[list[int]], concepts: list[dict[str, Any]], pool_size: int) -> str:
    """The head of the cluster-size distribution: the clusters that hold most of the pool.

    The clusters are taken from the largest down until they hold more than
    half the pool, so the line shows the redundancy that explains the merge
    rather than every cluster of a large corpus.
    """
    ranked = sorted(clusters, key=len, reverse=True)
    head: list[str] = []
    covered = 0
    for members in ranked:
        head.append(f"{len(members)} concepts/{len(_group_sources(concepts, members))} sources")
        covered += len(members)
        if covered * 2 >= pool_size:
            break
    return ", ".join(head)


def _group_sources(concepts: list[dict[str, Any]], members: list[int]) -> list[str]:
    """The distinct sources the group's members contribute."""
    sources: list[str] = []
    seen: set[str] = set()
    for m in members:
        for s in concepts[m].get("sources", []):
            if s not in seen:
                seen.add(s)
                sources.append(s)
    return sources


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
        if node.structured is not None:
            # A structured leaf is checked before its flattened chunk: an image
            # chunk's text is only ``"[binary data: N bytes]"``, a placeholder
            # no compiler can extract a claim from. What the figure says is its
            # caption, and that is what belongs where the figure stood.
            if isinstance(node.structured, StructuredFigure):
                # Prefer the chunk text: the resolve stage replaces a figure's
                # byte placeholder with what the VLM saw, and that description
                # is richer than the document's own label. The structured
                # fields stay as the fallback for parser-captioned figures
                # when no VLM ran.
                caption = ""
                if node.chunk is not None:
                    chunk_text = node.chunk.text.strip()
                    if chunk_text and not chunk_text.startswith("[binary data:"):
                        caption = chunk_text
                if not caption:
                    caption = node.structured.caption or node.structured.description or node.structured.alt_text
                if caption:
                    texts.append(f"{heading_prefix}{caption}")
                # A figure with no annotation contributes nothing to prose -
                # the page still embeds it, but an extraction prompt gains
                # nothing from a byte count.
                continue
            # Leaf structured node — equation always inlined; table depends on
            # asset_mode.
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
        elif node.chunk is not None:
            # Leaf prose node — full text.
            texts.append(f"{heading_prefix}{node.chunk.text}")
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
                        "type": EdgeType.ENTITY_REFERENCE,
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
                    {"source": node.id, "target": target_id, "type": EdgeType.RELATED, "entity": related_title},
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
                        "type": EdgeType.TITLE_REFERENCE,
                        "entity": other.title,
                    }
                )

    return edges
