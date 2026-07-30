"""Assign stable content-addressed ids to structured-asset leaves in a chunk tree.

Walks the tree in document order, and for each ``StructuredTable`` /
``StructuredFigure`` leaf (equations deferred) registers an
:class:`~knox.core.models.assets.AssetRecord` in the
:class:`~knox.core.pipeline.assets.AssetStore` and records the mapping from
the :class:`~knox.core.models.trees.ChunkNode` id to the asset id.

Content-addressed ids (``tab-<hash>``, ``fig-<hash>``) mean
identical tables/figures across sources dedup to a single record; identical
content always yields the same id (deterministic across runs).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from knox.compilation.asset_summarizer import summarize_asset
from knox.core.models.assets import AssetRecord
from knox.core.models.enums import AssetKind
from knox.core.models.structured import StructuredEquation, StructuredFigure, StructuredTable, structured_to_dict
from knox.core.models.trees import ChunkTree
from knox.core.pipeline.assets import AssetStore

__all__ = ["ASSET_ID_HASH_LEN", "AssetIdMap", "asset_id_for", "assign_asset_ids"]

#: Length of the SHA-256 hex prefix used in asset ids.
ASSET_ID_HASH_LEN = 6


@dataclass
class AssetIdMap:
    """Mapping from :class:`ChunkNode` id to assigned asset id, plus lookups.

    The ``label_to_asset`` mapping lets the mention rewriter turn a
    source-relative reference like "Figure 2" into the asset id for the
    second figure in document order.
    """

    node_to_asset: dict[str, str] = field(default_factory=dict)
    """``ChunkNode.id -> asset id`` for every registered structured leaf."""

    label_to_asset: dict[tuple[AssetKind, int], str] = field(default_factory=dict)
    """``(kind, source-relative number) -> asset id`` for label rewriting."""

    def get(self, node_id: str) -> str | None:
        """Return the asset id assigned to ``node_id``, or ``None``."""
        return self.node_to_asset.get(node_id)

    def asset_for_label(self, kind: AssetKind, number: int) -> str | None:
        """Return the asset id for the ``number``-th ``kind`` in this source."""
        return self.label_to_asset.get((kind, number))

    def __len__(self) -> int:
        return len(self.node_to_asset)


def assign_asset_ids(tree: ChunkTree, store: AssetStore, *, source: str = "") -> AssetIdMap:
    """Register structured-asset leaves and return the node→asset id map.

    Walks the tree in document order (depth-first from root). For each leaf
    whose ``structured`` is a table or figure (equations are skipped — Phase 1
    leaves them inlined), computes the content-addressed id, upserts an
    :class:`AssetRecord` (incrementing ``ref_count`` for repeats), and records
    the ``ChunkNode.id -> asset id`` mapping.

    Parameters
    ----------
    tree:
        The chunk tree to walk.
    store:
        The asset store to populate (mutated in place; caller saves).
    source:
        Source file path/name, stored on the record for provenance.
    """
    id_map = AssetIdMap()
    counters: dict[AssetKind, int] = {}
    for node in tree.walk():
        if node.structured is None:
            continue
        kind = _kind_of(node.structured)
        if kind is None:
            continue  # equation or unknown — deferred
        record_dict = structured_to_dict(node.structured)
        asset_id = asset_id_for(kind, record_dict)
        if asset_id in store:
            store.increment_ref(asset_id)
        else:
            store.put(
                AssetRecord(
                    id=asset_id,
                    kind=kind,
                    source=source,
                    data=node.structured,
                    summary=summarize_asset(node.structured),
                    ref_count=1,
                )
            )
        id_map.node_to_asset[node.id] = asset_id
        number = counters.get(kind, 0) + 1
        counters[kind] = number
        id_map.label_to_asset[(kind, number)] = asset_id
    return id_map


def asset_id_for(kind: AssetKind, record_dict: dict) -> str:
    """Compute the stable content-addressed asset id for a record dict.

    Deterministic: same content → same id. Hash is over the canonical
    serialized form (sorted keys) so field order does not affect the id.
    """
    raw = json.dumps(record_dict, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(raw.encode()).hexdigest()[:ASSET_ID_HASH_LEN]
    return f"{kind.prefix}-{h}"


def _kind_of(structured: object) -> AssetKind | None:
    """Return the :class:`AssetKind` for a structured record, or ``None`` to skip."""
    if isinstance(structured, StructuredTable):
        return AssetKind.TABLE
    if isinstance(structured, StructuredFigure):
        return AssetKind.FIGURE
    if isinstance(structured, StructuredEquation):
        return None  # deferred
    return None
