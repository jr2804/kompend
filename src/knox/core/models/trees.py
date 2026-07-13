"""Hierarchical chunk tree (Tier 2 output that replaces flat ``list[Chunk]``)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.core.models.chunks import Chunk
from knox.core.models.enums import (
    ChunkNodeField,
    ChunkStrategy,
    ChunkTreeField,
    MetadataField,
)
from knox.core.models.structured import StructuredData, _structured_from_dict

CHUNK_TREE_CACHE_VERSION: int = 1
"""Increment when ``ChunkTree.to_dict`` / ``ChunkTree.from_dict`` format changes."""


@dataclass
class ChunkNode:
    """A node in the hierarchical chunk tree.

    Leaf nodes carry content (``chunk`` for prose, ``structured`` for
    tables/equations/figures). Internal nodes carry a ``summary`` and
    aggregate ``token_count`` of their subtree.
    """

    id: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    heading_path: tuple[str, ...] = ()
    chunk: Chunk | None = None
    structured: StructuredData | None = None
    summary: str = ""
    token_count: int = 0
    contextual_context: str = ""
    contextual_embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            ChunkNodeField.ID: self.id,
            ChunkNodeField.PARENT_ID: self.parent_id,
            ChunkNodeField.CHILDREN_IDS: self.children_ids,
            ChunkNodeField.DEPTH: self.depth,
            ChunkNodeField.HEADING_PATH: list(self.heading_path),
            ChunkNodeField.CHUNK: self.chunk.to_dict() if self.chunk else None,
            ChunkNodeField.STRUCTURED: self.structured.to_dict() if self.structured else None,
            ChunkNodeField.SUMMARY: self.summary,
            ChunkNodeField.TOKEN_COUNT: self.token_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkNode:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            id=d[ChunkNodeField.ID],
            parent_id=d.get(ChunkNodeField.PARENT_ID),
            children_ids=d.get(ChunkNodeField.CHILDREN_IDS, []),
            depth=d.get(ChunkNodeField.DEPTH, 0),
            heading_path=tuple(d.get(ChunkNodeField.HEADING_PATH, ())),
            chunk=Chunk.from_dict(d[ChunkNodeField.CHUNK]) if d.get(ChunkNodeField.CHUNK) else None,
            structured=_structured_from_dict(d[ChunkNodeField.STRUCTURED]) if d.get(ChunkNodeField.STRUCTURED) else None,
            summary=d.get(ChunkNodeField.SUMMARY, ""),
            token_count=d.get(ChunkNodeField.TOKEN_COUNT, 0),
        )


@dataclass
class ChunkTree:
    """Hierarchical tree of ChunkNodes for one source document.

    Replaces the flat ``list[Chunk]`` produced by the legacy chunker. The tree
    mirrors the document's heading structure, enabling multi-resolution
    retrieval and structured content preservation.
    """

    root_id: str
    nodes: dict[str, ChunkNode] = field(default_factory=dict)
    source_file: Path | None = None
    strategy: ChunkStrategy = ChunkStrategy.SINGLE_PASS

    def leaves(self) -> list[ChunkNode]:
        """Return all leaf nodes (nodes with no children)."""
        return [n for n in self.nodes.values() if not n.children_ids]

    def walk(self) -> Iterator[ChunkNode]:
        """Depth-first traversal from root."""
        yield from self._walk(self.root_id)

    def get_path(self, node_id: str) -> list[ChunkNode]:
        """Return the path from root to ``node_id`` (inclusive)."""
        path: list[ChunkNode] = []
        current = self.nodes.get(node_id)
        while current is not None:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def siblings(self, node_id: str) -> list[ChunkNode]:
        """Return siblings of ``node_id`` (excluding itself)."""
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return []
        parent = self.nodes[node.parent_id]
        return [self.nodes[cid] for cid in parent.children_ids if cid != node_id]

    def subtree_tokens(self, node_id: str) -> int:
        """Total token count of a node's subtree (node + all descendants)."""
        node = self.nodes.get(node_id)
        if node is None:
            return 0
        total = node.token_count
        for child_id in node.children_ids:
            total += self.subtree_tokens(child_id)
        return total

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for caching."""
        return {
            ChunkTreeField.VERSION: CHUNK_TREE_CACHE_VERSION,
            ChunkTreeField.ROOT_ID: self.root_id,
            ChunkTreeField.NODES: {nid: n.to_dict() for nid, n in self.nodes.items()},
            ChunkTreeField.SOURCE_FILE: self.source_file.as_posix() if self.source_file else None,
            ChunkTreeField.STRATEGY: str(self.strategy),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkTree:
        """Deserialize from a dict produced by :meth:`to_dict`.

        Raises ``ValueError`` on cache-version mismatch so callers can
        regenerate instead of reading a stale format.
        """
        version = d.get(ChunkTreeField.VERSION, 0)
        if version != CHUNK_TREE_CACHE_VERSION:
            msg = f"ChunkTree cache version mismatch: expected {CHUNK_TREE_CACHE_VERSION}, got {version}"
            raise ValueError(msg)
        src = d.get(ChunkTreeField.SOURCE_FILE) or d.get(MetadataField.SOURCE_PATH)
        return cls(
            root_id=d[ChunkTreeField.ROOT_ID],
            nodes={nid: ChunkNode.from_dict(n) for nid, n in d.get(ChunkTreeField.NODES, {}).items()},
            source_file=Path(src) if src else None,
            strategy=ChunkStrategy(d.get(ChunkTreeField.STRATEGY, ChunkStrategy.SINGLE_PASS)),
        )

    def _walk(self, node_id: str) -> Iterator[ChunkNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return
        yield node
        for child_id in node.children_ids:
            yield from self._walk(child_id)


__all__ = ["CHUNK_TREE_CACHE_VERSION", "ChunkNode", "ChunkTree"]
