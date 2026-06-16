"""Knowledge graph compiler.

Groups resolved chunks by heading_context into WikiNodes and builds
cross-document entity edges.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from knox.compilation.base import BaseCompiler
from knox.core.models import CompiledWiki, ResolvedChunk, WikiNode

logger = logging.getLogger(__name__)


class GraphCompiler(BaseCompiler):
    """Blind compilation: assemble resolved chunks into a knowledge graph.

    Groups chunks by *heading_context* to form top-level sections.
    Each section becomes a single :class:`WikiNode` with its content merged.
    Cross-document entity links extracted from :attr:`ResolvedChunk.entities`
    become edges in the graph.

    Parameters
    ----------
    model:
        Model name (reserved for future LLM-guided compilation).
    """

    def __init__(self, model: str, *, min_content_length: int = 0) -> None:
        self.model = model
        self._min_content_length = min_content_length

    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Blind compilation of resolved chunks into a knowledge graph.

        Returns a :class:`CompiledWiki` - never raises.
        """
        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={
                    "chunk_count": 0,
                    "source_count": 0,
                    "source_names": [],
                    "model": self.model,
                    "kinds": [],
                },
            )

        groups = self._group_chunks(chunks)
        if self._min_content_length > 0:
            groups = self.filter_empty_groups(
                groups,
                min_content_length=self._min_content_length,
            )
        nodes, node_index = self._build_nodes(groups)
        self.disambiguate_duplicate_titles(nodes, chunks)
        edges, entity_location = self._build_entity_edges(chunks, nodes, node_index)
        edges.extend(self._build_typed_edges(chunks, nodes, entity_location))
        toc = self._build_toc(nodes)

        source_count, source_names = self._source_stats(chunks)

        kind_counts: dict[str, int] = defaultdict(int)
        for n in nodes:
            kind_counts[n.node_type] += 1

        return CompiledWiki(
            nodes=nodes,
            edges=edges,
            toc=toc,
            metadata={
                "chunk_count": len(chunks),
                "source_count": source_count,
                "source_names": source_names,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "model": self.model,
                "kinds": dict(kind_counts),
            },
        )

    def _build_nodes(
        self,
        groups: dict[str, list[ResolvedChunk]],
    ) -> tuple[list[WikiNode], dict[str, int]]:
        """Create one WikiNode per heading group and a lookup index."""
        nodes: list[WikiNode] = []
        node_index: dict[str, int] = {}

        for idx, (heading, group_chunks) in enumerate(groups.items()):
            title = heading if heading != "_root" else "Overview"
            merged_text = BaseCompiler.deduplicate_group_text(group_chunks)
            source_ids = list({rc.chunk.id for rc in group_chunks if rc.chunk.id})

            entity_names = {str(ent.get("name", "")).lower() for rc in group_chunks for ent in rc.entities if isinstance(ent, dict) and ent.get("name")}
            kind = self._infer_page_kind(title, heading, entity_names)

            node = WikiNode(
                id=f"node_{idx}",
                title=title,
                content=merged_text,
                node_type=kind,
                source_chunks=source_ids,
                metadata={
                    "heading_context": heading,
                    "chunk_count": len(group_chunks),
                    "frontmatter": {"kind": kind},
                },
            )
            nodes.append(node)
            node_index[heading] = idx

        return nodes, node_index

    @staticmethod
    def _infer_page_kind(title: str, heading: str, entity_names: set[str]) -> str:
        """Infer a built-in page kind from heading, title, and resolved entities.

        The heuristic is intentionally simple: it favours ``overview`` for root
        sections, ``comparison`` for comparison headings, ``entity`` when the
        title itself was extracted as an entity, and ``concept`` otherwise.
        """
        lowered = title.lower()
        if heading == "_root" or lowered == "overview":
            return "overview"
        if any(keyword in lowered for keyword in ("comparison", "compare", "versus", " vs ", " vs. ")):
            return "comparison"
        if lowered in entity_names:
            return "entity"
        return "concept"

    @staticmethod
    def _source_stats(chunks: list[ResolvedChunk]) -> tuple[int, list[str]]:
        """Count unique source files and return ordered file names.

        Chunks without a ``source_file`` are ignored for the name list but
        still contribute to the overall source count only when a file is
        known. This keeps the reported source count tied to actual documents.
        """
        names: list[str] = []
        seen: set[str] = set()
        for rc in chunks:
            src = rc.chunk.source_file
            if src is None:
                continue
            name = Path(src).name
            if name not in seen:
                seen.add(name)
                names.append(name)
        return len(names), names

    @staticmethod
    def _group_chunks(
        chunks: list[ResolvedChunk],
    ) -> dict[str, list[ResolvedChunk]]:
        """Group chunks by heading_context."""
        groups: dict[str, list[ResolvedChunk]] = defaultdict(list)
        for rc in chunks:
            key = rc.chunk.heading_context or "_root"
            groups[key].append(rc)
        return groups

    @staticmethod
    def _build_entity_edges(
        chunks: list[ResolvedChunk],
        nodes: list[WikiNode],
        node_index: dict[str, int],
    ) -> tuple[list[dict[str, Any]], dict[str, set[int]]]:
        """Build entity cross-reference edges and the entity-location index."""
        edges: list[dict[str, Any]] = []
        entity_location: dict[str, set[int]] = defaultdict(set)

        for rc in chunks:
            key = rc.chunk.heading_context or "_root"
            node_idx = node_index.get(key)
            if node_idx is None:
                continue
            for ent in rc.entities:
                name = ent.get("name", "") if isinstance(ent, dict) else str(ent)
                if name:
                    entity_location[name].add(node_idx)

        for name, node_indices in entity_location.items():
            sorted_indices = sorted(node_indices)
            if len(sorted_indices) > 1:
                for i in range(len(sorted_indices)):
                    for j in range(i + 1, len(sorted_indices)):
                        edges.append(
                            {
                                "source": nodes[sorted_indices[i]].id,
                                "target": nodes[sorted_indices[j]].id,
                                "type": "entity_reference",
                                "entity": name,
                            }
                        )

        return edges, entity_location

    @staticmethod
    def _build_typed_edges(
        chunks: list[ResolvedChunk],
        nodes: list[WikiNode],
        entity_location: dict[str, set[int]],
    ) -> list[dict[str, Any]]:
        """Build typed relation edges from resolver-extracted relations."""
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for rc in chunks:
            for rel in rc.relations:
                rel_type = rel.get("type", "related")
                confidence = rel.get("confidence")
                source_name = rel.get("source")
                target_name = rel.get("target")
                if not source_name or not target_name:
                    continue
                source_indices = entity_location.get(str(source_name), set())
                target_indices = entity_location.get(str(target_name), set())
                for src_idx in source_indices:
                    for tgt_idx in target_indices:
                        if src_idx == tgt_idx:
                            continue
                        src_id = nodes[src_idx].id
                        tgt_id = nodes[tgt_idx].id
                        edge_key = (src_id, tgt_id, rel_type)
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)
                        edge: dict[str, Any] = {
                            "source": src_id,
                            "target": tgt_id,
                            "type": rel_type,
                        }
                        if confidence is not None:
                            edge["confidence"] = float(confidence)
                        edges.append(edge)

        return edges

    @staticmethod
    def _build_toc(nodes: list[WikiNode]) -> list[dict[str, Any]]:
        """Build a top-level table of contents from nodes."""
        return [{"title": n.title, "id": n.id, "level": 1} for n in nodes]
