"""Knowledge graph compiler.

Groups resolved chunks by ``heading_context`` **and** ``source_file`` into
WikiNodes and builds cross-document entity edges. Keeping different source
documents separate prevents generic sections (e.g., ``Introduction``) from
different papers from being mixed on a single wiki page.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

from kompend.compilation.base import BaseCompiler
from kompend.compilation.concept_keys import is_structural_concept
from kompend.core.config import DEFAULTS
from kompend.core.config.keys import DISPLAY_EXCERPT_CHARS
from kompend.core.models import ChunkTree, CompiledWiki, GraphEdge, MetadataField, ResolvedChunk, TocEntry, WikiField, WikiNode
from kompend.logging import get_logger

logger = get_logger(__name__)


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

    def __init__(
        self,
        model: str,
        *,
        min_content_length: int = 0,
        min_concept_sources: int = 0,
        max_concept_pages: int = 0,
        display: DisplayConfig | None = None,
    ) -> None:
        self.model = model
        self._min_content_length = min_content_length
        self._min_concept_sources = min_concept_sources
        self._max_concept_pages = max_concept_pages
        self._display = display if display is not None else DisplayConfig()

    def compile(
        self,
        chunks: list[ResolvedChunk],
        *,
        trees: list[ChunkTree] | None = None,
        pinned_facts: list[str] | None = None,
    ) -> CompiledWiki:
        """Blind compilation of resolved chunks into a knowledge graph.

        When *trees* are provided, structured data from ChunkNode leaves
        is attached to the corresponding WikiNode's ``structured_sections``
        field, enabling downstream consumers to access typed table/equation/
        figure data directly.

        ``pinned_facts`` is accepted for interface symmetry with synthesis
        compilers but ignored here: this compiler rearranges raw chunk text
        rather than generating new content, so there is no prompt into which
        a preservation constraint could be injected.

        Returns a :class:`CompiledWiki` - never raises.
        """
        del pinned_facts  # accepted for BaseCompiler symmetry, not used
        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={
                    WikiField.CHUNK_COUNT: 0,
                    MetadataField.SOURCE_COUNT: 0,
                    MetadataField.SOURCE_NAMES: [],
                    MetadataField.MODEL: self.model,
                    MetadataField.KINDS: [],
                },
            )

        groups = self._group_chunks(chunks)
        if self._min_content_length > 0:
            groups = self.filter_empty_groups(
                dict(groups),
                min_content_length=self._min_content_length,
            )
        nodes, node_index = self._build_nodes(dict(groups))
        self.disambiguate_duplicate_titles(nodes, chunks)
        edges, entity_location = self._build_entity_edges(chunks, nodes, node_index)
        edges.extend(self._build_typed_edges(chunks, nodes, entity_location))
        edges.extend(self._build_source_sibling_edges(nodes, edges))

        if trees:
            self._attach_structured_sections(nodes, chunks, trees)

        if self._min_concept_sources > 0:
            concept_nodes, concept_edges = self._build_concept_pages(
                nodes,
                entity_location,
                min_sources=self._min_concept_sources,
                max_pages=self._max_concept_pages,
            )
            nodes.extend(concept_nodes)
            edges.extend(concept_edges)

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
                WikiField.CHUNK_COUNT: len(chunks),
                MetadataField.SOURCE_COUNT: source_count,
                MetadataField.SOURCE_NAMES: source_names,
                MetadataField.NODE_COUNT: len(nodes),
                MetadataField.EDGE_COUNT: len(edges),
                MetadataField.MODEL: self.model,
                MetadataField.KINDS: dict(kind_counts),
            },
        )

    def _build_nodes(
        self,
        groups: dict[tuple[str, str | None], list[ResolvedChunk]],
    ) -> tuple[list[WikiNode], dict[tuple[str, str | None], int]]:
        """Create one WikiNode per heading/source group and a lookup index."""
        nodes: list[WikiNode] = []
        node_index: dict[tuple[str, str | None], int] = {}

        for idx, ((heading, source), group_chunks) in enumerate(groups.items()):
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
                    MetadataField.HEADING_CONTEXT: heading,
                    MetadataField.SOURCE_FILE: source,
                    WikiField.CHUNK_COUNT: len(group_chunks),
                    "frontmatter": {"kind": kind},
                    MetadataField.TAGS: list({tag for rc in group_chunks for tag in (rc.chunk.metadata.get(MetadataField.TAGS) or [])}),
                    MetadataField.ATTRIBUTES: _collect_attributes(group_chunks),
                },
            )
            nodes.append(node)
            node_index[(heading, source)] = idx

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
    def _group_key(rc: ResolvedChunk) -> tuple[str, str | None]:
        """Return the grouping key for a resolved chunk.

        Chunks are grouped by ``heading_context`` and by ``source_file`` so
        that generic sections (e.g., ``Introduction``) from different
        source documents do not end up mixed on the same wiki page.
        """
        heading = rc.chunk.heading_context or "_root"
        source = str(rc.chunk.source_file) if rc.chunk.source_file else None
        return (heading, source)

    @staticmethod
    def _group_chunks(
        chunks: list[ResolvedChunk],
    ) -> dict[tuple[str, str | None], list[ResolvedChunk]]:
        """Group chunks by heading_context and source_file."""
        groups: dict[tuple[str, str | None], list[ResolvedChunk]] = defaultdict(list)
        for rc in chunks:
            groups[GraphCompiler._group_key(rc)].append(rc)
        return groups

    @staticmethod
    def _build_entity_edges(
        chunks: list[ResolvedChunk],
        nodes: list[WikiNode],
        node_index: dict[tuple[str, str | None], int],
    ) -> tuple[list[GraphEdge], dict[str, set[int]]]:
        """Build entity cross-reference edges and the entity-location index."""
        edges: list[GraphEdge] = []
        entity_location: dict[str, set[int]] = defaultdict(set)

        for rc in chunks:
            key = GraphCompiler._group_key(rc)
            node_idx = node_index.get(key)
            if node_idx is None:
                continue
            for ent in rc.entities:
                name = ent.get("name", "") if isinstance(ent, dict) else str(ent)
                # Skip VLM bookkeeping names so they never become concept pages.
                if name and name not in GraphCompiler._RESERVED_ENTITY_NAMES and not name.startswith("vlm_"):
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
    ) -> list[GraphEdge]:
        """Build typed relation edges from resolver-extracted relations."""
        edges: list[GraphEdge] = []
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
                        edge: GraphEdge = {
                            "source": src_id,
                            "target": tgt_id,
                            "type": rel_type,
                        }
                        if confidence is not None:
                            edge["confidence"] = float(confidence)
                        edges.append(edge)

        return edges

    @staticmethod
    def _build_source_sibling_edges(
        nodes: list[WikiNode],
        existing_edges: list[GraphEdge],
    ) -> list[GraphEdge]:
        """Create bidirectional edges between consecutive same-source nodes.

        Pages from a multi-section source document are linked in reading
        order so every page has at least one incoming link. Nodes without
        a ``source_file`` (provenance unknown) are skipped.
        """
        seen: set[tuple[str, str]] = {(e["source"], e["target"]) for e in existing_edges}

        by_source: dict[str, list[int]] = defaultdict(list)
        for idx, node in enumerate(nodes):
            source = node.metadata.get(MetadataField.SOURCE_FILE)
            if source is not None:
                by_source[source].append(idx)

        new_edges: list[GraphEdge] = []
        for indices in by_source.values():
            if len(indices) < 2:
                continue
            for i in range(len(indices) - 1):
                src_id = nodes[indices[i]].id
                tgt_id = nodes[indices[i + 1]].id
                for a, b in [(src_id, tgt_id), (tgt_id, src_id)]:
                    if (a, b) not in seen:
                        seen.add((a, b))
                        new_edges.append({"source": a, "target": b, "type": "sibling"})
        return new_edges

    @staticmethod
    def _build_concept_pages(
        nodes: list[WikiNode],
        entity_location: dict[str, set[int]],
        *,
        min_sources: int,
        max_pages: int,
    ) -> tuple[list[WikiNode], list[GraphEdge]]:
        """Create concept pages for entities spanning multiple source documents.

        Each entity appearing in nodes from at least *min_sources* distinct
        ``source_file`` values gets a synthesised page that aggregates
        relevant excerpts from every mentioning section.  Bidirectional
        edges connect the concept page to each source section.
        """
        node_sources: dict[str, str | None] = {n.id: n.metadata.get(MetadataField.SOURCE_FILE) for n in nodes}

        candidates: list[tuple[str, set[int], set[str]]] = []
        for entity_name, node_indices in entity_location.items():
            if not entity_name or not entity_name.strip():
                continue
            # Drop table/spreadsheet structural artifacts (column headers,
            # field labels, generic metadata labels) the same way the
            # TwoPhase backend does, so they never become concept pages on
            # either compiler.
            if is_structural_concept(entity_name):
                continue
            sources: set[str] = set()
            for idx in node_indices:
                src = node_sources.get(nodes[idx].id)
                # Skip pseudo-source ids emitted by the table layer (e.g.
                # ``table.data_sources``) — they are not real files and
                # only pollute concept provenance.
                if src is not None and not src.startswith("table"):
                    sources.add(src)
            if len(sources) >= min_sources:
                candidates.append((entity_name, node_indices, sources))

        candidates.sort(key=lambda c: len(c[2]), reverse=True)
        if max_pages > 0:
            candidates = candidates[:max_pages]

        concept_nodes: list[WikiNode] = []
        concept_edges: list[GraphEdge] = []
        entity_lower = ""

        for entity_name, node_indices, sources in candidates:
            concept_id = f"concept_{len(concept_nodes)}"
            entity_lower = entity_name.lower()
            excerpt_chars = DEFAULTS.get(DISPLAY_EXCERPT_CHARS, 300)
            excerpts: list[str] = []
            for idx in sorted(node_indices):
                node = nodes[idx]
                excerpt = ""
                for para in (node.content or "").split("\n\n"):
                    if entity_lower in para.lower():
                        excerpt = para.strip()[:excerpt_chars]
                        break
                if not excerpt:
                    excerpt = (node.content or "")[:excerpt_chars].strip()
                if excerpt:
                    excerpts.append(f"**{node.title}**\n\n{excerpt}")

            content = "\n\n---\n\n".join(excerpts) if excerpts else f"*{entity_name}*"

            concept_node = WikiNode(
                id=concept_id,
                title=entity_name,
                content=content,
                node_type="concept",
                source_chunks=[],
                metadata={
                    "cross_cutting": True,
                    MetadataField.SOURCE_COUNT: len(sources),
                    "referenced_nodes": [nodes[idx].id for idx in sorted(node_indices)],
                    "frontmatter": {"kind": "concept"},
                },
            )
            concept_nodes.append(concept_node)

            for idx in sorted(node_indices):
                target_id = nodes[idx].id
                concept_edges.append({"source": concept_id, "target": target_id, "type": "concept_reference"})
                concept_edges.append({"source": target_id, "target": concept_id, "type": "concept_reference"})

        return concept_nodes, concept_edges

    @staticmethod
    def _build_toc(nodes: list[WikiNode]) -> list[TocEntry]:
        """Build a top-level table of contents from nodes."""
        return [{"title": n.title, "id": n.id, "level": 1} for n in nodes]

    @staticmethod
    def _attach_structured_sections(
        nodes: list[WikiNode],
        chunks: list[ResolvedChunk],
        trees: list[ChunkTree],
    ) -> None:
        """Attach structured data from ChunkTree leaves to WikiNodes.

        For each WikiNode, finds matching ChunkNode leaves (by heading
        context and source file) and populates ``structured_sections``
        with typed table/equation/figure data.
        """
        tree_map: dict[str, ChunkTree] = {}
        for t in trees:
            if t.source_file is not None:
                tree_map[str(t.source_file)] = t

        chunk_source: dict[str, str] = {}
        for rc in chunks:
            if rc.chunk.source_file and rc.chunk.id:
                chunk_source[rc.chunk.id] = str(rc.chunk.source_file)

        for node in nodes:
            heading = node.metadata.get(MetadataField.HEADING_CONTEXT, "_root")
            source = node.metadata.get(MetadataField.SOURCE_FILE)
            if source is None:
                continue
            tree = tree_map.get(source)
            if tree is None:
                continue

            structured_sections: list[dict[str, Any]] = []
            for chunk_node in tree.walk():
                if chunk_node.structured is None:
                    continue
                chunk_heading = chunk_node.chunk.heading_context or "_root" if chunk_node.chunk is not None else heading
                if chunk_heading != heading:
                    continue
                structured_sections.append(
                    {
                        "type": type(chunk_node.structured).__name__,
                        "data": asdict(chunk_node.structured),
                        "heading_chain": list(chunk_node.heading_chain),
                    }
                )

            if structured_sections:
                node.structured_sections["items"] = structured_sections

    # Internal names injected by VLMEnricher for bookkeeping. These must
    # never become concept pages or cross-reference edges in the output.
    _RESERVED_ENTITY_NAMES: ClassVar[frozenset[str]] = frozenset({"vlm_filtered", "vlm_skipped", "vlm_error"})


def _collect_attributes(group_chunks: list[ResolvedChunk]) -> dict[str, list[str]]:
    """Aggregate attributes from a chunk group, preserving per-key values."""
    attr_values: dict[str, list[str]] = {}
    for rc in group_chunks:
        attrs = rc.chunk.metadata.get(MetadataField.ATTRIBUTES) if hasattr(rc, "chunk") else None
        if not attrs:
            continue
        for key, val in attrs.items():
            seen = attr_values.setdefault(key, [])
            if val not in seen:
                seen.append(val)
    return attr_values
