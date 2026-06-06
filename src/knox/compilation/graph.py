"""Knowledge graph compiler.

Groups resolved chunks by heading_context into WikiNodes and builds
cross-document entity edges.
"""

from __future__ import annotations

import logging
from collections import defaultdict

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
    provider:
        Provider prefix (reserved for future LLM-guided compilation).
    """

    def __init__(self, model: str = "local", provider: str = "ollama") -> None:
        self.model = model
        self.provider = provider

    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Blind compilation of resolved chunks into a knowledge graph.

        Returns a :class:`CompiledWiki` — never raises.
        """
        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={"chunk_count": 0, "model": self.model, "provider": self.provider},
            )

        # ------------------------------------------------------------------
        # Step 1: Group chunks by heading_context.
        # ------------------------------------------------------------------
        groups: dict[str, list[ResolvedChunk]] = defaultdict(list)
        for rc in chunks:
            key = rc.chunk.heading_context or "_root"
            groups[key].append(rc)

        # ------------------------------------------------------------------
        # Step 2: Build one WikiNode per group.
        # ------------------------------------------------------------------
        nodes: list[WikiNode] = []
        node_index: dict[str, int] = {}  # heading_context -> node index

        for idx, (heading, group_chunks) in enumerate(groups.items()):
            title = heading if heading != "_root" else "Overview"
            merged_text = "\n\n".join(rc.chunk.text for rc in group_chunks if rc.chunk.text)
            source_ids = list({rc.chunk.id for rc in group_chunks if rc.chunk.id})

            node = WikiNode(
                id=f"node_{idx}",
                title=title,
                content=merged_text,
                node_type="section",
                source_chunks=source_ids,
                metadata={"heading_context": heading, "chunk_count": len(group_chunks)},
            )
            nodes.append(node)
            node_index[heading] = idx

        # ------------------------------------------------------------------
        # Step 3: Build entity cross-reference edges.
        # ------------------------------------------------------------------
        edges: list[dict] = []
        entity_location: dict[str, set[int]] = defaultdict(set)  # entity name -> node indices

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
                # Connect all pairs
                for i in range(len(sorted_indices)):
                    for j in range(i + 1, len(sorted_indices)):
                        edges.append({
                            "source": nodes[sorted_indices[i]].id,
                            "target": nodes[sorted_indices[j]].id,
                            "type": "entity_reference",
                            "entity": name,
                        })
            # Also connect entity mentions within the same node as self-references
            # (not needed — skip self-loops)

        # ------------------------------------------------------------------
        # Step 4: Build table-of-contents.
        # ------------------------------------------------------------------
        toc = [{"title": n.title, "id": n.id, "level": 1} for n in nodes]

        return CompiledWiki(
            nodes=nodes,
            edges=edges,
            toc=toc,
            metadata={
                "chunk_count": len(chunks),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "model": self.model,
                "provider": self.provider,
            },
        )