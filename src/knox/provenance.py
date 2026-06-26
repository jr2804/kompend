"""N4 claim-level provenance: ``^[source.pdf:42-58]`` citations.

Provenance lives in two places:

- :class:`WikiNode.metadata["provenance"]` — a list of source labels per node.
- Inline markers inside ``WikiNode.content`` — parsed by :func:`extract_citations`
  and exposed via :func:`knox provenance <slug>`.

The format is intentionally simple and human-readable so it can survive
a round-trip through the markdown output writer.

Note: ``Citation``, ``extract_citations``, ``merge_duplicate_citations``,
and lifecycle constants moved to :mod:`knox.core.models`. Import from there.
"""

from __future__ import annotations

import json
from pathlib import Path

from knox.core.models import (
    Citation,
    CompiledWiki,
    WikiNode,
    extract_citations,
)
from knox.core.paths import VaultPaths
from knox.logging import get_logger

logger = get_logger(__name__)


def annotate_wiki_with_provenance(wiki: CompiledWiki, output_dir: Path) -> CompiledWiki:
    """Persist provenance citations for each node and return a new wiki.

    Writes a per-node ``.knox/provenance/<id>.json`` file containing the
    list of citations parsed from the node's content. The wiki itself
    is returned with updated metadata so callers can chain output steps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prov_dir = VaultPaths(output_dir.parent).provenance_dir
    prov_dir.mkdir(parents=True, exist_ok=True)

    new_nodes: list[WikiNode] = []
    for node in wiki.nodes:
        citations = extract_citations(node.content)
        record = {
            "node_id": node.id,
            "title": node.title,
            "citations": [c.to_marker() for c in citations],
        }
        (prov_dir / f"{node.id}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        new_nodes.append(
            WikiNode(
                id=node.id,
                title=node.title,
                content=node.content,
                node_type=node.node_type,
                source_chunks=node.source_chunks,
                metadata={**(node.metadata or {}), "citation_count": len(citations)},
            )
        )

    return CompiledWiki(
        nodes=new_nodes,
        edges=list(wiki.edges),
        toc=list(wiki.toc),
        quality_score=wiki.quality_score,
        iteration=wiki.iteration,
        metadata={**wiki.metadata, "provenance_annotated": True},
    )


def citations_for_slug(slug_to_node: dict[str, WikiNode]) -> dict[str, list[Citation]]:
    """Convenience helper for the ``knox provenance <slug>`` command."""
    return {slug: extract_citations(node.content) for slug, node in slug_to_node.items()}
