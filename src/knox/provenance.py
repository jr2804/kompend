"""N4 claim-level provenance: ``^[source.pdf:42-58]`` citations.

Provenance lives in two places:

- :class:`WikiNode.metadata["provenance"]` — a list of source labels per node.
- Inline markers inside ``WikiNode.content`` — parsed by :func:`extract_citations`
  and exposed via :func:`knox provenance <slug>`.

The format is intentionally simple and human-readable so it can survive
a round-trip through the markdown output writer.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from knox.core.models import CompiledWiki, WikiNode

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\^\[([^\]]+?):(\d+)-(\d+)\]")

LIFECYCLE_DRAFT = "draft"
LIFECYCLE_REVIEWED = "reviewed"
LIFECYCLE_VERIFIED = "verified"
LIFECYCLE_STALE = "stale"
LIFECYCLE_ARCHIVED = "archived"

VALID_LIFECYCLE_STATES: tuple[str, ...] = (
    LIFECYCLE_DRAFT,
    LIFECYCLE_REVIEWED,
    LIFECYCLE_VERIFIED,
    LIFECYCLE_STALE,
    LIFECYCLE_ARCHIVED,
)


@dataclass
class Citation:
    """A single ``^[source:start-end]`` citation."""

    source: str
    start: int
    end: int
    context: str = ""

    def to_marker(self) -> str:
        return f"^[{self.source}:{self.start}-{self.end}]"


def extract_citations(text: str) -> list[Citation]:
    """Parse all ``^[source:42-58]`` citations in *text*."""
    out: list[Citation] = []
    for match in _CITATION_RE.finditer(text):
        out.append(
            Citation(
                source=match.group(1),
                start=int(match.group(2)),
                end=int(match.group(3)),
            )
        )
    return out


def annotate_wiki_with_provenance(wiki: CompiledWiki, output_dir: Path) -> CompiledWiki:
    """Persist provenance citations for each node and return a new wiki.

    Writes a per-node ``.knox/provenance/<id>.json`` file containing the
    list of citations parsed from the node's content. The wiki itself
    is returned with updated metadata so callers can chain output steps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prov_dir = output_dir / ".knox" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    new_nodes: list[WikiNode] = []
    for node in wiki.nodes:
        citations = extract_citations(node.content)
        record = {
            "node_id": node.id,
            "title": node.title,
            "citations": [c.to_marker() for c in citations],
        }
        (prov_dir / f"{node.id}.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
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


def merge_duplicate_citations(citations: Iterable[Citation]) -> list[Citation]:
    """Remove duplicates by (source, start, end), preserving first-seen order."""
    seen: set[tuple[str, int, int]] = set()
    out: list[Citation] = []
    for c in citations:
        key = (c.source, c.start, c.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
