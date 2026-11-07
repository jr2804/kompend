"""Page writer for the two-phase compiler.

``upsert_node`` inserts a generated page or merges it into the existing page
with the same slug; ``merge_node`` is the section-aware merge it uses.
"""

from __future__ import annotations

from kompend.core.models import MetadataField, WikiNode
from kompend.core.models.metadata import meta_str_list


def upsert_node(
    new_node: WikiNode,
    nodes: list[WikiNode],
    slug_to_idx: dict[str, int],
) -> None:
    """Insert *new_node* into *nodes*, or merge with an existing same-slug node."""
    slug = new_node.id
    # ponytail: WikiNode.source_chunks must always be a list — ``None`` breaks
    # downstream consumers (dedup, lint, JSON round-trip). A generation
    # branch occasionally set it to None when chunk_ids was empty; coerce here.
    if new_node.source_chunks is None:
        new_node = WikiNode(
            id=new_node.id,
            title=new_node.title,
            content=new_node.content,
            node_type=new_node.node_type,
            source_chunks=[],
            metadata=new_node.metadata,
        )
    if slug in slug_to_idx:
        idx = slug_to_idx[slug]
        nodes[idx] = merge_node(nodes[idx], new_node)
    else:
        slug_to_idx[slug] = len(nodes)
        nodes.append(new_node)


def merge_node(
    existing: WikiNode,
    incoming: WikiNode,
) -> WikiNode:
    """Merge *incoming* node into *existing* using section-aware dedup.

    Sections with the same heading are de-duplicated, keeping the longer
    (richer) version.  Sections unique to either source are appended.
    Provenance lists are unioned.
    """
    existing_text = existing.content or ""
    incoming_text = incoming.content or ""

    if not existing_text.strip():
        merged_content = incoming_text
    elif not incoming_text.strip():
        merged_content = existing_text
    else:
        ex_sections = _split_sections(existing_text)
        inc_sections = _split_sections(incoming_text)
        merged_content = _merge_sections(ex_sections, inc_sections)

    merged_title = existing.title or incoming.title

    # Merge provenance
    # ponytail: LLM output may return dicts/non-strings in provenance/related
    # (e.g. {"chunk": ...} under timeout stress); coerce to str so the set
    # dedup never hits "unhashable type: 'dict'".
    existing_prov = set(meta_str_list(existing.metadata.get(MetadataField.PROVENANCE)))
    incoming_prov = set(meta_str_list(incoming.metadata.get(MetadataField.PROVENANCE)))
    merged_prov = sorted(existing_prov | incoming_prov)

    # Merge related
    existing_rel = set(meta_str_list(existing.metadata.get(MetadataField.RELATED)))
    incoming_rel = set(meta_str_list(incoming.metadata.get(MetadataField.RELATED)))
    merged_rel = sorted(existing_rel | incoming_rel)

    return WikiNode(
        id=existing.id,
        title=merged_title,
        content=merged_content,
        node_type=existing.node_type,
        source_chunks=list(
            dict.fromkeys(
                (existing.source_chunks or []) + (incoming.source_chunks or []),
            ),
        ),
        metadata={
            **existing.metadata,
            MetadataField.PROVENANCE: merged_prov,
            MetadataField.RELATED: merged_rel,
        },
    )


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown text into sections keyed by heading.

    Content before the first heading is stored under the empty-string key.
    """
    sections: dict[str, str] = {}
    current_heading = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("## "):
            if current_lines:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = stripped[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _merge_sections(existing: dict[str, str], incoming: dict[str, str]) -> str:
    """Merge two section dicts, keeping the richer version per heading."""
    merged: dict[str, str] = {}
    for heading in dict.fromkeys(list(existing) + list(incoming)):
        ex = existing.get(heading, "")
        inc = incoming.get(heading, "")
        merged[heading] = ex if len(ex) >= len(inc) else inc

    result = ""
    for heading, body in merged.items():
        if heading:
            result += f"## {heading}\n{body}\n\n"
        elif body:
            result += f"{body}\n\n"
    return result.strip()
