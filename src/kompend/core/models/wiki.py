"""Compiled wiki graph models (Tier 4: Compilation output) and cache loading."""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NotRequired, TypedDict

import yaml

from kompend.core.models.enums import MetadataField
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import WIKI_SNAPSHOT_TABLE
from kompend.core.utils.time import iso_local

#: Discriminator values for the ``wiki_snapshot`` table. ``last`` is the
#: post-run snapshot (lint/capture/MCP); ``compiled`` is the tier-handoff
#: snapshot (Tier 3 -> Tier 4).
WIKI_SNAPSHOT_LAST: Final[str] = "last"
WIKI_SNAPSHOT_COMPILED: Final[str] = "compiled"

__all__ = [
    "WIKI_SNAPSHOT_COMPILED",
    "WIKI_SNAPSHOT_LAST",
    "CompiledWiki",
    "CrossReferenceIssue",
    "CrossReferenceResult",
    "GraphEdge",
    "TocEntry",
    "WikiNode",
    "load_cached_wiki",
    "load_wiki_from_files",
    "load_wiki_snapshot",
    "write_wiki_snapshot_payload",
]

# Frontmatter ``kind`` extension maps back to the kompend node_type.
_KIND_TO_NODE_TYPE = {
    "Concept": "concept",
    "Entity": "entity",
    "Overview": "overview",
    "Procedure": "procedure",
    "Section": "section",
}

# Non-content pages that are not wiki nodes.
_SKIP_PAGES = {"index.md", "log.md", "references.md"}

# Links: ``[title](path)`` and ``[[wikilink]]``.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\.md\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


@dataclass
class CrossReferenceIssue:
    """A single consistency issue found by cross-reference checking."""

    issue_type: str  # "conflict", "unresolved", "contradiction"
    description: str
    entity: str = ""
    locations: list[str] = field(default_factory=list)


@dataclass
class CrossReferenceResult:
    """Output of Phase 3 cross-reference checking at data level."""

    issues: list[CrossReferenceIssue] = field(default_factory=list)
    entity_count: int = 0
    resolved_count: int = 0


@dataclass
class ExtractionStats:
    """Per-source extraction timing and counts metadata."""

    source: str = ""
    summary: str = ""
    duration_s: float = 0.0
    summary_chars: int = 0
    summary_words: int = 0
    entity_count: int = 0
    concept_count: int = 0
    tag_count: int = 0
    batches: int = 0


@dataclass
class WikiNode:
    """A node in the compiled knowledge graph."""

    id: str
    title: str
    content: str
    node_type: str  # e.g. "concept", "procedure", "entity", "section"
    source_chunks: list[str] | None = None
    structured_sections: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    heading_chain: tuple[str, ...] = field(default_factory=tuple)

    @property
    def description(self) -> str:
        """One-line description: the first sentence of the first prose line.

        The single source of truth for a page's description — the OKF
        frontmatter writer and the vector store's embed text both read this, so
        retrieval and the written index summarise a page identically. Leading
        markdown headings are skipped (generated pages open with ``##``
        subheadings, so the description must be the first sentence of prose and
        not a heading fragment) and ``[[wikilinks]]`` are reduced to their
        display text.
        """
        content = (self.content or "").strip()
        if not content:
            return ""
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first_line = stripped
            break
        else:
            return ""
        text = _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), first_line)
        # ponytail: heuristic sentence split — a boundary needs ". " +
        # capitalized/paren continuation, and pseudo-boundaries inside
        # identifiers like "Rec. P.863" are skipped when the next word itself
        # contains a period. Swap for a tokenizer if this ever misfires.
        desc = text
        for match in re.finditer(r"^(.+?[.!?])\s+(?=[A-Z(])", text):
            next_word = text[match.end() :].split(" ", 1)[0]
            if "." in next_word.rstrip(".,;:"):
                continue
            # Skip boundaries after initials — "John G. Beerends is ..." must
            # not truncate to "John G." (single-letter word before the dot).
            preceding = match.group(1).rstrip(".!?")
            last_word = preceding.split(" ")[-1].strip("(")
            if last_word and len(last_word) == 1:
                continue
            desc = match.group(1)
            break
        else:
            tail = re.match(r"^(.+?[.!?])\s*$", text)
            desc = tail.group(1) if tail else text
        if len(desc) > 200:
            desc = desc[:200].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        return desc


@dataclass
class CompiledWiki:
    """Output of Tier 4: Compilation."""

    nodes: list[WikiNode]
    edges: list[GraphEdge]
    toc: list[TocEntry]
    quality_score: float = 0.0
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphEdge(TypedDict):
    """Schema for one entry in ``CompiledWiki.edges``.

    ``entity`` is only present on ``entity_reference`` edges;
    ``confidence`` only on typed relation edges.
    """

    source: str
    target: str
    type: str
    entity: NotRequired[str]
    confidence: NotRequired[float]


class TocEntry(TypedDict):
    """Schema for one entry in ``CompiledWiki.toc``."""

    title: str
    id: str
    level: int


def write_wiki_snapshot_payload(
    base_dir: Path,
    kind: str,
    *,
    nodes: list[dict[str, Any]],
    edges: list[Any],
    toc: list[Any],
    quality_score: float,
    iteration: int,
    metadata: dict[str, Any],
) -> None:
    """Write a wiki snapshot to the ``wiki_snapshot`` table under *kind*.

    *nodes* are pre-shaped by the caller (the two writers rewrite paths
    differently). Loaded back via :func:`load_wiki_snapshot`.
    """
    with open_store(vault_db_path(base_dir)) as conn:
        conn.execute(
            f"INSERT INTO {WIKI_SNAPSHOT_TABLE}(kind, nodes, edges, toc, quality_score, iteration, metadata, written_at) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            f"ON CONFLICT(kind) DO UPDATE SET nodes = excluded.nodes, edges = excluded.edges, toc = excluded.toc, "
            f"quality_score = excluded.quality_score, iteration = excluded.iteration, "
            f"metadata = excluded.metadata, written_at = excluded.written_at",
            (
                kind,
                json.dumps(nodes, default=str),
                json.dumps(list(edges), default=str),
                json.dumps(list(toc), default=str),
                quality_score,
                iteration,
                json.dumps(dict(metadata), default=str),
                iso_local(),
            ),
        )


def load_cached_wiki(base_dir: Path) -> CompiledWiki | None:
    """Load the last compiled wiki from the on-disk cache.

    Args:
        base_dir: Project root directory (parent of the wiki output directory).

    Returns ``None`` if no cache exists (so callers can prompt the user to
    compile). A corrupt cache file is a bug — it raises so the real cause
    surfaces instead of producing a misleading "run compile first" message.
    Reconstructs nested ``WikiNode`` objects from their dict representation.
    """
    return load_wiki_snapshot(base_dir, WIKI_SNAPSHOT_LAST)


def load_wiki_snapshot(base_dir: Path, kind: str) -> CompiledWiki | None:
    """Load the wiki snapshot of *kind* from the store, or ``None`` on miss."""
    if not vault_db_path(base_dir).exists():
        return None
    with open_store(vault_db_path(base_dir)) as conn:
        row = conn.execute(
            f"SELECT nodes, edges, toc, quality_score, iteration, metadata FROM {WIKI_SNAPSHOT_TABLE} WHERE kind = ?",
            (kind,),
        ).fetchone()
    if row is None:
        return None
    raw_nodes = json.loads(row["nodes"])  # nocheck: wiki_snapshot column, not a metadata key
    nodes = [WikiNode(**n) if isinstance(n, dict) else n for n in raw_nodes]
    return CompiledWiki(
        nodes=nodes,
        edges=json.loads(row["edges"]),  # nocheck: wiki_snapshot column
        toc=json.loads(row["toc"]),  # nocheck: wiki_snapshot column
        quality_score=float(row["quality_score"]),  # nocheck: wiki_snapshot column
        iteration=int(row["iteration"]),  # nocheck: wiki_snapshot column
        metadata=json.loads(row["metadata"]),
    )


def load_wiki_from_files(output_dir: Path) -> CompiledWiki | None:
    """Load a compiled wiki from the generated markdown files alone.

    **Primary loader** for query/read paths — the markdown bundle is the
    source of truth (markdown-only invariant), with no ``.kompend/`` state
    and no LLM. Builds nodes from the ``.md`` files on disk (frontmatter +
    body + links). Returns ``None`` when *output_dir* has no pages.
    """
    if not output_dir.is_dir():
        return None

    nodes: list[WikiNode] = []
    edges: list[GraphEdge] = []
    slug_to_title: dict[str, str] = {}

    files = sorted(p for p in output_dir.rglob("*.md") if p.is_file() and p.name not in _SKIP_PAGES)
    if not files:
        return None

    for path in files:
        slug = path.stem
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        title = str(frontmatter.get("title") or slug.replace("-", " ").title())
        kind = str(frontmatter.get("kind") or _KIND_TO_NODE_TYPE.get(str(frontmatter.get("type", "")), "concept"))
        metadata: dict[str, Any] = {}
        if frontmatter.get("description"):  # nocheck: OKF frontmatter key, not wiki metadata
            metadata[MetadataField.DESCRIPTION] = frontmatter["description"]  # nocheck: value read from OKF frontmatter
        tags = frontmatter.get("tags")  # nocheck: OKF frontmatter key, not wiki metadata
        if tags:
            metadata[MetadataField.TAGS] = tags if isinstance(tags, list) else [tags]
        source_files = frontmatter.get("source_files")
        if source_files:
            metadata[MetadataField.PROVENANCE] = source_files if isinstance(source_files, list) else [source_files]

        nodes.append(WikiNode(id=slug, title=title, content=body.strip(), node_type=kind, metadata=metadata))
        slug_to_title[slug] = title

    # Edges from body links: ``[title](concepts/x.md)`` and ``[[x]]``.
    title_to_slug = {title: slug for slug, title in slug_to_title.items()}
    for node in nodes:
        body = node.content
        targets: set[str] = set()
        for m in _MD_LINK_RE.finditer(body):
            target_slug = Path(m.group(2)).stem
            targets.add(target_slug)
        for m in _WIKILINK_RE.finditer(body):
            target = m.group(1).strip()
            targets.add(title_to_slug.get(target, target))
        for target in targets:
            if target in slug_to_title and target != node.id:
                edges.append(GraphEdge(source=node.id, target=target, type="reference"))

    return CompiledWiki(nodes=nodes, edges=edges, toc=[], metadata={})


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split markdown into (frontmatter dict, body). Empty dict when absent."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    try:
        fm = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}, text
    return (fm if isinstance(fm, dict) else {}), text[end + 4 :].lstrip("\n")
