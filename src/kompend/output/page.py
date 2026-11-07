"""OKF v0.2 page rendering — frontmatter plus body conversion.

Two halves of "how one page is written" live here:

- :class:`OkfPageMixin` builds the canonical OKF v0.2 frontmatter every vault
  writer emits (the OKF field vocabulary, writer defaults, trust-event
  aggregation).
- The body helpers convert a node body into OKF prose: ``[[wikilinks]]`` become
  portable relative links, inline ``[sources/x.pdf]`` citations become ``[^id]``
  footnote references keyed to the ``sources`` frontmatter entries, and the
  footnote definitions for sources and ``¶ <asset-id>`` references are emitted
  only for ids that are actually referenced.

:class:`~kompend.output.vault.VaultWriter` inherits the mixin, which keeps the
documented ``VaultWriter.okf_frontmatter`` entry point as the single definition
rather than a re-export.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

from kompend import __version__ as _KOMP_VERSION
from kompend.compilation.concept_keys import nearest_alias, normalize_concept_key
from kompend.core.models import MetadataField, Reference, WikiNode
from kompend.core.models.enums import NodeType, OkfField
from kompend.core.models.metadata import meta_dict, meta_list_dict, meta_str_list, meta_str_list_dict
from kompend.core.models.wiki import truncate_description
from kompend.core.paths import safe_stem
from kompend.core.pipeline.assets import AssetStore
from kompend.core.pipeline.references import ReferenceStore
from kompend.output.models import OkfSource, TrustEvent
from kompend.types import MetadataValue

__all__ = [
    "DEFAULT_STATUS",
    "DEFAULT_TITLE",
    "DEFAULT_TYPE",
    "GENERATED_BY",
    "KIND_TO_TYPE",
    "OkfPageMixin",
    "alias_keys",
    "convert_wikilinks",
    "format_asset_footnotes",
    "format_source_footnotes",
    "provenance_sources",
    "reference_to_okf_source",
    "relativize",
    "rewrite_citations_to_footnotes",
    "trust_events",
]

# Regex to find [[wikilinks]] or [[wikilinks|alias]] in body text.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

# Match in-text asset references: ``¶ <id>``.
_ASSET_FOOTNOTE_RE = re.compile(r"¶ ([\w-]+)")

# Match inline source citations ``[sources/<file>.pdf]`` or multi-citations
# ``[sources/a.pdf, sources/b.pdf]``.
_CITATION_RE = re.compile(r"\[sources/[^\]\n]+\]")

# Match inline footnote references ``[^id]``.
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]]+)\]")

# OKF actor convention (§7) for the producer of generated content.
# ``<producer>/<version>`` for agents and tools.
GENERATED_BY: Final[str] = f"kompend/{_KOMP_VERSION or '0.0.0'}"

#: Fallback page title when a node carries no title.
DEFAULT_TITLE: Final[str] = "Untitled"

#: OKF lifecycle default (§5.4) — a page is stable unless stated otherwise.
DEFAULT_STATUS: Final[str] = "stable"

# Map kompend node_type values to OKF ``type`` values. OKF types are free-form
# strings (§4.1); the label is the node type's title case, derived from the
# single :class:`NodeType` vocabulary so a new node type cannot be written
# under one label and read back under another.
KIND_TO_TYPE: Final[dict[str, str]] = {node_type.value: node_type.okf_type for node_type in NodeType}
DEFAULT_TYPE: Final[str] = NodeType.CONCEPT.okf_type


class OkfPageMixin:
    """Canonical OKF v0.2 page metadata shared by every vault writer."""

    @staticmethod
    def okf_type(node_type: NodeType | None) -> str:
        """Map a kompend node_type to an OKF type string."""
        if not node_type:
            return DEFAULT_TYPE
        return KIND_TO_TYPE.get(node_type.value, node_type.value.title() or DEFAULT_TYPE)

    @staticmethod
    def okf_description(node: WikiNode, max_chars: int) -> str:
        """Frontmatter description for *node*, capped at *max_chars* characters.

        The derivation (first sentence of the first prose line) lives on the
        model so the vector store's embed text and the written index cannot
        drift apart; the cap is a rendering decision and belongs here.
        """
        return truncate_description(node.description, max_chars)

    @staticmethod
    def okf_frontmatter(
        node: WikiNode,
        compiled_at: str,
        frontmatter_defaults: dict[str, Any] | None = None,
        *,
        description_max_chars: int,
        sources: list[OkfSource] | None = None,
        verified: list[TrustEvent] | None = None,
    ) -> dict[str, Any]:
        """Build the canonical OKF v0.2 frontmatter for a wiki node.

        The result is OKF §11-conformant (parseable YAML, non-empty ``type``)
        and carries kompend extensions permitted by OKF §4.1. Config
        ``frontmatter_defaults`` and per-node ``metadata['frontmatter']``
        overrides are merged in.

        OKF v0.2 fields emitted:
        - ``type``, ``title``, ``description``, ``tags``, ``status`` (§4.1, §5.4)
        - ``generated: {by, at}`` — supersedes the v0.1 ``timestamp`` (§5.2, §13.1)
        - ``verified`` — trust events, only when ``verified`` is provided (§5.2)
        - ``sources`` — structured provenance objects (§5.1), only when
          ``sources`` is provided; bare filename provenance is NOT emitted
          here under the reserved ``sources`` key (it would collide with the
          v0.2 object shape) — see the ``source_files`` extension below.

        kompend extensions (OKF §4.1 permits additional keys):
        - ``kind`` — original kompend node_type
        - ``source_files`` — provenance filename stems (distinct from OKF
          ``sources`` to avoid shape collision; freed ``tags`` from this overload)
        """
        provenance = meta_str_list(node.metadata.get(MetadataField.PROVENANCE))
        source_files = sorted({safe_stem(p) for p in provenance if p})
        node_tags = meta_str_list(node.metadata.get(MetadataField.TAGS))

        frontmatter: dict[str, Any] = dict(frontmatter_defaults or {})
        # Per-node frontmatter overrides config defaults.
        frontmatter.update(meta_dict(node.metadata.get(MetadataField.FRONTMATTER)))

        # The per-node ``frontmatter`` override is an escape hatch and may
        # carry v0.1 bare-string ``sources`` (filename
        # stems) or ``tags`` overloaded with provenance. Drop both so the only
        # ``sources`` that survives is the structured §5.1 list passed in below,
        # and ``tags`` stays topic-only (provenance stems live in ``source_files``).
        frontmatter.pop(OkfField.SOURCES, None)
        frontmatter.pop(OkfField.COMPILED_AT, None)
        frontmatter.pop(OkfField.TIMESTAMP, None)

        # OKF-required fields (always present, spec-compliant).
        frontmatter[OkfField.TYPE] = frontmatter.get(OkfField.TYPE) or OkfPageMixin.okf_type(node.node_type)
        frontmatter.setdefault(OkfField.TITLE, node.title or DEFAULT_TITLE)
        description = OkfPageMixin.okf_description(node, description_max_chars)
        if description:
            frontmatter.setdefault(OkfField.DESCRIPTION, description)
        # OKF lifecycle (§5.4): default stable; absence ⇒ stable per spec.
        frontmatter.setdefault(OkfField.STATUS, DEFAULT_STATUS)
        # OKF tags (§4.1): cross-cutting topic tags from compilation.
        # Source-filename provenance lives in the ``source_files`` extension (OKF
        # v0.2 semantics), never in ``tags``.
        if node_tags:
            merged_tags = sorted(set(frontmatter.get(OkfField.TAGS, [])) | {str(t) for t in node_tags})
            frontmatter[OkfField.TAGS] = merged_tags
        node_attrs = meta_str_list_dict(node.metadata.get(MetadataField.ATTRIBUTES))
        if node_attrs and OkfField.ATTRIBUTES not in frontmatter:
            serialized: dict[str, str | list[str]] = {}
            for k, v in node_attrs.items():
                serialized[k] = v[0] if len(v) == 1 else v
            frontmatter[OkfField.ATTRIBUTES] = serialized

        # OKF trust (§5.2): ``generated`` records how/when content was produced.
        # Supersedes the v0.1 ``timestamp`` field (§13.1).
        frontmatter[OkfField.GENERATED] = {"by": GENERATED_BY, "at": compiled_at}
        # ``verified`` only when trust events are known (absence ⇒ unverified tier).
        if verified:
            frontmatter[OkfField.VERIFIED] = verified

        # OKF structured provenance (§5.1): list of objects with at least
        # ``resource``. Only emitted when callers supply real structured sources
        # (from the ReferenceStore); never populated with bare filename strings.
        if sources:
            frontmatter.setdefault(OkfField.SOURCES, sources)

        # kompend extensions (OKF §4.1 permits additional keys).
        frontmatter[OkfField.KIND] = node.metadata.get(MetadataField.FRONTMATTER, {}).get(OkfField.KIND, node.node_type.value)
        if source_files:
            frontmatter[OkfField.SOURCE_FILES] = source_files
        # kompend extension (OKF §4.1): the model that generated this page.
        # Survives cache hits, so a mixed-model wiki stays attributable per
        # page; absent for pages that are not LLM-generated (graph compiler).
        model = node.metadata.get(MetadataField.MODEL)
        if model:
            frontmatter[OkfField.GENERATED_BY_MODEL] = str(model)

        return frontmatter

    @staticmethod
    def node_quality(verification: dict[str, Any], node_id: str) -> dict[str, Any]:
        """Aggregate verification claims for *node_id* into a stats dict.

        Returns ``{}`` when no claim was verified for the node.
        """
        claims = [c for c in verification.get("claims", []) if isinstance(c, dict) and c.get("node_id") == node_id]
        if not claims:
            return {}
        supported = sum(1 for c in claims if c.get("verdict") == "supported")
        confidences = [float(c.get("confidence", 0.0)) for c in claims]
        return {
            "claims": len(claims),
            "supported": supported,
            "quality_score": round(sum(confidences) / len(confidences), 2),  # nocheck: OKF frontmatter key, not wiki metadata
        }


def trust_events(value: MetadataValue | None) -> list[TrustEvent] | None:
    """The metadata value as OKF trust events; ``None`` when there are none."""
    events = [TrustEvent(by=event.get("by", ""), at=event.get("at", "")) for event in meta_list_dict(value)]
    return events or None


def relativize(bundle_path: str, folder: str) -> str:
    """Rebase a bundle-relative path onto the current page's folder."""
    return Path(bundle_path).relative_to(folder, walk_up=True).as_posix()


def convert_wikilinks(body: str, slug_targets: dict[str, str]) -> str:
    """Convert ``[[wikilinks]]`` to portable bundle-relative markdown links.

    ``slug_targets`` maps page slugs/titles to bundle-relative paths
    (e.g. ``"my-page" -> "concepts/my-page.md"``). The lookup is
    alias-aware: short forms like ``[[P.863]]`` resolve to a page whose
    long title contains the alias, mirroring the lint rule's behaviour
    so the rendered output stays consistent with what the wiki considers
    resolvable. Unknown targets are left as Obsidian-style wikilinks
    (``[[target]]``) so the user can see broken cross-links in their
    markdown viewer rather than silently losing them.
    """
    # Build a richer lookup index from slug_targets keys. ``slug_targets``
    # is keyed by slug + long title; we need the bare-acronym and
    # normalized forms as well.
    alias_index: dict[str, str] = dict(slug_targets)
    alias_index.update(alias_keys(slug_targets.keys(), slug_targets))

    def _replace(match: re.Match[str]) -> str:
        target = match.group(1)
        alias = match.group(2) or target
        # 1. Direct lookup (slug or long title)
        path = alias_index.get(target)
        # 2. Alias-aware nearest-match (short forms like P.863).
        # nearest_alias returns the value at the best-scoring key — which
        # here is the path itself, so we can use it directly.
        if path is None:
            path = nearest_alias(target, alias_index)
        if path is not None:
            return f"[{alias}]({path})"
        # 3. Unknown — leave as Obsidian wikilink so the user sees it.
        return match.group(0)

    return _WIKILINK_RE.sub(_replace, body)


def alias_keys(titles: Any, paths: dict[str, str]) -> dict[str, str]:
    """Build alias-aware lookup keys for :func:`convert_wikilinks`.

    Mirrors :func:`kompend.lint.structural._node_lookup_index` — acronym,
    bare-prefix-before-paren, and slug forms — so the rendered output
    resolves the same targets the lint rule does. Returns a dict of
    alias-key → path suitable for ``setdefault`` merging.
    """
    out: dict[str, str] = {}
    for key in titles:
        path = paths[key]
        stripped = normalize_concept_key(key)
        if stripped and stripped != key.lower():
            out.setdefault(stripped, path)
        before_paren = key.split("(", 1)[0].strip().lower()
        if before_paren and before_paren != key.lower():
            out.setdefault(before_paren, path)
    return out


def rewrite_citations_to_footnotes(body: str, sources: list[OkfSource]) -> str:
    """Rewrite inline ``[sources/<file>.pdf]`` citations to ``[^id]`` refs.

    The footnote id is resolved from the source entry (by its ``id`` or by the
    stem of its ``resource``), so it works whether ids are file stems or
    structured reference ids. Unmatched citations stay as readable plain text
    (no orphan footnote references). Multi-source citations
    ``[sources/a.pdf, sources/b.pdf]`` become ``[^a][^b]``.
    """
    id_by_stem: dict[str, str] = {}
    for src in sources:
        sid = str(src.get("id") or "")
        if not sid:
            continue
        id_by_stem.setdefault(sid.lower(), sid)
        resource = str(src.get("resource") or "")
        if resource:
            id_by_stem.setdefault(safe_stem(resource).lower(), sid)

    def _replace(match: re.Match[str]) -> str:
        refs: list[str] = []
        for raw in match.group(0)[1:-1].split(","):
            item = raw.strip()
            if not item.startswith("sources/"):
                continue
            stem = Path(item[len("sources/") :]).stem
            sid = id_by_stem.get(stem.lower())
            if sid:
                refs.append(f"[^{sid}]")
        return "".join(refs) if refs else match.group(0)

    return _CITATION_RE.sub(_replace, body)


def provenance_sources(
    node: WikiNode,
    ref_store: ReferenceStore | None,
) -> list[OkfSource]:
    """Build the OKF v0.2 ``sources`` list for a node from its provenance.

    Provenance is a list of source identifiers (filenames/URLs). When a
    :class:`ReferenceStore` is available, each provenance source is expanded
    into one structured ``sources`` entry per extracted reference (§5.1). When
    no structured references exist, the raw provenance identifiers are wrapped
    as minimal ``{id, resource}`` entries so the field stays spec-shaped.
    """
    provenance = meta_str_list(node.metadata.get(MetadataField.PROVENANCE))
    if not provenance:
        return []

    # Deduplicate provenance identifiers, preserving order.
    seen: set[str] = set()
    prov_ids: list[str] = []
    for ps in provenance:
        if ps and ps not in seen:
            seen.add(ps)
            prov_ids.append(ps)

    sources: list[OkfSource] = []
    if ref_store is not None:
        for prov in prov_ids:
            refs = ref_store.get_source(prov) or ref_store.get_source(safe_stem(prov))
            for ref in refs:
                sources.append(reference_to_okf_source(ref))

    # If no structured references resolved, emit minimal spec-shaped entries.
    if not sources:
        for prov in prov_ids:
            sources.append({"id": safe_stem(prov), "resource": prov})
    return sources


def reference_to_okf_source(ref: Reference) -> OkfSource:
    """Build an OKF v0.2 ``sources`` entry (§5.1) from a CSL-aligned Reference.

    The ``id`` is the join key for per-claim footnote attribution (§5.1);
    ``resource`` is REQUIRED within an entry and prefers a URL/DOI when present.
    """
    resource = ref.url or (f"https://doi.org/{ref.doi}" if ref.doi else "")
    if not resource:
        # Fall back to the citing source path so the entry always has a resource.
        resource = ref.citing_source or ref.id
    entry: OkfSource = {"id": ref.id or ref.citing_source or ref.short_label()}
    if resource:
        entry["resource"] = resource
    if ref.title:
        entry["title"] = ref.title
    author = ref.display_authors()
    if author:
        entry["author"] = author
    return entry


def format_source_footnotes(rendered: str, sources: list[OkfSource]) -> str:
    """Render footnote definitions for OKF §5.1 per-claim attribution.

    Only sources whose ``[^id]`` is actually referenced in *rendered* get a
    definition, deduplicated by id and ordered by first reference — no
    orphan definitions (MD066), no orphan references (MD067), no ordering
    warnings.
    """
    ref_positions: dict[str, int] = {}
    for m in _FOOTNOTE_REF_RE.finditer(rendered):
        ref_positions.setdefault(m.group(1), m.start())
    if not ref_positions:
        return ""

    by_id: dict[str, OkfSource] = {}
    for src in sources:
        sid = str(src.get("id") or "")
        if sid and sid not in by_id:
            by_id[sid] = src

    ordered = sorted(
        (s for s in by_id.values() if str(s.get("id")) in ref_positions),
        key=lambda s: ref_positions[str(s["id"])],
    )
    lines: list[str] = []
    for src in ordered:
        sid = str(src["id"])
        label = src.get("title") or src.get("resource") or sid
        lines.append(f"[^{sid}]: {label}")
    return "\n".join(lines) + "\n"


def format_asset_footnotes(body: str, store: AssetStore) -> str:
    """Render asset footnote definitions for ``¶ <id>`` references in *body*.

    Scans *body* for asset references like ``¶ tab-a1f3`` and ``¶ fig-9c2e``,
    collects the unique asset ids, looks up their summaries from *store*,
    and emits ``[^<id>]: <summary>`` lines OKF-style.
    """
    if not body:
        return ""
    ids_found: set[str] = set()
    for m in _ASSET_FOOTNOTE_RE.finditer(body):
        ids_found.add(m.group(1))
    if not ids_found:
        return ""
    lines: list[str] = []
    for aid in sorted(ids_found):
        record = store.get(aid)
        if record is None:
            continue
        label = record.summary or aid
        lines.append(f"[^{aid}]: {label}")
    return "\n".join(lines) + "\n"
