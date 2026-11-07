"""OKF v0.2 page metadata — the canonical frontmatter every vault writer emits.

The OKF field vocabulary (``type``/``title``/``description``/``status``/
``generated``/``verified``/``sources``), the trust-event aggregation used by the
quality surfaces, and the writer defaults live here so a page's frontmatter is
built in exactly one place. :class:`~kompend.output.vault.VaultWriter` inherits
the mixin, which keeps the documented ``VaultWriter.okf_frontmatter`` entry
point as the single definition rather than a re-export.
"""

from __future__ import annotations

from typing import Any, Final

from kompend import __version__ as _KOMP_VERSION
from kompend.core.models import MetadataField, WikiNode
from kompend.core.models.enums import NodeType, OkfField
from kompend.core.models.metadata import meta_dict, meta_str_list, meta_str_list_dict
from kompend.core.models.wiki import truncate_description
from kompend.core.paths import safe_stem
from kompend.output.models import OkfSource, TrustEvent

__all__ = ["DEFAULT_STATUS", "DEFAULT_TITLE", "DEFAULT_TYPE", "GENERATED_BY", "KIND_TO_TYPE", "OkfPageMixin"]

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
