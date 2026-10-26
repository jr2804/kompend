"""StrEnum contracts and element-type definitions for kompend data models.

Every dict key, discriminator, or schema column name that crosses a file
boundary lives here as a ``StrEnum`` member — never as a bare literal.
"""

from __future__ import annotations

from enum import StrEnum

from aenum import MultiValueEnum

__all__ = [
    "CacheDeps",
    "ChunkField",
    "ChunkNodeField",
    "ChunkStrategy",
    "ChunkTreeField",
    "EdgeType",
    "ElementType",
    "ExtractionField",
    "MetadataField",
    "NodeType",
    "OkfField",
    "StructuredField",
    "WikiField",
    "node_type_from_okf_type",
]
#: Chars of the asset kind value used as its short id prefix when a kind
#: declares no explicit alias (``tab-<hash>``, ``fig-<hash>``).
_PREFIX_LEN: int = 3


class AssetField(StrEnum):
    """Dict key contract for ``AssetRecord`` JSON serialization.

    Asset records store separated tables/figures (and, when enabled later,
    equations) under content-addressed ids so extraction input can reference
    them by id + summary instead of inlining full content.
    """

    ID = "id"
    KIND = "kind"
    SOURCE = "source"
    DATA = "data"
    SUMMARY = "summary"
    REF_COUNT = "ref_count"


class AssetKind(StrEnum, MultiValueEnum):
    """Kinds of separated structured assets.

    Each member takes ``(canonical, prefix)`` where ``canonical`` is the
    serialization value (``table``/``figure``/``equation``) and ``prefix``
    is the short id prefix used in asset ids (``tab-<hash>``, ``fig-<hash>``).
    The canonical value is also accepted as an alias for lookups, so
    ``AssetKind("table") == AssetKind.TABLE`` works.
    Equations are deferred (Phase 1 leaves them inlined).
    """

    def __init__(self, value: str, *aliases: str) -> None:
        self.prefix = aliases[0] if aliases else value[:_PREFIX_LEN]

    TABLE = "table", "tab"
    FIGURE = "figure", "fig"
    EQUATION = "equation", "eq"


class CacheDeps(StrEnum):
    """Dict key contract for cache store dependency hashes.

    Written by ``extract_one``/``extract_incremental``/``gen_cache_deps``
    and validated by :meth:`CacheStore.get` on every lookup. A typo here
    silently breaks caching — the write stores under one key, the read
    never finds it.
    """

    SOURCE_TEXT = "source_text"
    CHUNK_SIZE_MIN = "chunk_size_min"
    CHUNK_SIZE_MAX = "chunk_size_max"
    EXTRACTIONS = "extractions"
    MODEL = "model"
    PINNED_FACTS = "pinned_facts"
    STAGE = "stage"


class ChunkField(StrEnum):
    """Dict key contract for ``Chunk`` JSON serialization."""

    ID = "id"
    TEXT = "text"
    ELEMENT_TYPES = "element_types"
    SOURCE_ELEMENTS = "source_elements"
    SOURCE_FILE = "source_file"
    HEADING_CONTEXT = "heading_context"
    TOKEN_COUNT = "token_count"  # noqa: S105
    READING_ORDER = "reading_order"
    METADATA = "metadata"


class ChunkNodeField(StrEnum):
    """Dict key contract for ``ChunkNode`` JSON serialization."""

    ID = "id"
    PARENT_ID = "parent_id"
    CHILDREN_IDS = "children_ids"
    DEPTH = "depth"
    HEADING_CHAIN = "heading_chain"
    CHUNK = "chunk"
    STRUCTURED = "structured"
    SUMMARY = "summary"
    TOKEN_COUNT = "token_count"  # noqa: S105
    CONTEXTUAL_CONTEXT = "contextual_context"


class ChunkStrategy(StrEnum):
    """How a document was chunked — determines downstream processing depth.

    - ``SINGLE_PASS``: document fits in context window; trivial tree (root +
      leaves). No summaries needed.
    - ``TWO_LEVEL``: document is <= 8× context window; root + section nodes +
      leaves. Root-level summary generated for navigation.
    - ``MULTI_RESOLUTION``: document is >> 8× context window; full-depth tree
      with summaries at every internal node for multi-resolution navigation.
    """

    SINGLE_PASS = "single_pass"  # noqa: S105
    TWO_LEVEL = "two_level"
    MULTI_RESOLUTION = "multi_resolution"


class ChunkTreeField(StrEnum):
    """Dict key contract for ``ChunkTree`` JSON serialization."""

    VERSION = "version"
    ROOT_ID = "root_id"
    NODES = "nodes"
    SOURCE_FILE = "source_file"
    STRATEGY = "strategy"


class EdgeType(StrEnum):
    """Wiki graph edge types — the ``GraphEdge.type`` vocabulary.

    Every producer of a graph edge (the graph compiler, the assembly flow, the
    VLM enricher, the markdown loader) uses these values, so the edge types
    have one home instead of a literal per call site.
    """

    ENTITY_REFERENCE = "entity_reference"
    CONCEPT_REFERENCE = "concept_reference"
    TITLE_REFERENCE = "title_reference"
    RELATED = "related"
    SIBLING = "sibling"
    REFERENCE = "reference"


class ElementType(StrEnum):
    """Document element types for structure-preserving extraction.

    StrEnum serializes to plain strings in JSON: ``"text"``, ``"table"``, etc.
    """

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    CODE = "code"
    METADATA = "metadata"

    @property
    def is_atomic(self) -> bool:
        """Whether this element type must never be split during chunking."""
        return self.value in {"table", "image", "formula", "code"}


class ExtractionField(StrEnum):
    """Dict key contract for ``Extraction`` serialization and LLM output.

    Produced by :meth:`Extraction.to_dict` and by pydantic model
    ``model_dump()``; consumed by ``combine_extractions``, ``concept_pages``,
    and ``gen_cache_deps``. These keys cross the extraction → cache →
    assembly/generation boundary.
    """

    SUMMARY = "summary"
    ENTITIES = "entities"
    CONCEPTS = "concepts"
    FINDINGS = "findings"
    NAME = "name"
    TYPE = "type"
    CONTEXT = "context"
    DEFINITION = "definition"
    IMPORTANCE = "importance"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    SLUG = "slug"


class MetadataField(StrEnum):
    """Canonical keys for ``WikiNode.metadata`` and provenance data dicts.

    Every metadata / source-file / provenance key that crosses file boundaries
    must live here, not as a bare literal.
    """

    SOURCE = "source"
    SOURCE_FILE = "source_file"
    SOURCE_COUNT = "source_count"
    SOURCE_NAMES = "source_names"
    HEADING_CONTEXT = "heading_context"
    READING_ORDER = "reading_order"
    PROVENANCE = "provenance"
    IMAGE_FILE = "image_file"
    IMAGE_BASE64 = "image_base64"
    TAGS = "tags"
    ATTRIBUTES = "attributes"
    DESCRIPTION = "description"
    MODEL = "model"
    EMBEDDING_MODEL = "embedding_model"
    KINDS = "kinds"
    NODE_COUNT = "node_count"
    EDGE_COUNT = "edge_count"
    TWO_PHASE = "two_phase"
    GENERATION_SKIPPED = "generation_skipped"
    CONCEPTS_AVAILABLE = "concepts_available"
    SKIP_REASON = "skip_reason"
    CONCEPTS_MERGED = "concepts_merged"
    CONTEXT_WINDOW = "context_window"
    PINNED_FACTS_COUNT = "pinned_facts_count"
    RELATED = "related"
    ENTITY_TYPE = "entity_type"
    WIKILINK_ALIASES = "wikilink_aliases"
    FRONTMATTER = "frontmatter"


class NodeType(StrEnum):
    """Canonical wiki node types — the ``WikiNode.node_type`` vocabulary.

    Every producer of a node type (the graph compiler, the capture writer,
    the OKF frontmatter writer) and every consumer (folder routing, the
    index catalog) uses these values, so the vocabulary has one home.

    The members below are the *built-in* kinds. A vault may add extra page
    kinds through ``compile.extraction.categories``; those are not members
    but are carried as extra pseudo-members of the same type (see
    :meth:`of`), so a node type is always a ``NodeType`` and never a bare
    string.
    """

    CONCEPT = "concept"
    COMPARISON = "comparison"
    ENTITY = "entity"
    SUMMARY = "summary"
    OVERVIEW = "overview"
    PROCEDURE = "procedure"
    SECTION = "section"
    PAGE = "page"
    SOURCE = "source"
    QUALITY = "quality"
    REFERENCE = "reference"

    @property
    def okf_type(self) -> str:
        """The OKF ``type`` label for this node type (e.g. ``Concept``)."""
        return self.value.title()

    @classmethod
    def of(cls, value: str | NodeType) -> NodeType:
        """Map a kind string onto the vocabulary.

        A built-in kind maps to its member. Any other non-empty kind — the
        extra categories a vault configures through
        ``compile.extraction.categories`` — maps to an extra pseudo-member of
        the same type, so the field stays one type end to end. An empty kind
        is a bug and raises.
        """
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("A node type must not be empty")
        member = cls._value2member_map_.get(normalized)
        if isinstance(member, cls):
            return member
        extra = str.__new__(cls, normalized)
        extra._name_ = normalized.title()
        extra._value_ = normalized
        # Cache it so the same extra kind is always the same object.
        cls._value2member_map_[normalized] = extra
        return extra


class OkfField(StrEnum):
    """OKF v0.2 frontmatter keys — the YAML contract every writer emits.

    The vault writers, the capture writers and the markdown reader all
    speak these keys; a bare ``"description"`` literal in one of them
    would be a second, silently divergent contract.
    """

    OKF_VERSION = "okf_version"
    TYPE = "type"
    TITLE = "title"
    DESCRIPTION = "description"
    STATUS = "status"
    TAGS = "tags"
    ATTRIBUTES = "attributes"
    GENERATED = "generated"
    VERIFIED = "verified"
    SOURCES = "sources"
    KIND = "kind"
    SOURCE_FILES = "source_files"
    GENERATED_BY_MODEL = "model"
    COMPILED_AT = "compiled_at"
    TIMESTAMP = "timestamp"


class StructuredField(StrEnum):
    """Dict key contract for ``StructuredData`` JSON serialization.

    Shared discriminator key plus per-variant keys. Each variant reuses
    ``_TYPE`` for its ``_type`` discriminator and inherits common fields
    like ``caption``.
    """

    _TYPE = "_type"
    TYPE_TABLE = "table"
    TYPE_EQUATION = "equation"
    TYPE_FIGURE = "figure"
    COLUMNS = "columns"
    ROWS = "rows"
    CAPTION = "caption"
    LATEX = "latex"
    VARIABLES = "variables"
    NAME = "name"
    DEFINITION = "definition"
    UNITS = "units"
    ALT_TEXT = "alt_text"
    IMAGE_REF = "image_ref"
    DESCRIPTION = "description"


class WikiField(StrEnum):
    """Dict key contract for ``CompiledWiki`` JSON serialization.

    Keys written by :meth:`CompiledWiki.to_dict` (or built by pipeline) and
    read by :func:`CompiledWiki.from_dict` and ``web/server.py``.
    """

    NODES = "nodes"
    EDGES = "edges"
    TOC = "toc"
    QUALITY_SCORE = "quality_score"
    ITERATION = "iteration"
    CHUNK_COUNT = "chunk_count"
    DURATION_SECONDS = "duration_seconds"
    LLM_USAGE = "llm_usage"
    LLM_MODEL = "llm_model"
    LLM_DEGRADATION = "llm_degradation"
    TIER_TIMINGS = "tier_timings"
    METADATA = "metadata"


def node_type_from_okf_type(okf_type: str) -> NodeType:
    """Map an OKF ``type`` label back to its :class:`NodeType`.

    Unknown labels fall back to :attr:`NodeType.CONCEPT` — a page with an
    unrecognized type still routes as a concept page.
    """
    try:
        return NodeType(okf_type.strip().lower())
    except ValueError:
        return NodeType.CONCEPT
