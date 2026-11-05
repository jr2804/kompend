"""Backend-independent vector-store layer: the field contract and the rules
that decide *what* gets indexed.

This module names no vector database. The store that implements
:class:`VectorStoreLike` lives beside it (``sqlite_vec_store.py``); everything
here is shared by every backend — the field names a hit dict carries, the
store protocol, and the composition rules whose fingerprints decide whether an
index is still current.

The split is deliberate: the composition rules are a DOX contract
(``src/kompend/rag/AGENTS.md``) and must not drift with the backend.

**This module stays import-light.** It must not pull pydantic-ai, the
provider SDKs or ``kompend.llm.models``: a module that only needs the
protocol or the composition rules must not pay the provider stack. The
pydantic-ai-backed provider lives in ``embedding_pydantic.py``.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from kompend.core.models import CompiledWiki, MetadataField, WikiNode
from kompend.core.models.elements import element_text
from kompend.core.models.metadata import meta_str_list_dict
from kompend.core.utils.exceptions import LLMOutputError
from kompend.logging import get_logger
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.multimodal import MediaEmbeddingUnsupported

# 3: content is projected through element_text, so a non-prose node embeds
# differently than it did when the composition passed node.content straight
# through. Bumped so existing indexes rebuild instead of serving pre-change vectors.
# 4: a node with figures now feeds its figure to the embedder as well (see
# embed_node), so a vector built from text alone is not the same vector.
EMBED_TEXT_VERSION = 4

#: Cache of figure digests, keyed by (path, size, mtime). The staleness
#: fingerprint hashes every figure on every check, and a query checks it.
_FIGURE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}

#: Upper bound for the digest cache; cleared wholesale when reached.
_FIGURE_DIGEST_CACHE_MAX: int = 4096

logger = get_logger(__name__)


class VectorField(StrEnum):
    """Field names of an indexed node — the single source of truth.

    Every schema field a store declares, every key it writes when indexing,
    and every key read from a search hit must use these members. Hard-coding
    the string literals defeats the schema/reader contract and is forbidden
    (see ``src/kompend/rag/AGENTS.md``).
    """

    NODE_ID = "node_id"
    TITLE = "title"
    NODE_TYPE = "node_type"
    SOURCE_FILE = "source_file"
    CONTENT_PREVIEW = "content_preview"
    TAGS = "tags"
    ATTRIBUTES = "attributes"
    ATTR_KEYS = "attr_keys"
    EMBEDDING = "embedding"


class VectorStoreLike(Protocol):
    """Vector-store surface used by the retrievers.

    A concrete store satisfies this structurally; tests inject a lightweight
    in-memory fake.
    """

    @property
    def available(self) -> bool: ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]: ...


# ── Embedding provider interface ────────────────────────────────────────


def to_pydantic_ai_embedding_string(embedding_model: str) -> str:
    """Translate a kompend embedding-model config value to pydantic-ai format.

    Uses the same ``provider/model`` or ``provider:model`` convention as
    :func:`kompend.llm.models.build_model`. An explicit provider prefix is
    required — there is no silent fallback to a local embedding provider.
    """
    # Known LLM/embedding providers — anything else is rejected.
    _KNOWN_PROVIDERS = frozenset(
        {
            "ollama",
            "openai",
            "google",
            "cohere",
            "bedrock",
            "groq",
            "mistral",
            "deepseek",
        }
    )

    if ":" in embedding_model:
        provider = embedding_model.split(":", 1)[0]
        if provider in _KNOWN_PROVIDERS:
            return embedding_model
    if "/" in embedding_model:
        provider = embedding_model.split("/", 1)[0]
        if provider in _KNOWN_PROVIDERS:
            return provider + ":" + embedding_model.split("/", 1)[1]
    raise ValueError(
        f"Embedding model {embedding_model!r} has no known provider prefix. Use provider/model or provider:model syntax, e.g. ollama:granite-embedding:278m."
    )


# ── What gets indexed ───────────────────────────────────────────────────


def indexable_nodes(wiki: CompiledWiki, *, embed_attachments: bool = False) -> list[WikiNode]:
    """The nodes the vector index covers — the single eligibility rule.

    Every builder and every staleness check must agree on this set: a
    staleness check that compares against *all* node ids reports the index
    stale forever as soon as anything is filtered out.

    Args:
        wiki: The compiled wiki (or the markdown bundle loaded from disk).
        embed_attachments: When ``False`` (default), pages derived from
            attachments (``source_asset=True``) stay out of the index.
    """
    if embed_attachments:
        return list(wiki.nodes)
    return [n for n in wiki.nodes if not (n.metadata or {}).get("source_asset", False)]


def content_hash(node: WikiNode) -> str:
    """Hash of what the index stores for *node* — the staleness fingerprint.

    Covers the embedded text (title, description, content) plus the filterable
    metadata (node type, tags, attributes), because search sees all of it: a
    retag that leaves the prose alone still makes the stored document wrong.

    Also covers the node's figures (their content digest): a figure is part of
    what the embedder sees, so a re-parse that changes one invalidates the
    vector that was built from it. A node without figures keeps the fingerprint
    it always had, so a vault that never had them is not re-indexed for nothing.

    Deliberately excludes the source path: it is relativized against
    ``base_dir`` at index time, so hashing it would report a stale index for a
    vault move that changed no content. Field order is fixed and tags are
    sorted, so the digest does not depend on dict ordering or
    ``PYTHONHASHSEED``.
    """
    metadata = node.metadata or {}
    tags = sorted(str(tag) for tag in metadata.get(MetadataField.TAGS, []))
    attributes = sorted(attributes_to_list(meta_str_list_dict(metadata.get(MetadataField.ATTRIBUTES))))
    payload = "\n".join(
        [
            embed_text(node),
            node.node_type.value,
            " ".join(tags),
            " ".join(attributes),
        ]
    )
    figures = figure_identity(node)
    if figures:
        payload = f"{payload}\nfigures:{figures}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def attributes_to_list(attrs: dict[str, list[str]]) -> list[str]:
    """Flatten an attribute dict to filterable ``key=value`` tokens.

    Stored one token per value so a store can answer "has key=value" and
    "has key" without a wildcard scan.
    """
    return [f"{key}={value}" for key, values in attrs.items() for value in values]


def embed_text(node: WikiNode) -> str:
    """Compose the text embedded for *node*.

    Includes :attr:`WikiNode.description` — the page's first prose sentence,
    derived from its own content and already used for the written index
    frontmatter. Leaving it out made the vector path weigh different evidence
    than the written summary. A node with no prose (empty content, or headings
    only) embeds as ``title`` plus a newline plus ``content``, as before.

    Content is passed through :func:`element_text`, the canonical projection
    for the ``str | list[list[str]] | bytes`` union that element content is
    declared as. A node whose content is not prose — a table kept structured, an
    asset payload — reaches the embedder as renderable text instead of whatever
    ``str()`` would produce for it. Node content is typed ``str``, so this is
    the tier that guarantees the contract rather than assuming it upstream.
    """
    description = node.description
    body = element_text(node.content)
    if description:
        return f"{node.title}\n{description}\n{body}"
    return f"{node.title}\n{body}"


def embed_node(
    provider: EmbeddingProvider,
    node: WikiNode,
    text: str,
    *,
    base_dir: Path | None = None,
) -> tuple[list[float], str | None]:
    """Embed *node*, feeding its figure to the embedder when it can take one.

    Returns the vector and a note when a figure could not be embedded. The
    node is still indexed from its text — the note says which figure was left
    out, so the degradation is reported instead of being silent.

    Only the first figure is embedded: the endpoint takes one image per
    request. A node with several figures says so in the note rather than
    quietly embedding one of them.
    """
    paths = figure_paths(node)
    if not paths:
        return provider.embed(text), None
    notes: list[str] = []
    if len(paths) > 1:
        notes.append(f"{len(paths)} figures; only the first was embedded")
    data = figure_bytes(paths[0], base_dir=base_dir)
    if data is None:
        notes.append(f"figure file is missing: {paths[0]}")
        return provider.embed(text), "; ".join(notes)
    try:
        vector = provider.embed_media(text, data)
    except MediaEmbeddingUnsupported as exc:
        notes.append(str(exc))
        return provider.embed(text), "; ".join(notes)
    except LLMOutputError as exc:
        notes.append(f"media embedding failed: {exc}")
        return provider.embed(text), "; ".join(notes)
    return vector, "; ".join(notes) if notes else None


def figure_paths(node: WikiNode) -> list[str]:
    """The figure files a node carries, in page order.

    The figures live in ``node.metadata["images"]`` as ``ImageRef`` mappings
    (``path``, ``caption``, ``element_type``, ``source_file``) written by the
    compilation flow.
    """
    raw = (node.metadata or {}).get(MetadataField.IMAGES)
    if not isinstance(raw, list):
        return []
    paths: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            path = item.get("path")
            if isinstance(path, str) and path:
                paths.append(path)
    return paths


def figure_bytes(path: str, *, base_dir: Path | None = None) -> bytes | None:
    """Read a figure file, or ``None`` when it cannot be read."""
    candidate = Path(path)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    try:
        return candidate.read_bytes()
    except OSError:
        return None


def figure_identity(node: WikiNode) -> str:
    """Content digest of a node's figures — part of the staleness fingerprint.

    Only the bytes count, not the path: a vault move rewrites the path but
    not the image, and :func:`content_hash` deliberately excludes paths for
    that reason. A figure that cannot be read counts as ``missing``, so the
    index is stale again once it reappears.
    """
    return "|".join(_figure_digest(path) for path in figure_paths(node))


def _figure_digest(path: str) -> str:
    """Digest of a figure file, cached by (path, size, mtime)."""
    try:
        stat = os.stat(path)
    except OSError:
        return "missing"
    key = (path, stat.st_size, stat.st_mtime_ns)
    cached = _FIGURE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    data = figure_bytes(path)
    if data is None:
        return "missing"
    digest = hashlib.sha256(data).hexdigest()[:16]
    if len(_FIGURE_DIGEST_CACHE) >= _FIGURE_DIGEST_CACHE_MAX:
        _FIGURE_DIGEST_CACHE.clear()
    _FIGURE_DIGEST_CACHE[key] = digest
    return digest
