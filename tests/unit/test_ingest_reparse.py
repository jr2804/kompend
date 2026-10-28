"""Regression: forced re-ingest must preserve user metadata.

Covers the ``kompend ingest --no-cache`` path:

- :func:`preserve_cached_metadata` — tags unioned, attributes merged.
- :func:`ingest_documents` — same content stays a cache hit; ``--no-cache`` forces re-parse
  while keeping tags and attributes.

Attachments are intentionally untouched: they live in
``.kompend/attachments.jsonl`` + ``attachments/<slug>/`` and ``ingest`` never
writes them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from kompend.chunking.hierarchical import HierarchicalChunker
from kompend.core.cache.store import CacheStore
from kompend.core.models import ChunkTree, DocumentElement, ElementType, MetadataField, ParsedDocument
from kompend.core.pipeline.ingest import (
    PARSE_CACHE_PARSER_KEY,
    ingest_documents,
    preserve_cached_metadata,
)
from kompend.core.pipeline.parse_cache import cache_key, iter_documents, write_document
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.pipeline.stages._parse import _chunk_cache_deps, _process_single_source  # noqa: PLC2701
from kompend.core.pipeline.state import TierState, TierStateStore
from kompend.core.store import vault_db_path
from kompend.preprocessing.parsers.base import BaseParser, ParserRouter
from tests.conftest import as_obj

# ── Router fingerprint ─────────────────────────────────────────────────


class _StubParser(BaseParser):
    """Minimal parser so a real :class:`ParserRouter` can be built."""

    def can_parse(self, source_file: Path) -> bool:  # noqa: PLR6301
        return False

    def parse(self, source_file: Path) -> ParsedDocument:  # noqa: ARG002
        return ParsedDocument(source_file=source_file, elements=[], title="")


# ── ingest_documents: parser-change re-parse + preservation ────────────


class _RecordingParser(BaseParser):
    """Parser that records each parse call and returns a fixed document."""

    def __init__(self, parsed_log: list[Path]) -> None:
        self._parsed_log = parsed_log

    def can_parse(self, source_file: Path) -> bool:
        return True

    def parse(self, source_file: Path) -> ParsedDocument:
        self._parsed_log.append(source_file)
        element = DocumentElement(type=ElementType.TEXT, content="body")
        return ParsedDocument(
            source_file=Path("sources/note.md"),
            elements=[element],
            title="Note",
        )


class _IngestRouter(ParserRouter):
    """Router double that reports a settable fingerprint and records parses."""

    def __init__(self, fingerprint: str) -> None:
        super().__init__()
        self._fingerprint_value = fingerprint
        self.parsed: list[Path] = []

    def fingerprint(self) -> str:
        return self._fingerprint_value

    def route(self, source_file: Path) -> BaseParser | None:
        return _RecordingParser(self.parsed)


class _StubEnricher:
    """Sets a contextual paragraph on every leaf, like the real agent path."""

    def enrich_tree(self, tree: ChunkTree, doc: ParsedDocument) -> ChunkTree:  # noqa: ARG002
        for leaf in tree.leaves():
            leaf.contextual_context = f"context for {leaf.id}"
        return tree


class _StubRuntime:
    """Minimal PipelineRuntime for the chunk+enrich+cache path."""

    def __init__(self, base: Path, *, enricher: object | None) -> None:
        self.base_directory = base
        self.config = SimpleNamespace(
            no_cache=False,
            force=False,
            privacy_filter=False,
            search=SimpleNamespace(contextual=SimpleNamespace(enabled=enricher is not None)),
        )
        self.router = None
        self.chunker = HierarchicalChunker(max_tokens=512)
        self.enricher = enricher
        self.state = TierState()
        self.cache_store = CacheStore(base / ".kompend" / "cache", base)
        self.state_store = TierStateStore(base / ".kompend" / "compile_state.json", base)

    def rel_file(self, file: Path) -> str:
        return str(file)

    def abs_file(self, file_str: str) -> Path:
        p = Path(file_str)
        return p if p.is_absolute() else self.base_directory / p

    def apply_tags_to_tree(self, tree: ChunkTree, source_file: Path) -> None:  # noqa: ARG002
        return


def test_fingerprint_is_stable() -> None:
    assert _router_of("Alpha").fingerprint() == _router_of("Alpha").fingerprint()


def test_fingerprint_changes_with_the_registered_parser() -> None:
    assert _router_of("Alpha").fingerprint() != _router_of("Beta").fingerprint()
    assert _router_of("Alpha").fingerprint() != _router_of("Alpha", "Beta").fingerprint()


# ── Cache-currency checks ──────────────────────────────────────────────


# ── Metadata preservation ──────────────────────────────────────────────


def test_preserve_metadata_unions_tags_and_merges_attributes() -> None:
    prior: dict[str, Any] = {
        "metadata": {
            MetadataField.TAGS: ["audio", "standards"],
            MetadataField.ATTRIBUTES: {"version": "2", "status": "old"},
        }
    }
    doc = ParsedDocument(source_file=Path("sources/note.md"), elements=[], title="")
    doc.metadata = {
        MetadataField.TAGS: ["standards", "new"],
        MetadataField.ATTRIBUTES: {"status": "draft"},
    }

    preserve_cached_metadata(doc, prior)

    assert doc.metadata[MetadataField.TAGS] == ["standards", "new", "audio"]
    # Current value ("draft") wins over the prior ("old"); "version" is kept.
    assert doc.metadata[MetadataField.ATTRIBUTES] == {"version": "2", "status": "draft"}


def test_preserve_metadata_is_noop_without_prior_entry() -> None:
    doc = ParsedDocument(source_file=Path("sources/note.md"), elements=[], title="")
    doc.metadata = {MetadataField.TAGS: ["keep"]}
    preserve_cached_metadata(doc, None)
    assert doc.metadata[MetadataField.TAGS] == ["keep"]


def test_preserve_metadata_ignores_missing_prior_entry() -> None:
    doc = ParsedDocument(source_file=Path("sources/note.md"), elements=[], title="")
    preserve_cached_metadata(doc, None)
    assert doc.metadata is None or MetadataField.TAGS not in doc.metadata


def test_same_content_stays_a_cache_hit(tmp_path: Path) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"

    first = _IngestRouter("chain-A")
    ingest_documents(files=[src], base_dir=base, router=first, config=None)
    assert len(first.parsed) == 1

    second = _IngestRouter("chain-A")
    result = ingest_documents(files=[src], base_dir=base, router=second, config=None)
    assert second.parsed == []
    assert result.cached == 1


def test_forced_reingest_preserves_and_merges_cli_metadata(tmp_path: Path) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"

    ingest_documents(files=[src], base_dir=base, router=_IngestRouter("chain-A"), config=None)

    key, entry = _cache_entry_with_key(base)
    entry["metadata"][MetadataField.TAGS] = ["audio"]
    entry["metadata"][MetadataField.ATTRIBUTES] = {"version": "2", "status": "old"}
    _write_entry(base, key, entry)

    forced = _IngestRouter("chain-A")
    result = ingest_documents(
        files=[src],
        base_dir=base,
        router=forced,
        no_cache=True,
        config=None,
        tags=["standards"],
        attributes={"status": "draft"},
    )

    assert len(forced.parsed) == 1
    assert result.cached == 0
    fresh = _cache_entry(base)
    assert fresh["metadata"][MetadataField.TAGS] == ["audio", "standards"]
    assert fresh["metadata"][MetadataField.ATTRIBUTES] == {"version": "2", "status": "draft"}


def _cache_entry(base: Path) -> dict[str, Any]:
    return _cache_entry_with_key(base)[1]


def _cache_entry_with_key(base: Path) -> tuple[str, dict[str, Any]]:
    db_path = vault_db_path(base / ".kompend" / "cache")
    records = list(iter_documents(db_path))
    assert records, "expected a parse-cache entry"
    return records[0]


def _write_entry(base: Path, key: str, entry: dict[str, Any]) -> None:
    write_document(vault_db_path(base / ".kompend" / "cache"), key, entry)


def test_attachments_untouched_by_reingest(tmp_path: Path) -> None:
    """Re-ingest must not modify the attachment manifest."""
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    manifest = base / ".kompend" / "attachments.jsonl"
    manifest.write_text('{"filename": "diagram.png"}\n', encoding="utf-8")
    before = manifest.read_text(encoding="utf-8")

    ingest_documents(files=[src], base_dir=base, router=_IngestRouter("chain-A"), no_cache=True, config=None)

    assert manifest.read_text(encoding="utf-8") == before


def test_chunk_cache_deps_gate_off_matches_default() -> None:
    """Default (gate off) chunk-cache identity is byte-identical to today."""
    deps = _chunk_cache_deps(_runtime_for_chunk_deps(contextual=False, enricher=None), "abc")
    assert deps["contextual"] == "0"
    assert "parser" not in deps


def test_chunk_cache_deps_gate_on_invalidates() -> None:
    """Enabling the contextual gate changes the chunk-cache key."""
    off = _chunk_cache_deps(_runtime_for_chunk_deps(contextual=False, enricher=None), "abc")
    on = _chunk_cache_deps(_runtime_for_chunk_deps(contextual=True, enricher=object()), "abc")
    assert off != on
    assert on["contextual"] == "1"


def _runtime_for_chunk_deps(*, contextual: bool, enricher: object | None) -> Any:
    """Minimal stand-in exposing only what ``_chunk_cache_deps`` reads."""
    config = SimpleNamespace(search=SimpleNamespace(contextual=SimpleNamespace(enabled=contextual)))
    return SimpleNamespace(
        chunker=SimpleNamespace(_effective_max=512),
        router=_router_of("chain-A"),
        config=config,
        enricher=enricher,
    )


def _router_of(*parser_names: str) -> ParserRouter:
    router = ParserRouter()
    for name in parser_names:
        router.register(type(name, (_StubParser,), {})())
    return router


def test_enriched_tree_is_what_gets_cached(tmp_path: Path) -> None:
    """TASK-20: enrichment runs after chunking and BEFORE the chunk-cache write,
    so the cached tree (and therefore the embeddings) carry the context.
    """
    base = _seed_vault(tmp_path)
    source_file = Path("sources/note.md")
    content_hash = hashlib.sha256((base / "sources" / "note.md").read_bytes()).hexdigest()
    _seed_parse_cache(base, source_file, content_hash)

    runtime = _StubRuntime(base, enricher=_StubEnricher())
    tree = _process_single_source(cast("PipelineRuntime", runtime), base / "sources" / "note.md", {})

    assert tree is not None
    cached = runtime.cache_store.get_dict("chunk", _chunk_cache_deps(cast("PipelineRuntime", runtime), content_hash))
    assert cached is not None
    nodes = as_obj(as_obj(cached)["nodes"])
    contexts = [node["contextual_context"] for node in nodes.values()]
    assert any(contexts), "cached tree must carry contextual_context"


def test_gate_off_cached_tree_has_no_context(tmp_path: Path) -> None:
    """Default-transparent: with the gate off, cached nodes carry no context
    (byte-identical to today's output).
    """
    base = _seed_vault(tmp_path)
    source_file = Path("sources/note.md")
    content_hash = hashlib.sha256((base / "sources" / "note.md").read_bytes()).hexdigest()
    _seed_parse_cache(base, source_file, content_hash)

    runtime = _StubRuntime(base, enricher=None)
    _process_single_source(cast("PipelineRuntime", runtime), base / "sources" / "note.md", {})

    cached = runtime.cache_store.get_dict("chunk", _chunk_cache_deps(cast("PipelineRuntime", runtime), content_hash))
    assert cached is not None
    nodes = as_obj(as_obj(cached)["nodes"])
    assert all(node["contextual_context"] == "" for node in nodes.values())


def _seed_vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir(parents=True)
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    (base / "sources" / "note.md").write_text("# Note\n", encoding="utf-8")
    return base


def _seed_parse_cache(base: Path, source_file: Path, content_hash: str) -> None:
    data: dict[str, Any] = {
        str(MetadataField.SOURCE_FILE): str(source_file),
        PARSE_CACHE_PARSER_KEY: "",
        "title": "Note",
        "author": "",
        "language": "en",
        "page_count": 1,
        "metadata": {},
        "elements": [
            {
                "type": str(ElementType.TEXT),
                "content": "body text",
                "page": None,
                str(MetadataField.READING_ORDER): 0,
                "heading_level": None,
                "metadata": {},
            }
        ],
    }
    write_document(vault_db_path(base), cache_key(source_file, content_hash), data)
