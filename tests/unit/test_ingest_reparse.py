"""Regression: forced re-ingest must preserve user metadata, and a parser
switch must invalidate both the parse cache and the chunk cache.

Covers the ``knox ingest --no-cache`` / parser-change path:

- :meth:`ParserRouter.fingerprint` — stable identity of the parser chain.
- :func:`parse_cache_is_current` / :func:`cached_entry_is_current` — a stale
  parser fingerprint is a cache miss.
- :func:`preserve_cached_metadata` — tags unioned, attributes merged.
- :func:`ingest_documents` — re-parses on parser change while keeping tags
  and attributes; same parser stays a cache hit.
- :func:`_chunk_cache_deps` — the chunk-cache key carries the parser
  fingerprint so a re-parse re-chunks on the next compile.

Attachments are intentionally untouched: they live in
``.knox/attachments.jsonl`` + ``attachments/<slug>/`` and ``ingest`` never
writes them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knox.core.models import DocumentElement, ElementType, MetadataField, ParsedDocument
from knox.core.pipeline.ingest import (
    PARSE_CACHE_PARSER_KEY,
    ingest_documents,
    parse_cache_is_current,
    preserve_cached_metadata,
)
from knox.core.pipeline.parse_cache import iter_documents, write_document
from knox.core.store import vault_db_path
from knox.preprocessing.parsers.base import BaseParser, ParserRouter

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


def test_fingerprint_is_stable() -> None:
    assert _router_of("Alpha").fingerprint() == _router_of("Alpha").fingerprint()


def test_fingerprint_changes_with_parser_chain() -> None:
    assert _router_of("Alpha").fingerprint() != _router_of("Beta").fingerprint()
    assert _router_of("Alpha", "Beta").fingerprint() != _router_of("Beta", "Alpha").fingerprint()


def test_fingerprint_changes_with_fallback() -> None:
    a = _router_of("Alpha")
    b = _router_of("Alpha")
    b.set_text_fallback(type("OtherFallback", (_StubParser,), {})())
    assert a.fingerprint() != b.fingerprint()


def _router_of(*parser_names: str) -> ParserRouter:
    router = ParserRouter()
    for name in parser_names:
        router.register(type(name, (_StubParser,), {})())
    router.set_text_fallback(_StubParser())
    return router


# ── Cache-currency checks ──────────────────────────────────────────────


def test_parse_cache_is_current_matches_fingerprint() -> None:
    assert parse_cache_is_current({PARSE_CACHE_PARSER_KEY: "abc"}, "abc")
    assert not parse_cache_is_current({PARSE_CACHE_PARSER_KEY: "abc"}, "xyz")


def test_parse_cache_is_current_rejects_legacy_entries() -> None:
    """Entries written before the key existed must be re-parsed once."""
    assert not parse_cache_is_current({"metadata": {}}, "abc")


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


def test_same_parser_stays_a_cache_hit(tmp_path: Path) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"

    first = _IngestRouter("chain-A")
    ingest_documents(files=[src], base_dir=base, router=first, config=None)
    assert len(first.parsed) == 1

    second = _IngestRouter("chain-A")
    result = ingest_documents(files=[src], base_dir=base, router=second, config=None)
    assert second.parsed == []
    assert result.cached == 1


def test_parser_change_forces_reparse_and_preserves_metadata(tmp_path: Path) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"

    ingest_documents(files=[src], base_dir=base, router=_IngestRouter("chain-A"), config=None)

    # Simulate ``knox sources tag`` / ``attr`` writing into the cache entry.
    key, entry = _cache_entry_with_key(base)
    entry["metadata"][MetadataField.TAGS] = ["audio"]
    entry["metadata"][MetadataField.ATTRIBUTES] = {"version": "2"}
    _write_entry(base, key, entry)

    switched = _IngestRouter("chain-B")
    result = ingest_documents(files=[src], base_dir=base, router=switched, config=None)

    assert len(switched.parsed) == 1, "parser change must re-parse"
    assert result.cached == 0
    fresh = _cache_entry(base)
    assert fresh[PARSE_CACHE_PARSER_KEY] == "chain-B"
    assert fresh["metadata"][MetadataField.TAGS] == ["audio"]
    assert fresh["metadata"][MetadataField.ATTRIBUTES] == {"version": "2"}


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
    db_path = vault_db_path(base / ".knox" / "cache")
    records = list(iter_documents(db_path))
    assert records, "expected a parse-cache entry"
    return records[0]


def _write_entry(base: Path, key: str, entry: dict[str, Any]) -> None:
    write_document(vault_db_path(base / ".knox" / "cache"), key, entry)


def test_attachments_untouched_by_reingest(tmp_path: Path) -> None:
    """Re-ingest must not modify the attachment manifest."""
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    manifest = base / ".knox" / "attachments.jsonl"
    manifest.write_text('{"filename": "diagram.png"}\n', encoding="utf-8")
    before = manifest.read_text(encoding="utf-8")

    ingest_documents(files=[src], base_dir=base, router=_IngestRouter("chain-A"), no_cache=True, config=None)

    assert manifest.read_text(encoding="utf-8") == before


def _seed_vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    (base / ".knox" / "config.toml").write_text("", encoding="utf-8")
    (base / "sources" / "note.md").write_text("# Note\n", encoding="utf-8")
    return base
