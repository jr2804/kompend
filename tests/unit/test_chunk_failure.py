"""A section the splitter cannot split fails its source, loudly (bead knox-sfqg).

Before this, ``HierarchicalChunker._split_text`` logged the splitter's error and
returned the section unsplit, so the pipeline chunked on, embedded an oversized
section, and the ledger said the chunk stage was fine. These tests pin the two
halves of the fix: the splitter failure is a named exception, and both pipeline
call sites turn it into a recorded item failure instead of a silent pass.
"""

# ruff: noqa: S608 — SQL is assembled only from the imported table-name constant.

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kompend.chunking.errors import ChunkingError
from kompend.chunking.hierarchical import HierarchicalChunker
from kompend.core.cache.store import CacheStore
from kompend.core.models import ChunkTree, ElementType, ParsedDocument
from kompend.core.paths import VaultPaths
from kompend.core.pipeline.ingest import PARSE_CACHE_PARSER_KEY
from kompend.core.pipeline.parse_cache import cache_key, write_document
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.pipeline.stages.parse_sources import _process_single_source  # noqa: PLC2701
from kompend.core.pipeline.state import TierState, TierStateStore
from kompend.core.reporting.ledger import stage_ledger
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import STAGES_TABLE
from kompend.core.types import ReasonCode, StageOutcome
from kompend.preprocessing.host_memory import HeavyParseGuard

_IDENTITY = "test-chain"


class _FailingChunker(HierarchicalChunker):
    """A chunker whose splitter is broken, the way a real one breaks."""

    def chunk(self, document: ParsedDocument) -> ChunkTree:  # noqa: ARG002 — fails before reading it
        raise ChunkingError(
            "Cannot split a 2500-character section to 100 tokens with chonkie (cl100k_base): ValueError: tokenizer exploded",
        )


class _StubParser:
    """Parser facade reporting the identity the seeded parse cache carries."""

    def backend_identity(self) -> str:
        return _IDENTITY

    def route(self, source_file: Path) -> None:  # noqa: ARG002 — the cache entry answers first
        return None


class _StubRuntime:
    """Minimal PipelineRuntime covering the parse-cache + chunk path."""

    def __init__(self, base: Path, chunker: HierarchicalChunker) -> None:
        self.base_directory = base
        self.config = SimpleNamespace(
            no_cache=False,
            reparse=False,
            force=False,
            privacy_filter=False,
            output=SimpleNamespace(rumdl_config=None, flavor=None),
            search=SimpleNamespace(contextual=SimpleNamespace(enabled=False)),
        )
        self.parser = _StubParser()
        self.chunker = chunker
        self.enricher = None
        self.state = TierState()
        self.cache_store = CacheStore(base / "cache", base)
        self.state_store = TierStateStore(base / "compile_state.json", base)

    def rel_file(self, file: Path) -> str:
        return str(file)

    def cache_directory(self) -> Path:
        return self.base_directory

    def apply_tags_to_tree(self, tree: ChunkTree, source_file: Path) -> None:  # noqa: ARG002
        return


def test_a_splitter_failure_fails_the_source_instead_of_chunking_it(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    base = tmp_path / "vault"
    runtime, source = _seed(base)

    with stage_ledger(vault_db_path(base)) as ledger, caplog.at_level(logging.WARNING, logger="kompend.core.pipeline.stages.parse_sources"):
        trees = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, _IDENTITY)

    assert trees == []
    chunk_rows = _chunk_rows(base)
    assert len(chunk_rows) == 1
    assert chunk_rows[0]["outcome"] == StageOutcome.FAILED.value
    assert chunk_rows[0]["reason_code"] == ReasonCode.BACKEND_UNAVAILABLE.value
    assert "Cannot split a 2500-character section" in chunk_rows[0]["detail"]
    assert _codes(caplog, "Failed to chunk") == {ReasonCode.BACKEND_UNAVAILABLE}


def test_a_splitter_failure_leaves_a_repro_bundle(tmp_path: Path) -> None:
    """The chunk failure gets the same bundle a parse failure gets."""
    base = tmp_path / "vault"
    runtime, source = _seed(base)

    with stage_ledger(vault_db_path(base)) as ledger:
        _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, _IDENTITY)

    bundles = list(VaultPaths(base).failure_bundles_dir.iterdir())
    assert len(bundles) == 1
    assert (bundles[0] / "meta.json").is_file()


def test_the_off_path_still_chunks_normally(tmp_path: Path) -> None:
    """The guard catches one exception type; a healthy chunker is untouched."""
    base = tmp_path / "vault"
    runtime, source = _seed(base, chunker=HierarchicalChunker(max_tokens=512))

    with stage_ledger(vault_db_path(base)) as ledger:
        trees = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, _IDENTITY)

    assert len(trees) == 1
    assert _chunk_rows(base)[0]["outcome"] == StageOutcome.OK.value


def _codes(caplog: pytest.LogCaptureFixture, needle: str) -> set[ReasonCode]:
    codes: set[ReasonCode] = set()
    for record in caplog.records:
        code = getattr(record, "reason_code", None)
        if needle in record.getMessage() and isinstance(code, ReasonCode):
            codes.add(code)
    return codes


def _chunk_rows(base: Path) -> list[Any]:
    with open_store(vault_db_path(base)) as conn:
        return conn.execute(f"SELECT * FROM {STAGES_TABLE} WHERE stage = ? ORDER BY id", ("chunk",)).fetchall()


def _seed(base: Path, *, chunker: HierarchicalChunker | None = None) -> tuple[Any, Path]:
    """Create the source and a current parse-cache entry of one oversized page."""
    base.mkdir(parents=True, exist_ok=True)
    source = base / "packet.md"
    source.write_text("# packet\n", encoding="utf-8")
    content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {
        "source_file": str(source),
        PARSE_CACHE_PARSER_KEY: _IDENTITY,
        "title": "Packet",
        "author": "",
        "language": "en",
        "page_count": 1,
        "metadata": {},
        "elements": cast(
            "Any",
            [
                {
                    "type": str(ElementType.TEXT),
                    "content": "word " * 500,
                    "page": 1,
                    "reading_order": 1,
                    "heading_level": None,
                    "metadata": {},
                }
            ],
        ),
    }
    write_document(vault_db_path(base), cache_key(source, content_hash), payload)
    return _StubRuntime(base, chunker or _FailingChunker(max_tokens=100, reserve_tokens=0)), source
