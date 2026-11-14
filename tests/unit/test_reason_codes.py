"""Pin ReasonCode emission sites on the item path (knox-ty8f.1).

Every item-path ``logger.error``/``logger.warning`` carries a
``reason_code`` via logger ``extra=...``. These tests pin one emission site
per reachable member so a refactor cannot silently drop the code again.
``JUDGE_DEGRADED`` and ``DECISION_FALLBACK`` have no emission site in this
wave (their sites live in the parser package and the decision tier, which
other waves own); they are pinned here as enum members only and get their
site pins when those waves land.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from kompend.core.models import ParsedDocument
from kompend.core.models.attachments import Attachment
from kompend.core.paths import VaultPaths
from kompend.core.pipeline.assets import AssetStore
from kompend.core.pipeline.attachments import ingest_attachment
from kompend.core.pipeline.bounded_parse import ParseDeadlineExceeded
from kompend.core.pipeline.ingest import ingest_documents
from kompend.core.pipeline.reasons import reason_code_for
from kompend.core.pipeline.references import ReferenceStore
from kompend.core.pipeline.stages import parse_store as parse_store_stage
from kompend.core.pipeline.stages.parse_sources import _process_single_source  # noqa: PLC2701
from kompend.core.pipeline.stages.parse_store import _load_cache  # noqa: PLC2701
from kompend.core.reporting.ledger import stage_ledger
from kompend.core.store import open_store
from kompend.core.types import ReasonCode
from kompend.preprocessing.host_memory import HeavyParseGuard, InsufficientHostMemoryError
from kompend.preprocessing.parsers import DocumentParser
from kompend.preprocessing.parsers.base import BinarySourceError

_INGEST = "kompend.core.pipeline.ingest"
_PARSE_SOURCES = "kompend.core.pipeline.stages.parse_sources"
_PARSE_STORE = "kompend.core.pipeline.stages.parse_store"
_ATTACHMENTS = "kompend.core.pipeline.attachments"
_ASSETS = "kompend.core.pipeline.assets"
_REFERENCES = "kompend.core.pipeline.references"


class _StubRuntime:
    """Minimal pipeline for driving ``_process_single_source``/``_load_cache``."""

    def __init__(self, base: Path, parser: object | None) -> None:
        self.base_directory = base
        self.config = SimpleNamespace(
            no_cache=False,
            reparse=True,
            force=False,
            privacy_filter=False,
            output=SimpleNamespace(rumdl_config=None, flavor=None),
            search=SimpleNamespace(contextual=SimpleNamespace(enabled=False)),
        )
        self.parser = parser
        self.chunker = None
        self.enricher = None
        self.state = MagicMock()
        self.cache_store = MagicMock()

    def rel_file(self, file: Path) -> str:
        return str(file)

    def abs_file(self, file_str: str) -> Path:
        p = Path(file_str)
        return p if p.is_absolute() else self.base_directory / p

    def cache_directory(self) -> Path:
        return self.base_directory / ".kompend" / "cache"


class _RoutedParser:
    """Parser facade: routes everything to a fixed inner object (or nothing)."""

    def __init__(self, inner: object | None, identity: str = "stub-chain") -> None:
        self._inner = inner
        self._identity = identity

    def backend_identity(self) -> str:
        return self._identity

    def route(self, source_file: Path) -> object | None:  # noqa: ARG002
        return self._inner


# --- ingest.py ------------------------------------------------------------------


class _FakeParser:
    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour

    def parse(self, target: Path) -> ParsedDocument:
        if self.behaviour == "mislabelled":
            raise BinarySourceError(f"{Path(target).name}: extension claims text but bytes are binary")
        return _document(Path(target))


class _RoutingParser:
    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour

    def backend_identity(self) -> str:
        return "fake-parser-chain"

    def route(self, source_file: Path) -> _FakeParser:  # noqa: ARG002
        return _FakeParser(self.behaviour)


def _document(source: Path) -> ParsedDocument:
    return ParsedDocument(source_file=source, elements=[], title=source.stem)


# --- classifier ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ParseDeadlineExceeded(Path("a.md"), 5.0), ReasonCode.TIMEOUT),
        (BinarySourceError("a.md: extension claims text"), ReasonCode.MISLABELLED_BINARY),
        (RuntimeError("BinarySourceError: a.md: extension claims text"), ReasonCode.MISLABELLED_BINARY),
        (InsufficientHostMemoryError("need 8 GiB"), ReasonCode.BACKEND_UNAVAILABLE),
        (ConnectionError("refused"), ReasonCode.BACKEND_UNAVAILABLE),
        (RuntimeError("docling exploded"), ReasonCode.BACKEND_UNAVAILABLE),
        (ValueError("unclassified"), ReasonCode.BACKEND_UNAVAILABLE),
    ],
)
def test_reason_code_for_maps_each_exception_family(exc: BaseException, expected: ReasonCode) -> None:
    assert reason_code_for(exc) is expected


def test_every_member_has_a_stable_value() -> None:
    assert {member.name for member in ReasonCode} == {
        "MISLABELLED_BINARY",
        "TIMEOUT",
        "BACKEND_UNAVAILABLE",
        "JUDGE_DEGRADED",
        "DECISION_FALLBACK",
        "TRAVERSAL",
        "CORRUPT_CACHE",
        "NO_PARSER",
        "EMPTY_EXTRACTION",
        "VLM_FILTERED",
        "VLM_RELEVANCE_SKIPPED",
        "VLM_SKIPPED",
        "EMBEDDING_MERGE_DISABLED",
        "PAYLOAD_TOO_LARGE",
    }
    assert ReasonCode.MISLABELLED_BINARY == "mislabelled_binary"
    assert ReasonCode.TIMEOUT == "timeout"
    assert ReasonCode.BACKEND_UNAVAILABLE == "backend_unavailable"
    assert ReasonCode.JUDGE_DEGRADED == "judge_degraded"
    assert ReasonCode.DECISION_FALLBACK == "decision_fallback"
    assert ReasonCode.TRAVERSAL == "traversal"
    assert ReasonCode.CORRUPT_CACHE == "corrupt_cache"
    assert ReasonCode.NO_PARSER == "no_parser"
    assert ReasonCode.EMPTY_EXTRACTION == "empty_extraction"
    assert ReasonCode.VLM_FILTERED == "vlm_filtered"
    assert ReasonCode.VLM_RELEVANCE_SKIPPED == "vlm_relevance_skipped"
    assert ReasonCode.VLM_SKIPPED == "vlm_skipped"
    assert ReasonCode.EMBEDDING_MERGE_DISABLED == "embedding_merge_disabled"
    assert ReasonCode.PAYLOAD_TOO_LARGE == "payload_too_large"


def test_ingest_parse_failure_carries_the_classified_code(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "mislabelled.md"
    src.write_text("# not really text\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=_INGEST):
        result = ingest_documents(
            files=[src],
            base_dir=base,
            parser=cast("DocumentParser", _RoutingParser("mislabelled")),
            reparse=True,
            config=None,
        )

    assert result.failed == 1
    assert _codes(caplog, "Failed to parse") == {ReasonCode.MISLABELLED_BINARY}


def test_ingest_deadline_failure_carries_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    stuck = base / "sources" / "stuck.md"
    stuck.write_text("# stuck\n", encoding="utf-8")

    def _parse(
        self: object,
        parser: object,
        target: Path,
        *,
        source: str | None = None,  # noqa: ARG002 — signature parity
    ) -> ParsedDocument:
        raise ParseDeadlineExceeded(Path(target), 5.0)

    monkeypatch.setattr("kompend.core.pipeline.ingest.DeadlineWorker.parse", _parse)

    with caplog.at_level(logging.WARNING, logger=_INGEST):
        result = ingest_documents(
            files=[stuck],
            base_dir=base,
            parser=cast("DocumentParser", _RoutingParser("fine")),
            reparse=True,
            config=None,
            parse_timeout_s=5.0,
        )

    assert result.failed == 1
    assert _codes(caplog, "Failed to parse") == {ReasonCode.TIMEOUT}


# --- stages/parse_sources.py, stages/parse_store.py -----------------------------------------------------------


def test_stage_no_backend_routes_to_no_parser(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    src.write_text("# note\n", encoding="utf-8")
    runtime = _StubRuntime(base, parser=_RoutedParser(None))

    with stage_ledger(tmp_path / "led.db") as ledger, caplog.at_level(logging.WARNING, logger=_PARSE_SOURCES):
        assert _process_single_source(cast("Any", runtime), src, {}, HeavyParseGuard(), ledger, None) == []

    assert _codes(caplog, "No parser backend") == {ReasonCode.NO_PARSER}


def test_stage_backend_failure_carries_the_classified_code(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    src.write_text("# note\n", encoding="utf-8")
    boom = MagicMock()
    boom.parse.side_effect = RuntimeError("docling exploded")
    runtime = _StubRuntime(base, parser=_RoutedParser(boom))

    with stage_ledger(tmp_path / "led.db") as ledger, caplog.at_level(logging.WARNING, logger=_PARSE_SOURCES):
        assert _process_single_source(cast("Any", runtime), src, {}, HeavyParseGuard(), ledger, None) == []

    assert _codes(caplog, "Failed to parse") == {ReasonCode.BACKEND_UNAVAILABLE}


def test_stage_typed_binary_error_keeps_its_code(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    src.write_text("# note\n", encoding="utf-8")
    mislabelled = MagicMock()
    mislabelled.parse.side_effect = BinarySourceError("note.md: extension claims text")
    runtime = _StubRuntime(base, parser=_RoutedParser(mislabelled))

    with stage_ledger(tmp_path / "led.db") as ledger, caplog.at_level(logging.WARNING, logger=_PARSE_SOURCES):
        assert _process_single_source(cast("Any", runtime), src, {}, HeavyParseGuard(), ledger, None) == []

    assert _codes(caplog, "Failed to parse") == {ReasonCode.MISLABELLED_BINARY}


def test_stage_corrupt_cache_entry_reparses_with_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    src = base / "sources" / "note.md"
    src.write_text("# note\n", encoding="utf-8")
    runtime = _StubRuntime(base, parser=_RoutedParser(None, identity="stub-chain"))
    # An entry the identity check accepts but the element rebuild cannot.
    monkeypatch.setattr(
        parse_store_stage,
        "read_document",
        lambda db_path, key: {"_parser_fingerprint": "stub-chain", "elements": [{"type": "bogus", "content": "x"}]},
    )

    with caplog.at_level(logging.WARNING, logger=_PARSE_STORE):
        assert _load_cache(cast("Any", runtime), src) is None

    assert _codes(caplog, "Corrupt parse cache") == {ReasonCode.CORRUPT_CACHE}


def test_attachment_missing_file_is_corrupt_state(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    vault = VaultPaths(tmp_path)
    att = Attachment(path="ghost/x.png", source="ghost", ingestible=True)
    parser = MagicMock()

    with caplog.at_level(logging.WARNING, logger=_ATTACHMENTS):
        assert ingest_attachment(vault, att, cast("DocumentParser", parser)) is None

    assert _codes(caplog, "Attachment file not found") == {ReasonCode.CORRUPT_CACHE}


def test_attachment_image_without_vlm_is_no_parser(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    vault = VaultPaths(tmp_path)
    att = _attachment(vault, "doc/pic.png", b"\x89PNG\r\n\x1a\n")
    parser = MagicMock()

    with caplog.at_level(logging.WARNING, logger=_ATTACHMENTS):
        assert ingest_attachment(vault, att, cast("DocumentParser", parser), vlm_captioner=None) is None

    assert _codes(caplog, "requires VLM captioning") == {ReasonCode.NO_PARSER}


def test_attachment_binary_without_backend_is_no_parser(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    vault = VaultPaths(tmp_path)
    att = _attachment(vault, "doc/blob.bin", b"\x00\x01\x02\x03binary")
    parser = MagicMock()

    with caplog.at_level(logging.WARNING, logger=_ATTACHMENTS):
        assert ingest_attachment(vault, att, cast("DocumentParser", parser)) is None

    assert _codes(caplog, "Unsupported attachment type") == {ReasonCode.NO_PARSER}


def test_attachment_empty_caption_is_empty_extraction(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    vault = VaultPaths(tmp_path)
    att = _attachment(vault, "doc/pic.png", b"\x89PNG\r\n\x1a\n")
    captioner = MagicMock()
    captioner.run.return_value = ""
    parser = MagicMock()

    with caplog.at_level(logging.WARNING, logger=_ATTACHMENTS):
        assert ingest_attachment(vault, att, cast("DocumentParser", parser), vlm_captioner=captioner) is None

    assert _codes(caplog, "VLM returned empty caption") == {ReasonCode.EMPTY_EXTRACTION}


# --- attachments.py ---------------------------------------------------------------


def _attachment(vault: VaultPaths, rel: str, content: bytes) -> Attachment:
    target = vault.attachments_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return Attachment(path=rel, source=Path(rel).stem, ingestible=True)


# --- assets.py / references.py -----------------------------------------------------


def test_asset_store_corrupt_row_is_skipped_with_code(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    with open_store(base / ".kompend" / "kompend.db") as conn:
        conn.execute("INSERT INTO assets(id, kind, source, data, summary, ref_count) VALUES ('a1', 'image', 's', '{not json', '', 0)")

    with caplog.at_level(logging.WARNING, logger=_ASSETS):
        AssetStore(base / ".kompend" / "assets.json", base).load()

    assert _codes(caplog, "Asset record") == {ReasonCode.CORRUPT_CACHE}


def test_reference_store_corrupt_payload_is_skipped_with_code(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    with open_store(base / ".kompend" / "kompend.db") as conn:
        conn.execute("INSERT INTO bib_references(source, ref_id, payload) VALUES ('s', 1, '{not json')")

    with caplog.at_level(logging.WARNING, logger=_REFERENCES):
        ReferenceStore(base / ".kompend" / "references.json", base).load()

    assert _codes(caplog, "corrupt payload") == {ReasonCode.CORRUPT_CACHE}


def _codes(caplog: pytest.LogCaptureFixture, needle: str) -> set[ReasonCode]:
    """Reason codes carried by captured records whose message contains *needle*."""
    codes: set[ReasonCode] = set()
    for record in caplog.records:
        code = getattr(record, "reason_code", None)
        if needle in record.getMessage() and isinstance(code, ReasonCode):
            codes.add(code)
    return codes


def _seed_vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir(parents=True)
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    return base
