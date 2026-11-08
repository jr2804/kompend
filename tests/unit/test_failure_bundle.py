"""Repro bundles: a failed parse leaves its evidence beside the ledger row."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kompend.core.paths import FAILURE_BUNDLES_DIR
from kompend.core.pipeline.failure_bundle import write_failure_bundle
from kompend.core.pipeline.parse_cache import cache_key as document_cache_key
from kompend.core.types import ReasonCode


class _BoomError(RuntimeError):
    """A parse failure with a stable message for the traceback assertion."""


def _write_bundle(tmp_path: Path, source_file: Path, *, cache_key: str | None = None) -> Path | None:
    return write_failure_bundle(
        base_dir=tmp_path,
        source="sources/paper.md",
        source_file=source_file,
        error=_BoomError("backend exploded"),
        reason_code=ReasonCode.TIMEOUT,
        stage="parse",
        parser="parsecraft",
        parser_version="1.2.3",
        backend="DoclingBackend",
        cache_key=cache_key,
        sha="",
        size_bytes=12,
        detail="backend exploded after 900s",
    )


def test_bundle_directory_is_the_input_digest(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("# Paper\n\nbody\n", encoding="utf-8")

    bundle = _write_bundle(tmp_path, source)

    assert bundle is not None
    assert bundle == tmp_path / FAILURE_BUNDLES_DIR / hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    assert bundle.parent == tmp_path / FAILURE_BUNDLES_DIR


def test_bundle_carries_the_evidence_needed_to_reproduce(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("# Paper\n\nbody\n", encoding="utf-8")

    bundle = _write_bundle(tmp_path, source)

    assert bundle is not None
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == 1
    assert meta["source"] == "sources/paper.md"
    assert meta["stage"] == "parse"
    assert meta["reason_code"] == ReasonCode.TIMEOUT.value
    assert meta["parser"] == "parsecraft"
    assert meta["parser_version"] == "1.2.3"
    assert meta["backend"] == "DoclingBackend"
    assert meta["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert meta["size_bytes"] == len(source.read_bytes())
    assert meta["source_copy"] == "source.md"
    assert meta["traceback"] == "traceback.txt"
    assert meta["repro"] == "repro.sh"
    assert "backend exploded" in (bundle / "traceback.txt").read_text(encoding="utf-8")
    assert (bundle / "source.md").read_bytes() == source.read_bytes()
    script = (bundle / "repro.sh").read_text(encoding="utf-8")
    assert "kompend init ." in script
    assert 'kompend ingest "$bundle/source.md"' in script
    assert (bundle / "repro.sh").stat().st_mode & 0o111


def test_bundle_is_deterministic_and_latest_write_wins(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("same bytes\n", encoding="utf-8")

    first = _write_bundle(tmp_path, source)
    second = _write_bundle(tmp_path, source)

    assert first == second
    assert first is not None
    assert (first / "source.md").read_text(encoding="utf-8") == "same bytes\n"


def test_cache_key_falls_back_to_the_document_key(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("same bytes\n", encoding="utf-8")

    bundle = _write_bundle(tmp_path, source)
    assert bundle is not None
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["cache_key"] == document_cache_key(source, meta["sha256"])


def test_missing_source_still_leaves_a_usable_meta_file(tmp_path: Path) -> None:
    source = tmp_path / "gone.md"

    bundle = _write_bundle(tmp_path, source)

    assert bundle is not None
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["source_copy"] is None
    assert meta["sha256"]
    assert not (bundle / "source.md").exists()


def test_a_broken_bundle_target_returns_none_instead_of_raising(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("body\n", encoding="utf-8")
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory\n", encoding="utf-8")

    result = write_failure_bundle(
        base_dir=blocker,
        source="sources/paper.md",
        source_file=source,
        error=_BoomError("backend exploded"),
        reason_code=ReasonCode.BACKEND_UNAVAILABLE,
        stage="parse",
    )

    assert result is None


def test_reason_codes_are_serialised_by_value(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("body\n", encoding="utf-8")

    bundle = _write_bundle(tmp_path, source)
    assert bundle is not None
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["reason_code"] == "timeout"


@pytest.mark.parametrize("reason", [ReasonCode.TIMEOUT, ReasonCode.BACKEND_UNAVAILABLE])
def test_bundle_accepts_every_parse_reason(tmp_path: Path, reason: ReasonCode) -> None:
    source = tmp_path / "paper.md"
    source.write_text("body\n", encoding="utf-8")

    bundle = write_failure_bundle(
        base_dir=tmp_path,
        source="sources/paper.md",
        source_file=source,
        error=_BoomError("backend exploded"),
        reason_code=reason,
        stage="parse",
    )

    assert bundle is not None
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    assert meta["reason_code"] == reason.value
