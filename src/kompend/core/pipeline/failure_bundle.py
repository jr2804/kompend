"""Failure repro bundles — a failed parse's evidence, copied beside it.

A parse that fails deep inside a backend leaves a traceback in the run log and
one ledger row, but the input that triggered it is still somewhere in the
vault. Reproducing the case meant re-walking the vault by hand (knox-ty8f.12).
This module drops a self-contained bundle instead::

    .kompend/tmp/failures/<sha16>/
        meta.json        measured facts: reason, parser, cache identity, sha
        traceback.txt    the exception that ended the attempt
        source<ext>      the exact bytes the parser received
        run.log          the run's flight-recorder log, when it exists
        repro.sh         scaffolds a scratch vault and re-ingests the source

The bundle is written best-effort: a full disk or a vanished source must not
turn one parse failure into two. Every write error is logged and swallowed,
the same contract as the stage ledger.

The source copy is the point. ``meta.json`` alone is a report; the bytes are
what make the directory reproducible offline. No absolute path is persisted —
the bundle is the reference.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from kompend.core.paths import VaultPaths
from kompend.core.pipeline.parse_cache import cache_key as document_cache_key
from kompend.core.reporting.handle import current_run_id
from kompend.core.types import ReasonCode
from kompend.logging import get_logger

#: Bundle format version; bump when a consumer needs to branch on the shape.
BUNDLE_SCHEMA_VERSION: Final[int] = 1
#: Hash prefix used for the bundle directory (matches the parse-cache prefix).
_HASH_CHARS: Final[int] = 16
_META_FILE: Final[str] = "meta.json"
_TRACEBACK_FILE: Final[str] = "traceback.txt"
_REPRO_FILE: Final[str] = "repro.sh"
_RUN_LOG_FILE: Final[str] = "run.log"
_SOURCE_PREFIX: Final[str] = "source"
#: Executable bit for the generated repro script (owner rwx, group/other rx).
_REPRO_MODE: Final[int] = 0o755

logger = get_logger(__name__)


def write_failure_bundle(
    *,
    base_dir: Path | str,
    source: str,
    source_file: Path,
    error: BaseException,
    reason_code: ReasonCode,
    stage: str,
    parse_target: Path | None = None,
    parser: str | None = None,
    parser_version: str | None = None,
    backend: str | None = None,
    judge_model: str | None = None,
    cache_key: str | None = None,
    sha: str = "",
    size_bytes: int = 0,
    detail: str = "",
) -> Path | None:
    """Write a repro bundle for *error* and return its directory.

    *source* is the vault-relative spelling used by the ledger. *parse_target*
    is the file the parser actually opened (an encoding sibling for text
    formats); it is copied byte-for-byte, falling back to *source_file*.

    Returns ``None`` when the bundle could not be written. Never raises.
    """
    try:
        return _write_failure_bundle(
            base_dir=base_dir,
            source=source,
            source_file=source_file,
            parse_target=parse_target,
            sha=sha,
            size_bytes=size_bytes,
            stage=stage,
            reason_code=reason_code,
            detail=detail,
            parser=parser,
            parser_version=parser_version,
            backend=backend,
            judge_model=judge_model,
            cache_key=cache_key,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001 — the bundle is best-effort by contract
        logger.warning("Could not write failure bundle for %s: %s", source, exc)
        return None


def _write_failure_bundle(
    *,
    base_dir: Path | str,
    source: str,
    source_file: Path,
    parse_target: Path | None,
    sha: str,
    size_bytes: int,
    stage: str,
    reason_code: ReasonCode,
    detail: str,
    parser: str | None,
    parser_version: str | None,
    backend: str | None,
    judge_model: str | None,
    cache_key: str | None,
    error: BaseException,
) -> Path | None:
    paths = VaultPaths(Path(base_dir))
    payload = _read_bytes(parse_target or source_file)
    digest = (sha or hashlib.sha256(payload).hexdigest()) if payload else _fallback_digest(source, sha)
    size = len(payload) if payload else size_bytes
    bundle_dir = paths.failure_bundles_dir / digest[:_HASH_CHARS]

    try:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        source_copy = _copy_source(bundle_dir, source_file, payload)
        _copy_run_log(paths, bundle_dir)
        _write_traceback(bundle_dir, error)
        _write_repro(bundle_dir, source_copy)
        meta = _metadata(
            source=source,
            source_copy=source_copy.name if source_copy is not None else None,
            digest=digest,
            size_bytes=size,
            stage=stage,
            reason_code=reason_code,
            detail=detail,
            parser=parser,
            parser_version=parser_version,
            backend=backend,
            judge_model=judge_model,
            cache_key=cache_key or (document_cache_key(source_file, digest) if digest else None),
        )
        _write_json(bundle_dir / _META_FILE, meta)
    except OSError as exc:
        logger.warning("Could not write failure bundle for %s: %s", source, exc)
        return None
    logger.info("Failure bundle for %s written to %s", source, bundle_dir)
    return bundle_dir


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _fallback_digest(source: str, sha: str) -> str:
    """Stable directory key when the input bytes are gone."""
    return sha or hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()


def _copy_source(bundle_dir: Path, source_file: Path, payload: bytes) -> Path | None:
    if not payload:
        return None
    suffix = source_file.suffix
    destination = bundle_dir / f"{_SOURCE_PREFIX}{suffix}"
    destination.write_bytes(payload)
    return destination


def _copy_run_log(paths: VaultPaths, bundle_dir: Path) -> None:
    run_id = current_run_id()
    if not run_id:
        return
    log_path = paths.run_log_file(run_id)
    try:
        if log_path.is_file():
            shutil.copyfile(log_path, bundle_dir / _RUN_LOG_FILE)
    except OSError as exc:
        logger.debug("Could not copy run log into failure bundle: %s", exc)


def _write_traceback(bundle_dir: Path, error: BaseException) -> None:
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    (bundle_dir / _TRACEBACK_FILE).write_text(text, encoding="utf-8")


def _write_repro(bundle_dir: Path, source_copy: Path | None) -> None:
    name = source_copy.name if source_copy is not None else f"{_SOURCE_PREFIX}.bin"
    script = f"""#!/bin/sh
# Reproduce the recorded failure from this bundle alone (knox-ty8f.12).
set -eu
bundle=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
scratch=${{TMPDIR:-/tmp}}/kompend-repro-$(basename "$bundle")
rm -rf "$scratch"
mkdir -p "$scratch"
cd "$scratch"
kompend init .
kompend ingest "$bundle/{name}"
"""
    path = bundle_dir / _REPRO_FILE
    path.write_text(script, encoding="utf-8")
    path.chmod(_REPRO_MODE)


def _metadata(
    *,
    source: str,
    source_copy: str | None,
    digest: str,
    size_bytes: int,
    stage: str,
    reason_code: ReasonCode,
    detail: str,
    parser: str | None,
    parser_version: str | None,
    backend: str | None,
    judge_model: str | None,
    cache_key: str | None,
) -> dict[str, object]:
    run_id = current_run_id()
    return {
        "schema": BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "source": source,
        "stage": stage,
        "reason_code": reason_code.value,
        "detail": detail,
        "parser": parser,
        "parser_version": parser_version,
        "backend": backend,
        "judge_model": judge_model,
        "cache_key": cache_key,
        "sha256": digest,
        "size_bytes": size_bytes,
        "source_copy": source_copy,
        "traceback": _TRACEBACK_FILE,
        "run_log": _RUN_LOG_FILE,
        "repro": _REPRO_FILE,
    }


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
