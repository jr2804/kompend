#!/usr/bin/env python3
r"""Runtime scanner for absolute paths persisted in ``.kompend/`` artifacts.

Enforces Antipattern 6 from ``src/kompend/AGENTS.md``: no absolute filesystem
path may be stored in any artifact under the vault's ``.kompend/`` folder.
Paths must be relative to the vault ``base_dir`` and POSIX-normalized so
the vault stays relocatable.

This scanner walks ``.kompend/`` after a compile and reads every scannable
artifact, flagging any string value that looks like an absolute path:

- POSIX absolute: ``/home/user/...``
- Windows drive: ``C:\\Users\\...`` / ``D:/svn/...``
- Windows UNC: ``\\\\server\\share\\...``

Both artifact layers are scanned, because the rule is about what the vault
persists, not about a file format:

1. **The SQLite store** (``kompend.db``) — every text column of every
   non-shadow table, read through a read-only connection. Columns holding JSON
   (``elements``, ``metadata``, ``payload``) are decoded and walked, so a path
   nested in a payload is caught too. This is where the vault's state actually
   lives: tier state, references, provenance, runs and the wiki snapshot all
   migrated out of flat files.
2. **Loose JSON/JSONL artifacts** — lint logs and anything else written beside
   the store. A stray ``.db`` holding no schema is skipped; a database that
   opens but fails mid-scan aborts the run.

Vector stores are skipped: they are binary and need a schema-aware probe.

A scan that reads no artifact at all exits 2 rather than 0 — an empty scan is
an untested vault, not a clean one. That is how this scanner stayed green for
the whole period after the store migration.

Each violation prints ``file:field  message`` and the script exits non-zero
on any violation. Exits 0 when clean.

Usage::

    uv run python scripts/checks/check_persisted_paths.py
    uv run python scripts/checks/check_persisted_paths.py --vault /path/to/vault
    uv run python scripts/checks/check_persisted_paths.py --json
"""

# ruff: noqa: S608 — the only interpolated identifiers are table and column
# names read back from ``sqlite_master``/``PRAGMA table_info`` of the vault's
# own database. SQLite offers no parameter binding for identifiers, and no
# caller-supplied string ever reaches a statement.

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Violation", "main", "scan_vault", "scan_vault_with_coverage"]

# Heuristics for detecting an absolute path inside a JSON string value.
# POSIX absolute: /home, /var, /tmp — but NOT a lone "/" or markdown links.
_POSIX_ABSOLUTE = re.compile(r"^/[a-zA-Z]")
# Windows drive letter: C:\, D:/, C:/ — both separator styles.
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:[\\/]")
# Windows UNC: \\server\share
_WINDOWS_UNC = re.compile(r"^\\{2}")


@dataclass
class Violation:
    """One absolute-path leak detected in a persisted artifact."""

    file: str
    field_path: str
    value: str

    def render(self) -> str:
        truncated = self.value if len(self.value) <= 80 else self.value[:77] + "..."
        return f"{self.file}:{self.field_path}  absolute-path-leak  {truncated!r}"


@dataclass
class _ScanAccumulator:
    violations: list[Violation] = field(default_factory=list)
    #: How many artifacts (JSON files + store tables) the scan actually read.
    #: Zero means the scan proved nothing, which is not a pass.
    scanned: int = 0


class ScanError(RuntimeError):
    """The scan itself could not run — an unreadable store, not a finding."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Vault directory (parent of .kompend/). Defaults to cwd.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit violations as JSON (for CI parsing).",
    )
    args = parser.parse_args(argv)

    vault = args.vault if args.vault is not None else _default_vault()
    try:
        acc = scan_vault_with_coverage(vault)
    except ScanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    violations = acc.violations

    if args.json:
        payload = [{"file": v.file, "field_path": v.field_path, "value": v.value} for v in violations]
        print(json.dumps(payload, indent=2))
    else:
        for violation in violations:
            print(violation.render())

    if violations:
        print(f"\n{len(violations)} absolute-path leak(s) found in {vault}/.kompend", file=sys.stderr)
        return 1
    # A scan that read nothing is not a clean vault, it is an untested one. The
    # store moved to kompend.db, so a JSON-only scanner would otherwise stay
    # green forever without reading a single artifact.
    if acc.scanned == 0:
        print(
            f"ERROR: no scannable artifacts under {vault}/.kompend - refusing to report success",
            file=sys.stderr,
        )
        return 2
    print(f"check-persisted-paths: {acc.scanned} artifact(s) scanned, no absolute paths")
    return 0


def scan_vault(vault_dir: Path) -> list[Violation]:
    """Scan *vault_dir*/.kompend and return all absolute-path violations."""
    return scan_vault_with_coverage(vault_dir).violations


def scan_vault_with_coverage(vault_dir: Path) -> _ScanAccumulator:
    """Scan *vault_dir*/.kompend, returning violations plus how much was read.

    The coverage count is what lets :func:`main` tell a clean vault from an
    untested one; callers that only care about findings use
    :func:`scan_vault`.
    """
    acc = _ScanAccumulator()
    kompend_dir = vault_dir / ".kompend"
    if not kompend_dir.is_dir():
        return acc
    for artifact in _iter_artifacts(kompend_dir):
        _scan_file(artifact, vault_dir, acc)
    for db_file in _iter_stores(kompend_dir):
        _scan_store(db_file, acc)
    return acc


def _iter_stores(kompend_dir: Path) -> Iterator[Path]:
    """Yield the SQLite stores under *kompend_dir*.

    The flat JSON artifacts this scanner was built for are gone: tier state,
    references, provenance, runs and the wiki snapshot all live in
    ``kompend.db`` now. Scanning only JSON would leave the rule unenforced.
    """
    for path in sorted(kompend_dir.rglob("*.db")):
        if path.is_file():
            yield path


def _scan_store(db_path: Path, acc: _ScanAccumulator) -> None:
    """Scan every text column of every non-shadow table in *db_path*.

    Read-only, so a scan can never damage a vault. A store the scanner cannot
    open aborts the run: a gate that silently skips the vault's only durable
    artifact is worse than no gate.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        raise ScanError(f"cannot open {db_path}: {exc}") from exc
    try:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            if not row[0].startswith("sqlite_") and "_fts" not in row[0]
        ]
        for table in tables:
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})") if _is_text(row[2])]
            if not columns:
                continue
            acc.scanned += 1
            _scan_table(conn, table, columns, acc)
    except sqlite3.DatabaseError as exc:
        # A file that is not a database at all (a stray .db, a corrupt store)
        # carries no scannable schema — skip it quietly. A database that opened
        # but failed mid-scan is a blind spot and aborts.
        if "file is not a database" not in str(exc).lower():
            raise ScanError(f"cannot scan {db_path}: {exc}") from exc
    finally:
        conn.close()


def _is_text(declared: object) -> bool:
    """True when a ``PRAGMA table_info`` type is text-like."""
    return "CHAR" in str(declared).upper() or "TEXT" in str(declared).upper() or "CLOB" in str(declared).upper()


def _scan_table(conn: sqlite3.Connection, table: str, columns: list[str], acc: _ScanAccumulator) -> None:
    """Flag absolute paths in one table's text columns."""
    selected = ", ".join(f'"{column}"' for column in columns)
    for row in conn.execute(f'SELECT rowid, {selected} FROM "{table}"'):
        rowid, values = row[0], row[1:]
        for column, value in zip(columns, values, strict=True):
            if not isinstance(value, str):
                continue
            field_path = f"{table}[{rowid}].{column}"
            if _looks_absolute(value):
                acc.violations.append(Violation(file="kompend.db", field_path=field_path, value=value))
                continue
            # Payload columns hold JSON; the leak hides inside it.
            decoded = _try_decode(value)
            if decoded is not None:
                _walk_json(decoded, field_path, "kompend.db", acc)


def _try_decode(value: str) -> Any:
    """The decoded JSON value, or ``None`` when *value* is not JSON."""
    stripped = value.lstrip()
    if stripped[:1] not in ("{", "["):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _iter_artifacts(kompend_dir: Path) -> Iterator[Path]:
    """Yield JSON/JSONL files under *kompend_dir*, skipping binaries."""
    for path in sorted(kompend_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            yield path


def _scan_file(path: Path, base_dir: Path, acc: _ScanAccumulator) -> None:
    """Scan a single JSON/JSONL artifact for absolute-path leaks."""
    rel = str(path.relative_to(base_dir).as_posix())
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ScanError(f"cannot read {path}: {exc}") from exc
    acc.scanned += 1
    if path.suffix.lower() == ".jsonl":
        for line_no, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            _walk_json(decoded, f"L{line_no}", rel, acc)
    else:
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return
        _walk_json(decoded, "", rel, acc)


def _walk_json(node: Any, field_path: str, file_rel: str, acc: _ScanAccumulator) -> None:
    """Recursively walk a decoded JSON value, flagging absolute-path strings."""
    if isinstance(node, str):
        if _looks_absolute(node):
            acc.violations.append(Violation(file=file_rel, field_path=field_path, value=node))
    elif isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{field_path}.{key}" if field_path else key
            _walk_json(value, child_path, file_rel, acc)
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _walk_json(item, f"{field_path}[{idx}]", file_rel, acc)


def _looks_absolute(value: str) -> bool:
    """True if *value* has the shape of an absolute filesystem path."""
    if len(value) < 3:
        return False
    return bool(_POSIX_ABSOLUTE.match(value) or _WINDOWS_DRIVE.match(value) or _WINDOWS_UNC.match(value))


def _default_vault() -> Path:
    """Best-effort default vault: cwd if it contains a .kompend folder."""
    cwd = Path.cwd()
    return cwd if (cwd / ".kompend").is_dir() else cwd


if __name__ == "__main__":
    sys.exit(main())
