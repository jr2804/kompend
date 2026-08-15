#!/usr/bin/env python3
r"""Runtime scanner for absolute paths persisted in ``.knox/`` artifacts.

Enforces Antipattern 6 from ``src/knox/AGENTS.md``: no absolute filesystem
path may be stored in any artifact under the vault's ``.knox/`` folder.
Paths must be relative to the vault ``base_dir`` and POSIX-normalized so
the vault stays relocatable.

This scanner walks ``.knox/`` after a compile and reads every JSON/JSONL
file, flagging any string value that looks like an absolute path:

- POSIX absolute: ``/home/user/...``
- Windows drive: ``C:\\Users\\...`` / ``D:/svn/...``
- Windows UNC: ``\\\\server\\share\\...``

SQLite databases (``*.db``, memory.db, search.db) and binary vector
stores are skipped — they're not human-readable JSON and would need a
schema-aware probe. The JSON/JSONL layer covers the confirmed leak
surfaces (references, assets, wiki cache, checkpoints, organize reports,
dedup, tier state).

Each violation prints ``file:field  message`` and the script exits non-zero
on any violation. Exits 0 when clean.

Usage::

    uv run python scripts/checks/check_persisted_paths.py
    uv run python scripts/checks/check_persisted_paths.py --vault /path/to/vault
    uv run python scripts/checks/check_persisted_paths.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Violation", "main", "scan_vault"]

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Vault directory (parent of .knox/). Defaults to cwd.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit violations as JSON (for CI parsing).",
    )
    args = parser.parse_args(argv)

    vault = args.vault if args.vault is not None else _default_vault()
    violations = scan_vault(vault)

    if args.json:
        payload = [{"file": v.file, "field_path": v.field_path, "value": v.value} for v in violations]
        print(json.dumps(payload, indent=2))
    else:
        for violation in violations:
            print(violation.render())

    if violations:
        print(f"\n{len(violations)} absolute-path leak(s) found in {vault}/.knox", file=sys.stderr)
        return 1
    return 0


def scan_vault(vault_dir: Path) -> list[Violation]:
    """Scan *vault_dir*/.knox and return all absolute-path violations."""
    knox_dir = vault_dir / ".knox"
    if not knox_dir.is_dir():
        return []
    acc = _ScanAccumulator()
    for artifact in _iter_artifacts(knox_dir):
        _scan_file(artifact, vault_dir, acc)
    return acc.violations


def _iter_artifacts(knox_dir: Path) -> Iterator[Path]:
    """Yield JSON/JSONL files under *knox_dir*, skipping binaries."""
    for path in sorted(knox_dir.rglob("*")):
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
    except OSError:
        return
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
    """Best-effort default vault: cwd if it contains a .knox folder."""
    cwd = Path.cwd()
    return cwd if (cwd / ".knox").is_dir() else cwd


if __name__ == "__main__":
    sys.exit(main())
