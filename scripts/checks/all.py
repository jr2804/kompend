#!/usr/bin/env python
"""Run every gate in one command.

The gate list used to live in whoever's memory was running the task, so one was
always skipped. This is the single entry point:

    python scripts/checks/all.py            # everything
    python scripts/checks/all.py --fast     # skip the smoke layer

Each step prints one line; the run stops at the first failure.

A gate's exit code says what it found, and the runner keeps that meaning
(``.agents/CHECKS.md``, "Gate exit codes"):

* ``0`` — clean;
* ``1`` — violations;
* ``2`` — the gate could not establish what it needs to judge.

The runner's own exit code follows the same convention: ``1`` when a gate found
violations, ``2`` when a gate could not judge, ``0`` only when every gate
judged and found nothing. A runner that reported an unjudged gate as passed
would be the same confidently-wrong bug the exit-2 code exists to prevent.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from kompend.core.store.schema import SCHEMA_VERSION, apply_schema

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: The gate that scans a VAULT's persisted artifacts rather than the
#: repository's source. The repository root is not a vault — it has no
#: ``kompend.toml`` and its ``.kompend/`` is gitignored, so a fresh checkout
#: never has one — and the gate rightly refuses to certify a scan that read
#: nothing. The runner gives it a subject instead: a throwaway vault holding a
#: real store, built from the repository's own schema code.
VAULT_GATE = "persisted_paths"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every repository gate.")
    parser.add_argument("--fast", action="store_true", help="skip the smoke layer")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="kompend-gate-vault-") as tmp:
        vault = Path(tmp)
        _build_vault(vault)
        steps = _steps(args.fast, vault)
        return _run_steps(steps)


def _run_steps(steps: list[tuple[str, list[str]]]) -> int:
    """Run the gate steps and report one verdict for the whole run.

    Each step is a ``(name, argv)`` pair. A step that exits 0 judged and found
    nothing; exit 1 found violations; exit 2 could not judge at all — no
    subject, no facts, a scan that read nothing. The third is not a violation
    and there is nothing to fix in the source, but the gate certified nothing
    either, so the run cannot claim to be clean.

    Returns the run's exit code: 0 when every gate judged and found nothing,
    1 when a gate found violations, 2 when a gate could not judge.
    """
    unjudged: list[str] = []
    for name, command in steps:
        started = time.monotonic()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, cwd=ROOT, capture_output=True, text=True, check=False
        )
        elapsed = time.monotonic() - started
        if result.returncode == 0:
            print(f"[ok]   {name} ({elapsed:.1f}s)")
            continue
        if result.returncode == 2:
            unjudged.append(name)
            print(f"[n/a]  {name} ({elapsed:.1f}s, exit 2 - could not judge)")
            print(result.stderr.rstrip(), file=sys.stderr)
            continue
        print(f"[FAIL] {name} ({elapsed:.1f}s, exit {result.returncode})")
        print(result.stdout.rstrip())
        print(result.stderr.rstrip(), file=sys.stderr)
        print("\nGATE FAILED - fix the step above before continuing.", file=sys.stderr)
        return 1

    if unjudged:
        print(
            f"\n{len(unjudged)} gate(s) could not judge and certified nothing: {', '.join(unjudged)}.",
            file=sys.stderr,
        )
        return 2
    print("\nAll gates passed.")
    return 0


def _steps(fast: bool, vault: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    steps: list[tuple[str, list[str]]] = [
        (name, [python, str(HERE / f"check_{name}.py")])
        for name in (
            "enum_cache",
            "config_literals",
            "cli_surface",
            "antipatterns",
            "dox",
            "field_consistency",
            "mcp_parity",
        )
    ]
    steps.append((VAULT_GATE, [python, str(HERE / f"check_{VAULT_GATE}.py"), "--vault", str(vault)]))
    steps += [
        ("ruff check", [_binary("ruff"), "check", "src", "tests", "scripts"]),
        ("ruff format", [_binary("ruff"), "format", "--check", "src", "tests", "scripts"]),
        # Same scope as `mise run typecheck` (see .config/mise/conf.d/lint.toml).
        # A narrower scope let a type error in tests/ pass this runner while the
        # documented task failed on the same tree — two verdicts for one gate.
        ("ty", [_binary("ty"), "check", "src/", "tests/", "scripts/"]),
    ]
    if not fast:
        steps.append(("smoke", [python, "-m", "pytest", "tests/smoke", "-q"]))
    return steps


def _build_vault(vault: Path) -> None:
    """Create a throwaway vault holding a real store, for the vault gate.

    The gate refuses to certify a scan that read nothing, and the repository
    root is not a vault. Giving it a store built by the repository's own
    schema code makes the gate exercise the real DDL — and makes a leaked
    absolute path in a schema default visible instead of unjudgeable.
    """
    db_path = vault / ".kompend" / "kompend.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        apply_schema(conn, SCHEMA_VERSION)
        conn.commit()
    finally:
        conn.close()


def _binary(name: str) -> str:
    """Locate a console script in the active environment, or fail loudly."""
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).with_name(f"{name}.exe" if sys.platform == "win32" else name)
    if candidate.is_file():
        return str(candidate)
    raise SystemExit(f"ERROR: `{name}` is not installed in this environment ({sys.executable})")


if __name__ == "__main__":
    sys.exit(main())
