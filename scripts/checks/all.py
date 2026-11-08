#!/usr/bin/env python
"""Run every gate in one command.

The gate list used to live in whoever's memory was running the task, so one was
always skipped. This is the single entry point:

    python scripts/checks/all.py            # everything
    python scripts/checks/all.py --fast     # skip the smoke layer

Each step prints one line; the gates run in parallel, so verdict lines
arrive in completion order — every line names its gate.

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
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kompend.core.store.schema import SCHEMA_VERSION, apply_schema

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: A gate step: ``(name, argv, extra_env)``. ``extra_env`` layers over the
#: inherited environment for that subprocess only (``None`` inherits as-is).
Step = tuple[str, list[str], dict[str, str] | None]

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


def _run_steps(steps: list[Step]) -> int:
    """Run the gate steps in parallel and report one verdict for the whole run.

    Each step is a ``(name, argv, extra_env)`` triple. A step that exits 0
    judged and found nothing; exit 1 found violations; exit 2 could not judge
    at all — no subject, no facts, a scan that read nothing. The third is not
    a violation and there is nothing to fix in the source, but the gate
    certified nothing either, so the run cannot claim to be clean.

    The gates are independent processes (read-only scans plus the smoke
    layer), so they run concurrently in a thread pool; the wall-clock cost of
    the run is the slowest gate, not the sum. Every gate runs to completion —
    nothing is skipped when another gate fails.

    Returns the run's exit code: 0 when every gate judged and found nothing,
    1 when a gate found violations, 2 when a gate could not judge.
    """

    def run(step: Step) -> tuple[str, subprocess.CompletedProcess[str], float]:
        name, command, extra_env = step
        env = {**os.environ, **extra_env} if extra_env else None
        started = time.monotonic()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, cwd=ROOT, capture_output=True, text=True, check=False, env=env
        )
        return name, result, time.monotonic() - started

    workers = min(len(steps), os.cpu_count() or 2)
    failed: list[str] = []
    unjudged: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, step) for step in steps]
        for future in as_completed(futures):
            name, result, elapsed = future.result()
            if result.returncode == 0:
                print(f"[ok]   {name} ({elapsed:.1f}s)")
                continue
            if result.returncode == 2:
                unjudged.append(name)
                print(f"[n/a]  {name} ({elapsed:.1f}s, exit 2 - could not judge)")
                print(result.stderr.rstrip(), file=sys.stderr)
                continue
            failed.append(name)
            print(f"[FAIL] {name} ({elapsed:.1f}s, exit {result.returncode})")
            print(result.stdout.rstrip())
            print(result.stderr.rstrip(), file=sys.stderr)

    if failed:
        print(
            f"\nGATE FAILED - {', '.join(failed)} found violations; fix before continuing.",
            file=sys.stderr,
        )
        return 1
    if unjudged:
        print(
            f"\n{len(unjudged)} gate(s) could not judge and certified nothing: {', '.join(unjudged)}.",
            file=sys.stderr,
        )
        return 2
    print("\nAll gates passed.")
    return 0


def _steps(fast: bool, vault: Path) -> list[Step]:
    python = sys.executable
    steps: list[Step] = [
        (name, [python, str(HERE / f"check_{name}.py")], None)
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
    steps.append((VAULT_GATE, [python, str(HERE / f"check_{VAULT_GATE}.py"), "--vault", str(vault)], None))
    steps += [
        ("ruff check", [_binary("ruff"), "check", "src", "tests", "scripts"], None),
        ("ruff format", [_binary("ruff"), "format", "--check", "src", "tests", "scripts"], None),
        # Same scope as `mise run typecheck` (see .config/mise/conf.d/lint.toml).
        # A narrower scope let a type error in tests/ pass this runner while the
        # documented task failed on the same tree — two verdicts for one gate.
        ("ty", [_binary("ty"), "check", "src/", "tests/", "scripts/"], None),
    ]
    if not fast:
        # The smoke layer's own convention (tests/AGENTS.md): skip the pytest
        # plugin-autoload tax (~6s per pytest process), and zero the transient
        # backoff so a stubbed provider error cannot stall the run.
        smoke_env: dict[str, str] = {
            "KOMP_LLM_TRANSIENT_BACKOFF": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        steps.append(
            (
                "smoke",
                [
                    python,
                    "-m",
                    "pytest",
                    "tests/smoke",
                    "-q",
                    "-p",
                    "pytest_asyncio.plugin",
                    "-p",
                    "pytest_timeout",
                ],
                smoke_env,
            )
        )
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
