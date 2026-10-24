#!/usr/bin/env python
"""Run every gate in one command.

The gate list used to live in whoever's memory was running the task, so one was
always skipped. This is the single entry point:

    python scripts/checks/all.py            # everything
    python scripts/checks/all.py --fast     # skip the smoke layer

Each step prints one line; the run stops at the first failure and exits 1.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every repository gate.")
    parser.add_argument("--fast", action="store_true", help="skip the smoke layer")
    args = parser.parse_args()

    failures = 0
    for name, command in _steps(args.fast):
        started = time.monotonic()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, cwd=ROOT, capture_output=True, text=True, check=False
        )
        elapsed = time.monotonic() - started
        if result.returncode == 0:
            print(f"[ok]   {name} ({elapsed:.1f}s)")
            continue
        failures += 1
        print(f"[FAIL] {name} ({elapsed:.1f}s, exit {result.returncode})")
        print(result.stdout.rstrip())
        print(result.stderr.rstrip(), file=sys.stderr)
        break

    if failures:
        print("\nGATE FAILED - fix the step above before continuing.", file=sys.stderr)
        return 1
    print("\nAll gates passed.")
    return 0


def _steps(fast: bool) -> list[tuple[str, list[str]]]:
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
            "persisted_paths",
        )
    ]
    steps += [
        ("ruff check", [_binary("ruff"), "check", "src", "tests", "scripts"]),
        ("ruff format", [_binary("ruff"), "format", "--check", "src", "tests", "scripts"]),
        ("ty", [_binary("ty"), "check", "src/kompend"]),
    ]
    if not fast:
        steps.append(("smoke", [python, "-m", "pytest", "tests/smoke", "-q"]))
    return steps


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
