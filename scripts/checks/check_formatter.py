#!/usr/bin/env python
"""The structural sorter must be a no-op on the tree the gate judges.

``mise run format`` runs three tools over the tree: the structural sorter
(rord, the pyreorder CLI), the formatter and the import sorter. Nothing in the
gate list asked the sorter what it would do, so a formatting pass could
rewrite the code -- and break it -- without any gate noticing: the sorter
hoisted a functional TypedDict into the module constants while leaving the
class it belongs to below the dataclasses, and the dataclass field that
defaulted to it raised ``NameError`` at import. Every kompend command and all
of pytest collection died, and the tree stayed that way because every gate
that ran on it passed.

This gate asks the sorter what it would change and rewrites nothing: the
sorter's own exit code is the verdict -- ``0`` means the tree is already
sorted, ``1`` names the files it would reorder. The formatter and the import
sorter are already covered by the ``ruff format`` and ``ruff check`` gates, so
this gate asks about the one that is not.

A missing sorter is not a pass. The gate could not judge, so it exits ``2`` --
the rule every other gate follows (``.agents/CHECKS.md``, "Gate exit codes").
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: The sorter's own configuration -- the same file ``mise run format`` uses.
CONFIG = ROOT / ".config" / "reorder.toml"
#: The trees the format task sorts.
PATHS: tuple[str, ...] = ("src", "tests", "scripts")


def main() -> int:
    """Judge the tree with the sorter; return the gate's exit code."""
    if not CONFIG.is_file():
        print(f"could not judge: the sorter configuration {CONFIG} is missing")
        return 2
    sorter = shutil.which("rord")
    if sorter is None:
        print("could not judge: the sorter `rord` (pyreorder) is not installed in this environment")
        return 2

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sorter, "check", *PATHS, "--config", str(CONFIG)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
        return 0
    if result.returncode == 1:
        print(result.stdout.strip())
        print("the sorter would reorder these files -- run `mise run format` and commit the result")
        return 1
    print(f"could not judge: the sorter exited {result.returncode}")
    print(result.stdout.strip(), file=sys.stderr)
    print(result.stderr.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
