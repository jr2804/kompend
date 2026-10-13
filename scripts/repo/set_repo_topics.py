"""Set repository topics on GitHub from pyproject.toml keywords.

Was a Codeberg/Forgejo API call needing a FORGEJO_TOKEN; the project is canonical on
GitHub now, so this shells out to the authenticated `gh` CLI instead. That removes the
token plumbing entirely — no env var, no mise token lookup, no urllib.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = "jr2804/kompend"


def main() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    keywords = pyproject.get("project", {}).get("keywords", [])
    if not keywords:
        print("error: no keywords found in pyproject.toml [project.keywords]")
        sys.exit(1)

    # Absolute path rather than a bare name: a partial path is resolved through PATH,
    # so whatever `gh` shadows first on the host would be executed.
    gh = shutil.which("gh")
    if gh is None:
        print("error: gh not found on PATH")
        sys.exit(1)

    result = subprocess.run(  # noqa: S603 — fixed argv, no shell, no untrusted input
        [gh, "api", "--method", "PUT", f"repos/{REPO}/topics", "--input", "-"],
        input=json.dumps({"names": keywords}),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"error: gh api returned {result.returncode}: {result.stderr.strip()}")
        sys.exit(1)

    names = json.loads(result.stdout).get("names", [])
    print(f"Set {len(names)} topics on {REPO}:")
    for name in names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
