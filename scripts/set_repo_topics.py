"""Set repository topics on Codeberg from pyproject.toml keywords."""

from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


def _get_token() -> str | None:
    """Get FORGEJO_TOKEN from env or mise env --json."""
    token = os.environ.get("FORGEJO_TOKEN")
    if token:
        return token
    # mise run doesn't pass [env] vars on Windows, try mise env --json
    import subprocess

    try:
        result = subprocess.run(
            ["mise", "env", "--json"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            env = json.loads(result.stdout)
            return env.get("FORGEJO_TOKEN")
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


def main() -> None:
    # Read keywords from pyproject.toml
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    keywords = pyproject.get("project", {}).get("keywords", [])
    if not keywords:
        print("error: no keywords found in pyproject.toml [project.keywords]")
        sys.exit(1)

    # Get token from env (set by mise.local.toml)
    token = _get_token()
    if not token:
        print("error: FORGEJO_TOKEN not set (add to mise.local.toml)")
        sys.exit(1)

    # Call Forgejo API
    url = "https://codeberg.org/api/v1/repos/jr2804/knox/topics"
    data = json.dumps({"topics": keywords}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Content-Type", "application/json")

    try:
        resp = urllib.request.urlopen(req)
        body = resp.read()
        if body:
            result = json.loads(body.decode("utf-8"))
            print(f"Set {len(result.get('topics', []))} topics:")
            for t in result.get("topics", []):
                print(f"  - {t}")
        else:
            print(f"Topics updated (status {resp.status})")
    except urllib.error.HTTPError as e:
        print(f"error: API returned {e.code}: {e.read().decode('utf-8')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
