#!/usr/bin/env python3
"""DOX file size-budget and duplication scanner.

Enforces the DOX authoring contract from ``.agents/POLICIES.md``:

- ``.agents/`` files: ≤ **120 lines** (always-injected context).
- Subtree ``AGENTS.md`` files: ≤ **250 lines** (loaded cumulatively with
  every parent on entry).
- Near-verbatim sentence duplication across DOX files is flagged.

Exits non-zero on any violation. Exits 0 when clean.

Usage::

    uv run python scripts/checks/check_dox.py
    uv run python scripts/checks/check_dox.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ── Budgets ──────────────────────────────────────────────────────────────
AGENTS_FILE_BUDGET = 120  # .agents/*.md
SUBTREE_BUDGET = 250  # subtree AGENTS.md

#: The fixed core state files under ``.agents/`` — always-injected context, so
#: their size is budgeted. Everything else in ``.agents/`` (skills, plans,
#: archived history) is not project DOX and is out of scope.
CORE_AGENTS_FILES: tuple[str, ...] = (
    "ONBOARDING.md",
    "POLICIES.md",
    "FILES.md",
    "CHECKS.md",
    "MAINTENANCE.md",
    "HISTORY.md",
)

# Files exempt from size budget (root AGENTS.md is always loaded but is the
# project rail — it's allowed to be larger than a subtree doc).
SIZE_EXEMPT: set[str] = set()

# ── Duplication detection ────────────────────────────────────────────────
# Minimum sentence length to consider for dup detection (avoids flagging
# short common phrases like "## Purpose" or "## Ownership").
_MIN_DUP_LEN = 40

# Ignore sentences matching these patterns (headings, list markers, code).
_IGNORE_RE = re.compile(r"^(#+\s|[-*]\s|\d+\.\s|```|`[^`]+`$|\||\[rtk:|<!--|// )")


def main() -> int:
    # Findings quote file contents, which can hold any Unicode (→, ≤, typographic
    # quotes). A Windows console defaults to cp1252, so printing them raises
    # UnicodeEncodeError — the checker crashed instead of reporting, and a
    # crashed gate reads as "no findings".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="DOX file size-budget and duplication scanner")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output violations as JSON array",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: cwd)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    files = _collect_dox_files(root)

    violations: list[str] = []
    for path in files:
        violations.extend(_check_size(path, root))
    violations.extend(_check_duplication(files))

    if args.json:
        print(json.dumps(violations, indent=2))
    else:
        for v in violations:
            print(v)

    return 1 if violations else 0


def _collect_dox_files(root: Path) -> list[Path]:
    """Return AGENTS.md files and core .agents/ policy files under *root*.

    ``.agents/`` holds two different things: the fixed core state files (a
    size budget matters — they are always injected) and material that is not
    project DOX at all (``skills/**``, ``plans/``, ``history/``, including the
    parallel example AGENTS.md files a skill ships as reference). Only the core
    files are checked; skipping the rest is what keeps the duplication check
    from flagging those examples as redundancy.
    """
    files: list[Path] = []
    # All AGENTS.md files outside .agents/ (root + subtree DOX contracts).
    agents_root = root / ".agents"
    for p in root.rglob("AGENTS.md"):
        if ".venv" in p.parts or "node_modules" in p.parts or ".git" in p.parts:
            continue
        if agents_root in p.parents:
            continue
        files.append(p)
    # Core .agents/ state files only (not skills/, plans/, history/, etc.)
    if agents_root.is_dir():
        for name in CORE_AGENTS_FILES:
            p = agents_root / name
            if p.is_file():
                files.append(p)
    return sorted(files)


def _check_size(path: Path, root: Path) -> list[str]:
    """Check line count against budget. Return violations."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    count = len(lines)

    if _is_subtree_agents(path, root):
        budget = SUBTREE_BUDGET
        label = "subtree AGENTS.md"
    elif _is_agents_core(path, root):
        budget = AGENTS_FILE_BUDGET
        label = ".agents/ core file"
    else:
        return []  # root AGENTS.md — no budget

    if count > budget:
        return [f"{path}: {count} lines exceeds {label} budget of {budget}"]
    return []


def _is_subtree_agents(path: Path, root: Path) -> bool:
    """True if *path* is a subtree AGENTS.md (not root, not .agents/)."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    if rel == Path("AGENTS.md"):
        return False
    if ".agents" in rel.parts:
        return False
    return rel.name == "AGENTS.md"


def _is_agents_core(path: Path, root: Path) -> bool:
    """True if *path* is a core .agents/ policy file."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    if rel.parts[:1] != (".agents",):
        return False
    return rel.name in ("POLICIES.md", "CHECKS.md", "FILES.md", "MAINTENANCE.md", "HISTORY.md")


def _check_duplication(files: list[Path]) -> list[str]:
    """Check for near-verbatim sentence duplication across files."""
    violations: list[str] = []
    sentence_map: dict[str, list[Path]] = {}

    for path in files:
        text = path.read_text(encoding="utf-8")
        for sentence in _extract_sentences(text):
            # Use first 60 chars as a fingerprint
            fingerprint = sentence[:60].lower()
            sentence_map.setdefault(fingerprint, []).append(path)

    for fingerprint, paths in sentence_map.items():
        if len(paths) >= 2:
            # Deduplicate to unique paths
            unique = sorted(set(paths))
            if len(unique) >= 2:
                locations = ", ".join(str(p) for p in unique)
                violations.append(f"Duplicate sentence (fingerprint {fingerprint!r}) found in: {locations}")

    return violations


def _extract_sentences(text: str) -> list[str]:
    """Split *text* into sentences, filtering out headings, code, and tables."""
    # Split on sentence boundaries (. ! ? followed by space or end) AND newlines
    raw = re.split(r"(?<=[.!?])\s+|\n", text)
    result = []
    for line in raw:
        s = line.strip()
        if len(s) < _MIN_DUP_LEN:
            continue
        if _IGNORE_RE.match(s):
            continue
        # Skip table rows
        if s.startswith("|") or "|" in s and s.count("|") >= 3:
            continue
        # Normalize whitespace
        s = re.sub(r"\s+", " ", s)
        result.append(s)
    return result


if __name__ == "__main__":
    sys.exit(main())
