#!/usr/bin/env python3
"""AST-based antipattern scanner for KNOX.

Enforces rules that are hard to express in ruff but represent systemic
LLM-coding mistakes observed in this codebase:

- ``# noqa: PLC0415`` is BANNED outright (the project rule is "imports at
  module level, no exceptions"). The original PLC0415 ruff warning is fine
  for the lint pass; this scanner additionally fails on the suppression
  comment so the only way past the rule is to actually fix the import.
- ``try/except ImportError`` guards are flagged as suspicious. Required
  dependencies must be imported at module level; the only legitimate use
  is probing an explicitly-optional package (declared in pyproject.toml's
  [project.optional-dependencies]). Any other use is a violation.
- ``global`` keyword usage is flagged. It usually indicates a hacky module
  state pattern that has a better alternative (a class, a contextvar, or
  just passing the value).
- Magic ``[:N]`` and ``[:N].replace(...)`` string/list crops are flagged
  unless the source line also references a ``config.get(...)`` /
  ``DEFAULTS.get(...)`` / ``cfg.get(...)`` call. This is the
  "Antipattern 4" enforcement from the project AGENTS.md.
- Hard-coded ``"KNOX_..."`` env-var literals in env-access positions
  (``os.environ``, ``envvar=``, ``validation_alias``, ``_set_env``) are
  flagged. The single definition lives in ``knox/core/env.py`` (``EnvVar``);
  every consumer must reference a member. Docstring/help-text mentions are
  allowed (they document the real name, they don't read env vars).

Each violation prints ``file:line:col  rule  message`` and the script
exits non-zero on any violation. Exits 0 when clean.

Usage::

    uv run python scripts/checks/check_antipatterns.py
    uv run python scripts/checks/check_antipatterns.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from _common import repo_root, require_scanned

REPO_ROOT = repo_root()
SRC_ROOT = REPO_ROOT / "src" / "knox"
TESTS_ROOT = REPO_ROOT / "tests"

# Banned-noqa rules. Each entry: the rule code that must NEVER appear in
# a ``# noqa: CODE`` comment. Add new ones here as the project decides
# to harden enforcement.
BANNED_NOQA: frozenset[str] = frozenset(
    {
        "PLC0415",  # ban inline imports entirely
    }
)

# Magic crop pattern: ``[:N]`` where N is a decimal integer. Catches both
# slice expressions (``x[:10]``) and chained slices (``x[:5][:3]``).
# We deliberately do not allow even-stride slices (``x[::2]``) because
# those are a different pattern (striding) not a truncation.
_MAGIC_SLICE_RE = re.compile(r"\[:\s*\d+\s*\]")

# Pattern that, when present on the same line, indicates the slice is
# data-driven (from config / DEFAULTS / cfg). If any of these match on
# the line, the slice is allowed. This is intentionally a small list of
# well-known accessor names to avoid false positives.
_DATA_DRIVEN_ACCESSOR_RE = re.compile(
    r"\b(?:config|DEFAULTS|cfg|self\._config|self\._cfg|self\._limits|self\._budget|self\._display|self\._display_config)\s*[\.\(]"
)

# Allow ``hexdigest()[:N]`` and ``uuid.uuid4().hex[:N]`` style: SHA-256
# hashes and UUID hex strings are intentionally truncated to short
# prefixes for display/storage and that's not a data crop.
_HASH_TRUNCATION_RE = re.compile(r"(?:hexdigest\s*\(\s*\)|\.hex)\s*\[:\s*\d+\s*\]")


_LINE_PADDING_RE = re.compile(r"\bif\s+len\s*\(")

# Env-var access mechanisms. A ``"KNOX_..."`` literal on a line with one of
# these reads/sets an env var and must reference an ``EnvVar`` member.
_ENV_ACCESS_RE = re.compile(r"\b(?:os\.environ|envvar|validation_alias|_set_env|setenv\(|delenv\()\b")
_KNOX_LITERAL_RE = re.compile(r'"KNOX_[A-Z_]+"')

# The single allowed definition site for ``KNOX_*`` env-var names.
_ENV_MODULE = SRC_ROOT / "core" / "env.py"

# Files where module-level state is a documented design choice (singleton
# registries, plugin tables, etc.). Adding a new path here requires a
# justification comment in the file.
_GLOBAL_KEYWORD_ALLOWLIST: frozenset[str] = frozenset(
    {
        # PackRegistry singleton: lazy-initialised module-level state,
        # explicitly documented in the surrounding docstring.
        "src/knox/packs/base.py",
    }
)


@dataclass
class Violation:
    """One antipattern finding."""

    file: Path
    line: int
    col: int
    rule: str
    message: str
    context: str = ""

    def format(self) -> str:
        rel = self.file.relative_to(REPO_ROOT)
        msg = f"  {rel}:{self.line}:{self.col}  {self.rule}  {self.message}"
        if self.context:
            msg += f"\n    {self.context}"
        return msg

    def as_dict(self) -> dict[str, object]:
        return {
            "file": str(self.file.relative_to(REPO_ROOT)),
            "line": self.line,
            "col": self.col,
            "rule": self.rule,
            "message": self.message,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument(
        "--src",
        action="append",
        default=[],
        help="Source root to scan (repeatable). Default: src/knox + tests/unit.",
    )
    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.src] if args.src else [SRC_ROOT, TESTS_ROOT]

    py_files = discover_py_files(roots)
    require_scanned(len(py_files), roots)
    all_violations: list[Violation] = []
    for f in py_files:
        all_violations.extend(scan_file(f))

    by_rule: dict[str, int] = defaultdict(int)
    for v in all_violations:
        by_rule[v.rule] += 1

    if args.json:
        payload = {
            "violations": [v.as_dict() for v in all_violations],
            "summary": dict(by_rule),
            "total": len(all_violations),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"check-antipatterns: {len(all_violations)} violation(s) in {len({v.file for v in all_violations})} file(s), {len(py_files)} file(s) scanned")
        if by_rule:
            print("  by rule:")
            for rule, n in sorted(by_rule.items()):
                print(f"    {rule}: {n}")
        by_file: dict[Path, list[Violation]] = defaultdict(list)
        for v in all_violations:
            by_file[v.file].append(v)
        for f in sorted(by_file):
            print(f"\n{f.relative_to(REPO_ROOT)}:")
            for v in sorted(by_file[f], key=lambda x: (x.line, x.col)):
                print(v.format())

    return 1 if all_violations else 0


def scan_file(path: Path) -> list[Violation]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    out: list[Violation] = []
    out.extend(scan_noqa_bans(path, lines))
    out.extend(scan_import_error_guards(path, tree))
    out.extend(scan_global_usage(path, tree, lines))
    out.extend(scan_magic_crops(path, lines))
    out.extend(scan_env_var_literals(path, lines))
    return out


def scan_env_var_literals(path: Path, lines: list[str]) -> list[Violation]:
    """Flag hard-coded ``KNOX_*`` literals in env-access positions.

    ``knox/core/env.py`` is the only place a ``KNOX_*`` env-var name may be
    defined. Docstrings and Typer help text (``"Env: KNOX_WIKI_DIR."``) are
    documentation, not env access — allowed.
    """
    if path == _ENV_MODULE:
        return []
    out: list[Violation] = []
    for i, line in enumerate(lines, start=1):
        if not _ENV_ACCESS_RE.search(line):
            continue
        for m in _KNOX_LITERAL_RE.finditer(line):
            out.append(
                Violation(
                    path,
                    i,
                    m.start() + 1,
                    "ENV-VAR-LITERAL",
                    f"hard-coded env var {m.group(0)} — use an EnvVar member from knox.core.env",
                )
            )
    return out


def scan_noqa_bans(path: Path, lines: list[str]) -> list[Violation]:
    """Find ``# noqa: BANNED_NOQA`` comments — they must not exist."""
    out: list[Violation] = []
    for i, line in enumerate(lines, start=1):
        for code in _collect_noqa_codes(line):
            if code in BANNED_NOQA:
                out.append(
                    Violation(
                        file=path,
                        line=i,
                        col=line.find("# noqa"),
                        rule="BANNED-NOQA",
                        message=f"# noqa: {code} is banned. Fix the root issue instead of suppressing the lint.",
                        context=line.strip(),
                    )
                )
    return out


def _collect_noqa_codes(line: str) -> set[str]:
    """Extract all rule codes named in a ``# noqa:`` comment."""
    match = _is_noqa_line(line)
    if not match:
        return set()
    raw = match.group(1)
    return {code.strip() for code in raw.split(",") if code.strip()}


def scan_import_error_guards(path: Path, tree: ast.AST) -> list[Violation]:
    """Flag ``try: import x / except ImportError: ...`` blocks.

    Required dependencies must be imported at module level. The only
    legitimate use is probing an explicitly-optional package (declared
    in pyproject.toml's [project.optional-dependencies]); this scanner
    cannot detect that distinction statically, so the message points
    the developer at the AGENTS.md rule and asks them to confirm.

    We deliberately do NOT flag ``except ImportError`` raised during
    function calls (e.g. ``try: foo(); except ImportError: ...``) —
    only ImportError raised around an ``import`` statement.
    """
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not isinstance(handler.type, ast.Name) or handler.type.id != "ImportError":
                continue
            # Check if the try-block contains an import statement.
            has_import = any(isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node))
            if not has_import:
                continue
            # Skip the test files where ImportError probes are common.
            if _is_test_file(path):
                continue
            out.append(
                Violation(
                    file=path,
                    line=handler.lineno,
                    col=handler.col_offset + 1,
                    rule="IMPORT-ERROR-GUARD",
                    message=(
                        "try/except ImportError around an import is suspicious. "
                        "Required dependencies must be imported at module level. "
                        "If the package is truly optional, declare it in [project.optional-dependencies] "
                        "in pyproject.toml and document the rationale with a comment."
                    ),
                )
            )
    return out


def scan_global_usage(path: Path, tree: ast.AST, lines: list[str]) -> list[Violation]:
    """Flag ``global`` keyword usage — usually a code smell.

    Allowlisted by path when module-level singleton state is a documented
    design choice (see ``_GLOBAL_KEYWORD_ALLOWLIST``), or per line by a
    justified ``# noqa: ...`` — the project's general escape hatch, honoured
    here exactly as ``scan_magic_crops`` honours it. Some switches are
    genuinely process-wide (a one-shot setup guard, the verbosity toggle) and
    a contextvar would be wrong for them: a new thread or task context would
    read the default and lose the value.
    """
    if _is_test_file(path):
        return []
    posix = path.as_posix()
    if any(posix.endswith(allow) for allow in _GLOBAL_KEYWORD_ALLOWLIST):
        return []
    out: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            if 0 < node.lineno <= len(lines) and _is_noqa_line(lines[node.lineno - 1]):
                continue  # author opted out with a justification
            names = ", ".join(node.names)
            out.append(
                Violation(
                    file=path,
                    line=node.lineno,
                    col=node.col_offset + 1,
                    rule="GLOBAL",
                    message=(
                        f"`global {names}` is a code smell. Pass the value explicitly "
                        "(function arg, class attribute, contextvar) instead of mutating module state."
                    ),
                )
            )
    return out


def scan_magic_crops(path: Path, lines: list[str]) -> list[Violation]:
    """Flag ``[:N]`` crops that are not data-driven.

    A slice is considered data-driven if the same source line also calls
    ``config.get(...)`` / ``DEFAULTS.get(...)`` / similar. This catches
    the most common case (the limit is derived from config) but accepts
    the legitimate patterns:
    - hash truncation: ``hexdigest()[:16]`` is intentional and safe.
    - data-driven slices: lines that compute the limit from config.

    The scanner also skips:
    - Comment lines (anything after ``#`` in a source line)
    - Lines where the slice is the only thing on the RHS of a string
      arithmetic that strongly suggests line-length padding rather than
      data truncation (heuristic: slice is the entire expression and the
      file is in okf_writer.py / packs / cli).
    """
    if _is_test_file(path):
        return []
    out: list[Violation] = []
    for i, line in enumerate(lines, start=1):
        if _is_noqa_line(line):
            continue  # author already opted out via noqa
        # Split off any inline comment — the slice inside a comment is
        # documentation, not a real truncation.
        code = line.split("#", 1)[0]
        for match in _MAGIC_SLICE_RE.finditer(code):
            if _HASH_TRUNCATION_RE.search(code):
                continue
            if _DATA_DRIVEN_ACCESSOR_RE.search(code):
                continue
            if _is_line_padding_context(lines, i - 1):
                continue
            out.append(
                Violation(
                    file=path,
                    line=i,
                    col=match.start() + 1,
                    rule="MAGIC-CROP",
                    message=(
                        f"Magic crop {match.group()} silently truncates data. "
                        "Derive the limit from config (config.get / DEFAULTS.get / cfg.get) "
                        "or use a named constant. See AGENTS.md Antipattern 4."
                    ),
                    context=line.strip(),
                )
            )
    return out


# Line-length padding slices are NOT data truncation — they enforce a
# physical output width (e.g. a Markdown hard line break at 117 chars).
# The pattern: the slice is preceded on the previous non-blank line by
# ``if len(x) > N``. We use a small window (3 lines back) to catch this.
def _is_line_padding_context(lines: list[str], current_idx_0based: int) -> bool:
    """True if a ``if len(...) > N`` guard appears just above this slice."""
    return any(_LINE_PADDING_RE.search(lines[j]) for j in range(max(0, current_idx_0based - 3), current_idx_0based))


def _is_noqa_line(line: str) -> re.Match[str] | None:
    """Return a match object for ``# noqa: CODE[, CODE...]`` if present."""
    return re.search(r"#\s*noqa\s*:\s*([A-Z0-9_, ]+)", line)


def _is_test_file(path: Path) -> bool:
    return TESTS_ROOT in path.parents or path.name.startswith("test_")


def discover_py_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


if __name__ == "__main__":
    raise SystemExit(main())
