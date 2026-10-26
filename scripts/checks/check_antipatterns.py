#!/usr/bin/env python3
"""AST-based antipattern scanner for kompend.

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
  "Antipattern 4" enforcement from the project AGENTS.md. A crop whose bound
  is a *named constant* is not matched by the scanner at all — the constant
  is the traceable form the rule asks for.
- Hard-coded wiki file names (``"index.md"``, ``"references.md"``, …) are
  flagged: the file names live once in :mod:`kompend.core.paths` as
  ``Final[str]`` constants and every other site imports them. A hard-coded
  name is a second, silent source of truth.
- Hard-coded ``"KOMP_..."`` env-var literals in env-access positions
  (``os.environ``, ``envvar=``, ``validation_alias``, ``_set_env``) are
  flagged. The single definition lives in ``kompend/core/env.py`` (``EnvVar``);
  every consumer must reference a member. Docstring/help-text mentions are
  allowed (they document the real name, they don't read env vars).
- Module-level ``dict[str, ...]`` literals whose keys are all bare strings are
  flagged (``STR-KEYED-REGISTRY``). A name→something mapping keyed by strings
  is a registry, and its vocabulary must be a ``StrEnum`` so the keys live in
  exactly one place (the original ``PARSER_BACKENDS`` defect). Derived
  mappings (dict comprehensions over enum members) and external-standard
  vocabularies are fine — the latter waive with ``# nocheck: str-registry —
  <reason>``.
- Comparisons against a bare string literal that duplicates a member value of
  a known project ``StrEnum`` are flagged (``ENUM-VALUE-LITERAL``): compare
  against the enum member instead, so a vocabulary change cannot silently
  desync a call site. External vocabularies waive with
  ``# nocheck: enum-literal — <reason>``.

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
from functools import cache
from pathlib import Path

from _common import repo_root, require_scanned

REPO_ROOT = repo_root()
SRC_ROOT = REPO_ROOT / "src" / "kompend"
TESTS_ROOT = REPO_ROOT / "tests"

# Dependency installs go through uv alone. `pip install` / `uv pip install`
# bypass uv.lock, so nothing is tracked and the environment stops being
# reproducible - see .agents/POLICIES.md (Boundaries).
_PIP_INSTALL_RE = re.compile(r"\buv pip install\b|\bpip3? install\b")
#: The policy files must be able to name the forbidden form.
_PIP_POLICY_FILES: frozenset[Path] = frozenset({REPO_ROOT / ".agents" / "POLICIES.md", REPO_ROOT / ".agents" / "CODE.md"})
#: Documentation and manifests that may spell out an install command.
_DOC_GLOBS: tuple[str, ...] = (
    "*.md",
    "docs/**/*.md",
    ".agents/**/*.md",
    "src/**/AGENTS.md",
    "skills/**/*.md",
    "tests/**/*.md",
    "*.example",
    "pyproject.toml",
    "ruff.toml",
)

# Banned-noqa rules. Each entry: the rule code that must NEVER appear in
# a ``noqa: CODE`` comment. Add new ones here as the project decides
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

# Allow ``hexdigest()[:N]``, ``uuid.uuid4().hex[:N]`` and
# ``make_cache_key(...)[:N]`` style: SHA-256 hashes, UUID hex strings and
# cache keys are intentionally truncated to short prefixes for display/storage
# and that's not a data crop.
_HASH_TRUNCATION_RE = re.compile(r"(?:hexdigest\s*\(\s*\)|\.hex|make_cache_key\s*\(.*?\))\s*\[:\s*\d+\s*\]")

# Env-var access mechanisms. A ``"KOMP_..."`` literal on a line with one of
# these reads/sets an env var and must reference an ``EnvVar`` member.
_ENV_ACCESS_RE = re.compile(r"\b(?:os\.environ|envvar|validation_alias|_set_env|setenv\(|delenv\()\b")
_KOMP_LITERAL_RE = re.compile(r'"KOMP_[A-Z_]+"')

# Hard-coded wiki file names. The canonical names live in
# ``kompend/core/paths.py`` as ``Final[str]`` constants; a literal that
# duplicates one of them is a second source of truth for a path that
# several modules share.
_WIKI_FILENAME_RE = re.compile(r"""["']([A-Za-z0-9][A-Za-z0-9_.-]*\.md)["']""")

# The single allowed definition site for the wiki file names.
_PATHS_MODULE = SRC_ROOT / "core" / "paths.py"

# The single allowed definition site for ``KOMP_*`` env-var names.
_ENV_MODULE = SRC_ROOT / "core" / "env.py"

# STR-KEYED-REGISTRY: a module-level ``NAME: dict[str, ...] = {...}`` whose
# keys are all bare string literals is a string-keyed registry — the
# vocabulary must be a StrEnum (keys as members) or a derived comprehension.
# ``# nocheck: str-registry — <reason>`` waives external-standard maps.
_NO_CHECK_RE = re.compile(r"#\s*nocheck:", re.IGNORECASE)

#: StrEnums whose member values participate in ENUM-VALUE-LITERAL. Config /
#: registry vocabularies whose bare literals have historically leaked into
#: call sites. Extend when a new enum gains consumers.
_VOCAB_ENUM_NAMES: frozenset[str] = frozenset(
    {
        "Compiler",
        "ContextManagerPreset",
        "Flavor",
        "ImageMode",
        "MatcherName",
        "NodeType",
        "OutputFormat",
        "ParsecraftEffort",
        "ParsecraftJudge",
        "ParsecraftPreference",
        "ParserName",
        "ReasoningEffort",
        "Tier",
    }
)

# Files where module-level state is a documented design choice (singleton
# registries, plugin tables, etc.). Adding a new path here requires a
# justification comment in the file.
_GLOBAL_KEYWORD_ALLOWLIST: frozenset[str] = frozenset(
    {
        # PackRegistry singleton: lazy-initialised module-level state,
        # explicitly documented in the surrounding docstring.
        "src/kompend/packs/base.py",
    }
)


#: Vendored skill assets quote other projects' instructions - not our policy.
_DOC_EXCLUDED_PARTS: tuple[str, ...] = (".agents/skills", "tmp", "node_modules")


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
        help="Source root to scan (repeatable). Default: src/kompend + tests/unit.",
    )
    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.src] if args.src else [SRC_ROOT, TESTS_ROOT]

    py_files = discover_py_files(roots)
    doc_files = discover_doc_files()
    require_scanned(len(py_files) + len(doc_files), roots)
    all_violations: list[Violation] = []
    for f in py_files:
        all_violations.extend(scan_file(f))
    for f in doc_files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        all_violations.extend(scan_pip_installs(f, lines))

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
        touched = len({v.file for v in all_violations})
        scanned = f"{len(py_files)} python + {len(doc_files)} doc"
        print(f"check-antipatterns: {len(all_violations)} violation(s) in {touched} file(s), {scanned} scanned")
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


def discover_doc_files() -> list[Path]:
    """Markdown and manifests that carry install instructions."""
    files: set[Path] = set()
    for pattern in _DOC_GLOBS:
        for found in REPO_ROOT.glob(pattern):
            rel = found.relative_to(REPO_ROOT).as_posix()
            if any(part in rel for part in _DOC_EXCLUDED_PARTS):
                continue
            if found.is_file():
                files.add(found)
    return sorted(files)


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
    out.extend(scan_wiki_filenames(path, lines))
    out.extend(scan_str_keyed_registries(path, tree, lines))
    out.extend(scan_enum_value_literals(path, tree, lines))
    out.extend(scan_pip_installs(path, lines))
    return out


def scan_pip_installs(path: Path, lines: list[str]) -> list[Violation]:
    """Flag documentation that tells a reader to install with pip."""
    if path in _PIP_POLICY_FILES:
        return []
    out: list[Violation] = []
    for line_no, line in enumerate(lines, 1):
        match = _PIP_INSTALL_RE.search(line)
        if match is None:
            continue
        out.append(
            Violation(
                file=path,
                line=line_no,
                col=match.start() + 1,
                rule="PIP-INSTALL",
                message="dependencies arrive through uv add / uv lock / uv sync, never pip (.agents/POLICIES.md)",
                context=line.strip()[:120].encode("ascii", "replace").decode("ascii"),
            )
        )
    return out


def scan_wiki_filenames(path: Path, lines: list[str]) -> list[Violation]:
    """Flag hard-coded wiki file names that duplicate a path constant.

    A wiki file name is a shared path component: the vault writer, the index
    page, the lint fixer and the MCP tools all have to agree on it. The
    canonical names live in :mod:`kompend.core.paths` as ``Final[str]``
    constants — a literal that repeats one of them is a second source of
    truth. Names that no constant covers (scaffold templates such as
    ``README.md``) are not flagged: they are not shared path components.
    """
    if path == _PATHS_MODULE or _is_test_file(path):
        return []
    known = _path_filename_constants()
    if not known:
        return []
    out: list[Violation] = []
    for i, line in enumerate(lines, start=1):
        code = line.split("#", 1)[0]
        if '"""' in code:
            continue
        for m in _WIKI_FILENAME_RE.finditer(code):
            if m.group(1) not in known:
                continue
            out.append(
                Violation(
                    path,
                    i,
                    m.start() + 1,
                    "HARD-CODED-FILENAME",
                    f"hard-coded wiki file name {m.group(0)} — use the Final constant from kompend.core.paths",
                )
            )
    return out


@cache
def _path_filename_constants() -> frozenset[str]:
    """The ``*.md`` file names defined as ``Final[str]`` in ``core/paths.py``."""
    try:
        source = _PATHS_MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(_PATHS_MODULE))
    except (OSError, SyntaxError):
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if "Final" not in ast.unparse(node.annotation):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and node.value.value.endswith(".md"):
            names.add(node.value.value)
    return frozenset(names)


def scan_env_var_literals(path: Path, lines: list[str]) -> list[Violation]:
    """Flag hard-coded ``KOMP_*`` literals in env-access positions.

    ``kompend/core/env.py`` is the only place a ``KOMP_*`` env-var name may be
    defined. Docstrings and Typer help text (``"Env: KOMP_WIKI_DIR."``) are
    documentation, not env access — allowed.
    """
    if path == _ENV_MODULE:
        return []
    out: list[Violation] = []
    for i, line in enumerate(lines, start=1):
        if not _ENV_ACCESS_RE.search(line):
            continue
        for m in _KOMP_LITERAL_RE.finditer(line):
            out.append(
                Violation(
                    path,
                    i,
                    m.start() + 1,
                    "ENV-VAR-LITERAL",
                    f"hard-coded env var {m.group(0)} — use an EnvVar member from kompend.core.env",
                )
            )
    return out


def scan_str_keyed_registries(path: Path, tree: ast.Module, lines: list[str]) -> list[Violation]:
    """Flag module-level ``dict[str, ...]`` literals with all-bare-string keys.

    A string-keyed registry (``KNOWN_BACKENDS = {"parsecraft": ...}``) hides a
    closed vocabulary in dict keys: nothing validates it, nothing derives help
    text or schema choices from it, and every consumer re-types the strings.
    The fix is a ``StrEnum`` keyed registry (StrEnum members hash as their
    value, so raw-string lookups keep working). Derived mappings — dict
    comprehensions over enum members — are the compliant form and never match.
    """
    if _is_test_file(path):
        return []
    out: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if not _is_plain_str_keyed_dict_annotation(node.annotation):
            continue
        value = node.value
        if not isinstance(value, ast.Dict) or not value.keys:
            continue  # empty registries populated via register calls, caches, etc.
        if any(key is None for key in value.keys):
            continue  # ** merge — treat as derived
        if not all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in value.keys):
            continue
        line = lines[node.lineno - 1]
        if _NO_CHECK_RE.search(line):
            continue
        keys = ", ".join(repr(key.value) for key in value.keys[:4])
        out.append(
            Violation(
                file=path,
                line=node.lineno,
                col=node.col_offset + 1,
                rule="STR-KEYED-REGISTRY",
                message=(f"{node.target.id} is a string-keyed registry ({keys}…) — key it by a StrEnum so the vocabulary lives in one place"),
            )
        )
    return out


def _is_plain_str_keyed_dict_annotation(annotation: ast.expr) -> bool:
    """True for ``dict[str, ...]`` / ``Final[dict[str, ...]]`` annotations."""
    target = annotation
    while isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "Final":
        target = target.slice
    if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id != "dict":
        return False
    key_type = target.slice.elts[0] if isinstance(target.slice, ast.Tuple) else target.slice
    return isinstance(key_type, ast.Name) and key_type.id == "str"


def scan_enum_value_literals(path: Path, tree: ast.Module, lines: list[str]) -> list[Violation]:
    """Flag comparisons against bare string literals duplicating enum values.

    ``if preset == "sliding"`` compiles fine and silently desyncs the day the
    ``ContextManagerPreset`` vocabulary changes; ``preset is
    ContextManagerPreset.SLIDING`` cannot. Only vocabularies in
    ``_VOCAB_ENUM_NAMES`` are checked — their values are collected from the
    enum definitions in ``src/kompend`` at scan time.
    """
    if _is_test_file(path):
        return []
    values = _vocab_enum_values()
    if not values:
        return []
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not all(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
            continue
        for comparator in node.comparators:
            if not (isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)):
                continue
            enum_names = values.get(comparator.value)
            if not enum_names:
                continue
            line_idx = node.lineno - 1
            if _NO_CHECK_RE.search(lines[line_idx]):
                continue
            names = "/".join(sorted(enum_names))
            out.append(
                Violation(
                    file=path,
                    line=node.lineno,
                    col=node.col_offset + 1,
                    rule="ENUM-VALUE-LITERAL",
                    message=f'comparison with "{comparator.value}" duplicates a {names} member value — compare with the enum member',
                )
            )
    return out


@cache
def _vocab_enum_values() -> dict[str, set[str]]:
    """Map member value → enum class names for ``_VOCAB_ENUM_NAMES``.

    Collected by scanning ``src/kompend`` for ``class X(StrEnum)`` bodies.
    """
    value_map: dict[str, set[str]] = defaultdict(set)
    for py in SRC_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"), filename=str(py))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in _VOCAB_ENUM_NAMES:
                continue
            if not any(isinstance(base, ast.Name) and base.id == "StrEnum" for base in node.bases):
                continue
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    value_map[stmt.value.value].add(node.name)
    return dict(value_map)


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
    - named constants: a bound that is a module constant (``title[:MAX]``)
      is not matched by the scanner — the constant is the traceable form.

    The scanner also skips:
    - Comment lines (anything after ``#`` in a source line)
    - Lines the author opted out of via ``# noqa:`` (a justification for
      the crop is required in that case).
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
# Such a crop must use a named constant (or a config-derived limit) and
# carry a ``noqa:`` justification; the scanner no longer guesses from
# the surrounding lines, because that guess hid real data crops.
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
