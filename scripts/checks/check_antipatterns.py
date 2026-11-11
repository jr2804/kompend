#!/usr/bin/env python3
"""AST-based antipattern scanner for kompend.

Enforces rules that are hard to express in ruff but represent systemic
LLM-coding mistakes observed in this codebase:

- ``# noqa: PLC0415`` is BANNED outright (the project rule is "imports at
  module level, no exceptions"). The original PLC0415 ruff warning is fine
  for the lint pass; this scanner additionally fails on the suppression
  comment so the only way past the rule is to actually fix the import.
- ``# noqa: BLE001`` and ``# noqa: S110`` are BANNED for the same reason.
  A broad ``except`` is fixed, not silenced: BLE001 is not even a selected
  ruff rule, so the marker hid the shape without silencing anything, while
  S110 silenced a live finding outright.
- ``try/except ImportError`` guards are flagged as suspicious. Required
  dependencies must be imported at module level; the only legitimate use
  is probing an explicitly-optional package (declared in pyproject.toml's
  [project.optional-dependencies]). Any other use is a violation.
- ``global`` keyword usage is **not** checked here: ruff's ``PLW0603`` owns
  it (``PL`` is selected in ``ruff.toml`` and the rule is not ignored), and
  the checker previously re-implemented it with an allowlist that is now
  empty. One rule, one owner.
- Magic ``[:N]`` and ``[:N].replace(...)`` string/list crops are flagged
  unless the source line also references a ``config.get(...)`` /
  ``DEFAULTS.get(...)`` / ``cfg.get(...)`` call. This is the "never crop"
  rule from the root ``AGENTS.md`` (table row in ``.agents/CODE.md``). A crop
  whose bound is a *named constant* is not matched by the scanner at all —
  the constant is the traceable form the rule asks for. No ruff rule covers
  this: ``PLR2004`` flags a magic value in a *comparison*, never a slice, and
  it is in the ``ignore`` list here anyway. This checker is the only
  enforcement of the project's cardinal rule, so do not delete it as
  "duplicate".
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
  desync a call site. The same rule covers the keyword arguments that are the
  documented carrier of a vocabulary (``detail=`` for ``ProgressDetail``,
  ``kind=`` for ``CaptureKind``, ``page_type=`` for ``SourceZone``, ``step=``
  for ``LintStep``) — see ``_VOCAB_KEYWORD_ARGUMENTS``. External vocabularies
  and unrelated arguments that merely share a spelling waive with
  ``# nocheck: enum-literal — <reason>``.
- ``if TYPE_CHECKING:`` blocks are flagged (``TYPE-CHECKING-BLOCK``). They
  split runtime from the type checker: a name imported only under the guard is
  invisible to an import cycle test and resolves to a different object than the
  code it stands for. The fix is a runtime import, a protocol
  (:mod:`kompend.agents.base.protocols`), or a local import cycle break — never
  a guard.
- ``# pyreorder: off`` markers are flagged (``PYREORDER-MARKER``). The sorter is
  a project convention, not an obstacle; a module that cannot be sorted is a
  module whose ordering needs a decision, and the marker hides that decision.
- A file-level ``# ruff: noqa`` comment is flagged (``MODULE-NOQA``) when it
  names no rule codes, or names codes but states no reason. Nineteen of these
  exist and every one is load-bearing, so the rule is not "ban them": an
  unjustified module-wide suppression is a licence nobody audits. Say which
  codes and why, in the comment.
- A bare string literal in a ``reason_code`` position — the keyword argument
  or the ``extra`` dict key the item path uses to carry a failure reason —
  is flagged (``REASON-CODE-LITERAL``). The reason-code vocabulary is a
  ``StrEnum`` (:class:`kompend.core.types.ReasonCode`) precisely so the
  ledger, doctor and explain screens can count a failure class instead of
  pattern-matching prose; a bare string there desyncs the log from the
  vocabulary it claims to speak. The position is the signal, not the value:
  a typo'd reason ("timeouts") is the case worth catching, and a string
  that happens to match a member value is no more correct than one that
  does not. Waive with ``# nocheck: reason-code — <reason>``.

Each violation prints ``file:line:col  rule  message`` and the script
exits non-zero on any violation. Exits 0 when clean, and **2** when the scan
itself could not be trusted (nothing to scan, or a file it could not read or
parse) — a gate that cannot fail is worse than no gate.

Comment scanning is tokenizer-based, not ``line.split("#")``: a ``#`` inside a
string or a docstring is not a comment, and ``#noqa`` / ``#  noqa`` are valid
suppressions that a plain ``str.find`` misses.

Usage::

    uv run python scripts/checks/check_antipatterns.py
    uv run python scripts/checks/check_antipatterns.py --json
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from _common import ScanError, repo_root, require_scanned

REPO_ROOT = repo_root()
SRC_ROOT = REPO_ROOT / "src" / "kompend"
TESTS_ROOT = REPO_ROOT / "tests"
#: Every root the project owns Python code in. One declared set, used by every
#: rule: a violation in a gate script is the same blind spot as one in the
#: package, and while the scans kept separate lists they drifted apart. The
#: package and the tests stay named so `_PATHS_MODULE` and the vocabulary scan
#: keep a single package root to read facts from. ``--src`` overrides the set.
PROJECT_ROOTS: tuple[Path, ...] = (
    SRC_ROOT,
    TESTS_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / ".config" / "mise" / "scripts",
)

# Dependency installs go through uv alone. `pip install` / `uv pip install`
# bypass uv.lock, so nothing is tracked and the environment stops being
# reproducible - see .agents/POLICIES.md (Boundaries).
_PIP_INSTALL_RE = re.compile(r"\buv pip install\b|\bpip3? install\b")
#: Files that must be able to name the forbidden form: the policy docs that
#: forbid it, and this module, whose pattern and docstring spell it out. The
#: exemption is by exact path — an install anywhere else is still a finding.
_PIP_EXEMPT_FILES: frozenset[Path] = frozenset(
    {
        REPO_ROOT / ".agents" / "POLICIES.md",
        REPO_ROOT / ".agents" / "CODE.md",
        Path(__file__).resolve(),
    }
)
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
        "BLE001",  # ban the marker that hides a broad except
        "S110",  # ban silencing a live try/except/pass finding
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
# ``kompend/core/paths/`` as ``Final[str]`` constants; a literal that
# duplicates one of them is a second source of truth for a path that
# several modules share.
_WIKI_FILENAME_RE = re.compile(r"""["']([A-Za-z0-9][A-Za-z0-9_.-]*\.md)["']""")

# The single allowed definition site for the wiki file names.
_PATHS_MODULE = SRC_ROOT / "core" / "paths" / "names.py"

# The package that owns those names. Siblings such as ``resolve.py`` must not
# repeat a constant either, so the whole package is exempt from the scan.
_PATHS_DIR = _PATHS_MODULE.parent

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
        "ReasoningEffort",
        "Tier",
        # Added by the antipattern epic: vocabularies whose literals had leaked
        # into call sites or into a second hand-written list.
        "CaptureKind",
        "LintStep",
        "ProgressDetail",
        "SourceZone",
    }
)

#: Keyword arguments that are the documented carrier of a project vocabulary.
#: Pairs of (argument name, owning StrEnum) rather than a name-keyed dict: a
#: dict here would itself be the STR-KEYED-REGISTRY shape it sits beside.
#: Only these positions are checked — an unrelated ``mode="json"`` (pydantic)
#: or ``name="capture"`` (an agent name) merely shares a spelling with a
#: member value and is not drift.
_VOCAB_KEYWORD_ARGUMENTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("detail", "ProgressDetail"),
        ("kind", "CaptureKind"),
        ("page_type", "SourceZone"),
        ("step", "LintStep"),
    }
)

#: Prefix of a file-level ruff suppression comment.
_RUFF_NOQA_PREFIX_RE = re.compile(r"^#\s*ruff:\s*noqa", re.IGNORECASE)
#: A ruff rule code (``S608``, ``I001``) or a whole prefix (``S``, ``PLR``),
#: which ruff accepts as a file-level suppression too.
_RUFF_CODE_RE = re.compile(r"^[A-Z]{1,4}[0-9]{0,4}$")

#: Shortest reason accepted after a suppression's codes. Shorter is a shrug,
#: not a justification.
_MIN_NOQA_REASON_CHARS = 8

#: Vendored skill assets quote other projects' instructions - not our policy.
_DOC_EXCLUDED_PARTS: tuple[str, ...] = (".agents/skills", "tmp", "node_modules")


# Line-length padding slices are NOT data truncation — they enforce a
# physical output width (e.g. a Markdown hard line break at 117 chars).
# Such a crop must use a named constant (or a config-derived limit) and
# carry a ``noqa:`` justification; the scanner no longer guesses from
# the surrounding lines, because that guess hid real data crops.

#: A ``noqa`` directive at the start of a comment. Anchored, so prose that
#: merely mentions the directive mid-sentence does not match; tolerant of the
#: no-space and double-space spellings ruff accepts. Group 1 is the code list,
#: empty for a blanket directive.
_NOQA_RE = re.compile(r"#\s*noqa(?:\s*:\s*(.*))?$", re.IGNORECASE)

#: Separator inside a code list: ruff accepts commas and whitespace alike.
_NOQA_CODE_SEP_RE = re.compile(r"[,\s]+")


#: The pyreorder opt-out marker the sorter reads.
_PYREORDER_OFF_RE = re.compile(r"#\s*pyreorder:\s*off", re.IGNORECASE)

#: Token types whose text is literal, never code. F-string expressions are made
#: of ordinary tokens between these, so they survive the blanking.
_STRING_TOKEN_TYPES: frozenset[int] = frozenset(
    {
        tokenize.STRING,
        tokenize.FSTRING_START,
        tokenize.FSTRING_MIDDLE,
        tokenize.FSTRING_END,
    }
)


#: The logger ``extra`` key the item path uses to carry a failure reason.
_REASON_CODE_KEY = "reason_code"


#: Width of the source line quoted beside a PIP-INSTALL finding — a preview,
#: not a data crop: the finding names the line, the line is still in the file.
_CONTEXT_PREVIEW_CHARS = 120

#: How many registry keys a STR-KEYED-REGISTRY message previews before "…".
_REGISTRY_PREVIEW_KEYS = 4


@dataclass(frozen=True)
class Comment:
    """One ``#`` comment token: its column and its text."""

    col: int
    text: str


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
        help="Source root to scan (repeatable). Default: every project Python root.",
    )
    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.src] if args.src else list(PROJECT_ROOTS)

    try:
        py_files = discover_py_files(roots)
    except ScanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    doc_files = discover_doc_files()
    # Per set, not on the sum: doc files are discovered repo-wide, so a sum
    # would let ``--src <empty dir>`` scan zero Python files and still pass.
    require_scanned(len(py_files), roots)
    require_scanned(len(doc_files), [REPO_ROOT])
    all_violations: list[Violation] = []
    try:
        for f in py_files:
            all_violations.extend(scan_file(f))
    except ScanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for f in doc_files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"ERROR: cannot read {f}: {exc}", file=sys.stderr)
            return 2
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
        print(
            f"check-antipatterns: {len(all_violations)} violation(s) in {touched} file(s) with findings, {len(py_files)} python + {len(doc_files)} doc scanned",
        )
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
    """Scan one Python file; unreadable or unparseable files abort the run.

    A file the checker cannot read or parse is not a pass: it is a blind spot
    in a gate that is supposed to catch a whole bug class. The caller turns the
    raised :class:`ScanError` into exit code 2.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ScanError(f"cannot read {path}: {exc}") from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ScanError(f"cannot parse {path}:{exc.lineno}: {exc.msg}") from exc
    code_lines, comments = _split_comments(source)
    out: list[Violation] = []
    out.extend(scan_noqa_bans(path, comments))
    out.extend(scan_import_error_guards(path, tree))
    out.extend(scan_magic_crops(path, code_lines, comments))
    out.extend(scan_env_var_literals(path, code_lines))
    out.extend(scan_wiki_filenames(path, code_lines))
    out.extend(scan_str_keyed_registries(path, tree, code_lines, comments))
    out.extend(scan_enum_value_literals(path, tree, code_lines, comments))
    out.extend(scan_type_checking_blocks(path, tree, comments))
    out.extend(scan_pyreorder_markers(path, comments))
    out.extend(scan_module_noqa(path, comments))
    out.extend(scan_reason_code_literals(path, tree, code_lines, comments))
    # Comments keep their own scan: a documented install command is still an
    # install command, and it is the form that actually reaches a reader.
    out.extend(scan_pip_installs(path, source.splitlines()))
    return out


def scan_pip_installs(path: Path, lines: list[str]) -> list[Violation]:
    """Flag documentation that tells a reader to install with pip."""
    if path.resolve() in _PIP_EXEMPT_FILES:
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
                context=line.strip()[:_CONTEXT_PREVIEW_CHARS].encode("ascii", "replace").decode("ascii"),
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
    if _PATHS_DIR in path.parents or _is_test_file(path):
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
    """The ``*.md`` file names defined as ``Final[str]`` in ``core/paths/``.

    An unreadable or unparseable paths module means the rule has no facts to
    judge by, so it raises :class:`ScanError` rather than returning an empty
    set — an empty set reads as "no path constants exist" and disarms
    WIKI-FILENAME without saying so.
    """
    try:
        source = _PATHS_MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(_PATHS_MODULE))
    except (OSError, SyntaxError) as exc:
        raise ScanError(f"cannot collect path constants from {_PATHS_MODULE}: {exc}") from exc
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


def scan_str_keyed_registries(path: Path, tree: ast.Module, lines: list[str], comments: dict[int, Comment]) -> list[Violation]:
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
        if _waived(node.lineno, comments):
            continue
        keys = ", ".join(repr(key.value) for key in value.keys[:_REGISTRY_PREVIEW_KEYS])
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


def scan_enum_value_literals(path: Path, tree: ast.Module, lines: list[str], comments: dict[int, Comment]) -> list[Violation]:
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
            if _waived(node.lineno, comments):
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
    out.extend(_enum_keyword_literals(path, tree, values, comments))
    return out


def _enum_keyword_literals(
    path: Path,
    tree: ast.Module,
    values: dict[str, set[str]],
    comments: dict[int, Comment],
) -> list[Violation]:
    """Flag a vocabulary value passed as the vocabulary's own keyword argument.

    ``emit(detail="done")`` is the same desync as ``x == "done"``: the emitter
    claims to speak :class:`ProgressDetail` and then spells a member by hand.
    Position is what makes it checkable — only ``_VOCAB_KEYWORD_ARGUMENTS``
    are inspected, so pydantic's ``mode="json"`` (which collides with
    ``OutputFormat.JSON``) is never reported.
    """
    out: list[Violation] = []
    by_argument = dict(_VOCAB_KEYWORD_ARGUMENTS)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg is None or not isinstance(keyword.value, ast.Constant):
                continue
            owner = by_argument.get(keyword.arg)
            if owner is None or not isinstance(keyword.value.value, str):
                continue
            if owner not in values.get(keyword.value.value, set()):
                continue
            if _waived(node.lineno, comments):
                continue
            out.append(
                Violation(
                    file=path,
                    line=node.lineno,
                    col=node.col_offset + 1,
                    rule="ENUM-VALUE-LITERAL",
                    message=(f"{keyword.arg}={keyword.value.value!r} duplicates a {owner} member value — pass the enum member"),
                )
            )
    return out


def scan_type_checking_blocks(path: Path, tree: ast.Module, comments: dict[int, Comment]) -> list[Violation]:
    """Flag ``if TYPE_CHECKING:`` guards.

    The guard splits a name's runtime identity from the type checker's view:
    a symbol that exists only under the guard is a different object than the
    code it stands for, invisible to an import-cycle test, and it makes the
    module's import graph a fiction. Break the cycle (a protocol, a runtime
    import, a local import) instead of hiding it.
    """
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"):
            continue
        if _waived(node.lineno, comments):
            continue
        guarded = len(node.body)
        out.append(
            Violation(
                file=path,
                line=node.lineno,
                col=node.col_offset + 1,
                rule="TYPE-CHECKING-BLOCK",
                message=(f"if TYPE_CHECKING guards {guarded} statement(s) — import at runtime, or break the cycle with a protocol"),
            )
        )
    return out


def scan_pyreorder_markers(path: Path, comments: dict[int, Comment]) -> list[Violation]:
    """Flag ``# pyreorder: off`` markers.

    The sorter is the project convention for statement order. A module that
    cannot be sorted holds an ordering decision nobody wrote down, and the
    marker is what keeps it undocumented.
    """
    out: list[Violation] = []
    for line_no, comment in sorted(comments.items()):
        if not _PYREORDER_OFF_RE.search(comment.text):
            continue
        out.append(
            Violation(
                file=path,
                line=line_no,
                col=comment.col + 1,
                rule="PYREORDER-MARKER",
                message="# pyreorder: off hides an ordering decision — order the module, or state why in prose",
                context=comment.text.strip(),
            )
        )
    return out


def scan_module_noqa(path: Path, comments: dict[int, Comment]) -> list[Violation]:
    """Flag a file-level ``# ruff: noqa`` that names no codes or no reason.

    Every module-wide suppression in this tree is load-bearing (they silence
    ``S608`` on SQL built from ``Final[str]`` table names, and one ``I001``),
    so banning them would be a rule that lies. What is worth catching is the
    unexamined one: a blanket with no codes silences everything, and one with
    no stated reason cannot be audited when the code under it changes.
    """
    out: list[Violation] = []
    for line_no, comment in sorted(comments.items()):
        text = comment.text.strip()
        if not _RUFF_NOQA_PREFIX_RE.match(text):
            continue
        codes, reason = _split_module_noqa(text)
        if not codes:
            message = "# ruff: noqa names no rule codes — it silences every rule; name the codes it silences"
        elif len(reason) < _MIN_NOQA_REASON_CHARS:
            message = f"# ruff: noqa: {', '.join(sorted(codes))} states no reason — say why the codes are safe to silence here"
        else:
            continue
        out.append(
            Violation(
                file=path,
                line=line_no,
                col=comment.col + 1,
                rule="MODULE-NOQA",
                message=message,
                context=text,
            )
        )
    return out


def _split_module_noqa(comment: str) -> tuple[set[str], str]:
    """Split a file-level suppression into ``(rule codes, stated reason)``.

    ``# ruff: noqa: S608 — SQL is built from table-name constants`` yields
    ``({'S608'}, 'SQL is built from table-name constants')``. Codes are the
    first comma-separated run of rule-shaped words; everything after it is the
    reason, whichever separator the author used (``—``, ``--``, ``#``).
    """
    body = _RUFF_NOQA_PREFIX_RE.sub("", comment).strip()
    if body.startswith(":"):
        body = body[1:]
    codes: set[str] = set()
    reason_parts: list[str] = []
    for chunk in _NOQA_CODE_SEP_RE.split(body):
        token = chunk.strip()
        if not token:
            continue
        if not reason_parts and _RUFF_CODE_RE.match(token):
            codes.add(token)
            continue
        reason_parts.append(token)
    reason = " ".join(reason_parts)
    for separator in ("—", "--", "#"):
        reason = reason.replace(separator, " ")
    return codes, reason.strip()


def scan_reason_code_literals(path: Path, tree: ast.Module, lines: list[str], comments: dict[int, Comment]) -> list[Violation]:
    """Flag a bare-string reason code where the ``ReasonCode`` enum belongs.

    The item path reports every failure with ``extra={"reason_code": ...}``.
    A bare string there is a silent desync: the ledger counts enum members,
    the log carries prose, and nothing notices until a histogram shows two
    spellings of one failure class — exactly what the vocabulary was
    introduced to prevent.

    The position is the signal, not the value. A typo'd reason ("timeouts")
    is the case worth catching, and a string that happens to match a member
    value is no more correct than one that does not, so the rule does not
    compare against the enum: it requires the member form.
    """
    if _is_test_file(path):
        return []
    out: list[Violation] = []

    def _report(node: ast.AST, value: str) -> None:
        lineno = getattr(node, "lineno", 0)
        if _waived(lineno, comments):
            return
        out.append(
            Violation(
                file=path,
                line=lineno,
                col=getattr(node, "col_offset", 0) + 1,
                rule="REASON-CODE-LITERAL",
                message=f'reason_code "{value}" is a bare string — use the ReasonCode member from kompend.core.types',
            )
        )

    def _is_bare_str(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == _REASON_CODE_KEY and _is_bare_str(keyword.value):
                    _report(keyword.value, keyword.value.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    continue
                if isinstance(key, ast.Constant) and key.value == _REASON_CODE_KEY and _is_bare_str(value):
                    _report(value, value.value)
    # ``ast.walk`` is breadth-first, so the keyword form and the dict form
    # arrive out of source order. Sort so the report reads like the file.
    return sorted(out, key=lambda v: (v.line, v.col))


@cache
def _vocab_enum_values() -> dict[str, set[str]]:
    """Map member value → enum class names for ``_VOCAB_ENUM_NAMES``.

    Collected by scanning ``src/kompend`` for ``class X(StrEnum)`` bodies.
    """
    value_map: dict[str, set[str]] = defaultdict(set)
    for py in SRC_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"), filename=str(py))
        except (OSError, SyntaxError) as exc:
            # A file the collector cannot parse is a blind spot in its own
            # fact set: the enum values it defines would silently go missing.
            raise ScanError(f"cannot collect enum values from {py}: {exc}") from exc
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


def scan_noqa_bans(path: Path, comments: dict[int, Comment]) -> list[Violation]:
    """Find ``# noqa: BANNED_NOQA`` comments — they must not exist.

    Reads the tokenizer's comment tokens, so prose in a docstring that quotes
    ``# noqa: PLC0415`` is documentation, not a suppression.
    """
    out: list[Violation] = []
    for line_no, comment in sorted(comments.items()):
        for code in sorted(_noqa_codes(comment.text)):
            if code not in BANNED_NOQA:
                continue
            out.append(
                Violation(
                    file=path,
                    line=line_no,
                    col=comment.col + 1,
                    rule="BANNED-NOQA",
                    message=f"# noqa: {code} is banned. Fix the root issue instead of suppressing the lint.",
                    context=comment.text.strip(),
                )
            )
    return out


def _collect_noqa_codes(line: str) -> set[str]:
    """Extract every rule code named in a ``noqa`` comment (whitespace- or comma-separated)."""
    return _noqa_codes(line)


def scan_import_error_guards(path: Path, tree: ast.AST) -> list[Violation]:
    r"""Flag ``try: import x / except ImportError: ...`` blocks.

    Required dependencies must be imported at module level. The only
    legitimate use is probing an explicitly-optional package (declared
    in pyproject.toml's [project.optional-dependencies]); this scanner
    cannot detect that distinction statically, so the message points
    the developer at the AGENTS.md rule and asks them to confirm.

    We deliberately do NOT flag ``except ImportError`` raised during
    function calls (e.g. ``try: foo(); except ImportError: ...``) —
    only ImportError raised around an ``import`` statement. The import must be
    in the *guarded body* (``body`` or ``orelse``); walking the whole ``Try``
    node would include the handlers, so ``try: risky()\nexcept ImportError:\n
    import fallback`` would be reported as an import guard although the import
    is what the handler does, not what it protects.
    """
    out: list[Violation] = []
    if _is_test_file(path):
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(isinstance(handler.type, ast.Name) and handler.type.id == "ImportError" for handler in node.handlers):
            continue
        guarded = [*node.body, *node.orelse]
        if not any(isinstance(stmt, (ast.Import, ast.ImportFrom)) for stmt in guarded):
            continue
        handler = next(h for h in node.handlers if isinstance(h.type, ast.Name) and h.type.id == "ImportError")
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


def scan_magic_crops(path: Path, lines: list[str], comments: dict[int, Comment]) -> list[Violation]:
    """Flag ``[:N]`` crops that are not data-driven.

    A slice is considered data-driven if the same source line also calls
    ``config.get(...)`` / ``DEFAULTS.get(...)`` / similar. This catches
    the most common case (the limit is derived from config) but accepts
    the legitimate patterns:
    - hash truncation: ``hexdigest()[:16]`` is intentional and safe.
    - data-driven slices: lines that compute the limit from config.
    - named constants: a bound that is a module constant (``title[:MAX]``)
      is not matched by the scanner — the constant is the traceable form.

    *lines* are code only (comments stripped by the tokenizer), so a ``#``
    inside a string cannot hide a crop and a crop inside a comment cannot
    raise one. Any ``noqa`` on the line — with or without codes — opts out;
    the author then owns the justification.
    """
    if _is_test_file(path):
        return []
    out: list[Violation] = []
    for i, code in enumerate(lines, start=1):
        if i in comments and _NOQA_RE.match(comments[i].text.strip()):
            continue  # author already opted out (bare noqa is enough here)
        if _is_blank(code):
            continue
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
                        "or use a named constant. Never crop (AGENTS.md, .agents/CODE.md)."
                    ),
                    context=code.strip(),
                )
            )
    return out


def _split_comments(source: str) -> tuple[list[str], dict[int, Comment]]:
    """Split *source* into code-only lines and a line→comment map.

    Uses :mod:`tokenize`, so a ``#`` inside a string literal or a docstring is
    code, not a comment — ``line.split("#", 1)[0]`` silently discards real
    code such as ``tag = "#tag"[:5]``, and a docstring line that quotes
    ``# noqa: PLC0415`` is not a suppression.

    String literal text is blanked for the same reason: ``x = "text[:120]"`` is
    data, not a crop. F-string literal parts are blanked too, while the
    expressions inside them stay visible — ``f"{v[:3]}"`` is still a crop.

    Falls back to a plain split when the source cannot be tokenized (a file
    whose syntax is not valid Python); the caller reports that case separately.
    """
    lines = source.splitlines()
    code_lines = list(lines)
    comments: dict[int, Comment] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                row, col = token.start
                if 0 < row <= len(code_lines):
                    code_lines[row - 1] = code_lines[row - 1][:col]
                comments[row] = Comment(col=col, text=token.string)
            elif token.type in _STRING_TOKEN_TYPES:
                _blank_span(code_lines, token.start, token.end)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return list(lines), {}
    return code_lines, comments


def _blank_span(code_lines: list[str], start: tuple[int, int], end: tuple[int, int]) -> None:
    """Blank a token's columns in *code_lines*, keeping later columns aligned."""
    srow, scol = start
    erow, ecol = end
    if srow == erow:
        if 0 < srow <= len(code_lines):
            line = code_lines[srow - 1]
            code_lines[srow - 1] = line[:scol] + " " * max(0, ecol - scol) + line[ecol:]
        return
    for row in range(srow, erow + 1):
        if not (0 < row <= len(code_lines)):
            continue
        line = code_lines[row - 1]
        code_lines[row - 1] = line[:scol] if row == srow else (" " * ecol + line[ecol:] if row == erow else "")


def _noqa_codes(comment: str) -> set[str]:
    """The rule codes a single ``noqa`` comment names (empty for a blanket)."""
    match = _NOQA_RE.match(comment.strip())
    if not match:
        return set()
    raw = match.group(1) or ""
    return {code for code in _NOQA_CODE_SEP_RE.split(raw.strip()) if code}


def _waived(line_no: int, comments: dict[int, Comment]) -> bool:
    """Whether the author waived this line with a ``# nocheck:`` comment."""
    comment = comments.get(line_no)
    return comment is not None and _NO_CHECK_RE.search(comment.text) is not None


def _is_blank(code: str) -> bool:
    """True when nothing but whitespace is left of a code line."""
    return not code.strip()


def _is_test_file(path: Path) -> bool:
    return TESTS_ROOT in path.parents or path.name.startswith("test_")


def discover_py_files(roots: list[Path]) -> list[Path]:
    """Every ``*.py`` under *roots*; a root that does not exist aborts the scan.

    A typo in a repeated ``--src`` would otherwise shrink the scanned set
    silently — the run would look clean while covering less than the caller
    believes it does. A missing root is a misconfiguration, not an empty
    directory.
    """
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"ERROR: source root does not exist: {root}", file=sys.stderr)
            msg = f"nothing to scan under {root}"
            raise ScanError(msg)
        if root.is_file():
            out.append(root)
            continue
        out.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


if __name__ == "__main__":
    raise SystemExit(main())
