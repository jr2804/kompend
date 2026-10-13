#!/usr/bin/env python3
"""Detect bare-string literals used as cross-file contract keys.

Enforces src/kompend/AGENTS.md §Antipatterns #5 (and the older Antipattern 5
documentation in the root DOX): field-name strings that have a canonical
StrEnum member must reference that member, not be re-typed by hand.

The checker imports the configured StrEnums, collects their member *values*,
then scans every ``.py`` file under ``src/kompend/`` for bare string literals
matching those values when used in a contract position:

  - as a dict literal key:        ``{"source": x}``
  - as a ``.get()`` argument:     ``d.get("source")``
  - as a subscript argument:      ``d["source"]``

A hit is a violation unless the file is allowlisted (the defining module, or
a file that legitimately parses an external schema).

Exits non-zero on any violation. Designed for ``mise run check-field-consistency``
or direct CI invocation.

Usage::

    uv run python scripts/checks/check_field_consistency.py
    uv run python scripts/checks/check_field_consistency.py --enum MetadataField
    uv run python scripts/checks/check_field_consistency.py --json
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from _common import repo_root, require_scanned

REPO_ROOT = repo_root()
SRC_ROOT = REPO_ROOT / "src" / "kompend"

# Each enforced enum is scoped to the modules that consume its contract.
# A value-only match is a violation ONLY if the hit is in one of the enum's
# consumer modules. Cross-cutting enums (MetadataField, WikiField) apply
# repo-wide; module-local schema enums (ZvecField) apply only to their own
# module + direct readers — enforcing them project-wide would drown in false
# positives on generic words like "content", "title", "id".
#
# Entry shape: ``(module_name, class_name, consumer_glob_patterns)``.
# ``consumer_glob_patterns`` is matched as POSIX-style suffix; an empty list
# means "everywhere except the defining module and the global allowlist".
_IMPORT_ALL: list[str] = []  # sentinel: apply repo-wide

ENFORCED_ENUMS: list[tuple[str, str, list[str]]] = [
    # Cross-cutting node-metadata / wiki-serialization keys.
    ("kompend.core.models", "MetadataField", _IMPORT_ALL),
    ("kompend.core.models", "WikiField", _IMPORT_ALL),
    # NOTE: ElementType and ChunkStrategy are tag VALUES, not field names.
    # They are out of scope here — enforcing them would mean flagging every
    # literal "text"/"table"/"code" as a string, which is noise. They belong
    # to a separate element-type-tagger concern.
    # Memory store contract — written and read only by memory subsystem.
    (
        "kompend.memory.store",
        "MemoryField",
        ["memory/store.py", "memory/__init__.py", "cli/cmds/memory.py"],
    ),
    # zvec collection schema — module-local to zvec_store.py (the DDL contract).
    # Retrievers read hit dicts but also build unrelated dicts with "title" keys,
    # so consumer files are not policed.
    (
        "kompend.rag.zvec_store",
        "ZvecField",
        ["rag/zvec_store.py", "rag/__init__.py"],
    ),
]

# Keyword-argument contracts: keyword name -> enum class name. A bare string
# literal passed as this keyword is a violation regardless of value — it must
# reference the enum member (e.g. ``role=LLMRole.COMPILE_EXTRACT``, never
# ``role="compile.extract"``). Unlike field-name contracts this is
# name-based, so generic English words cause no false positives. These are
# NOT added to ENFORCED_ENUMS to avoid repo-wide value-matching false
# positives on common words ("default", "doctor", "verify", …).
KEYWORD_ARG_ENUMS: dict[str, str] = {
    "role": "LLMRole",
    "task": "AgentTask",
    "compiler": "Compiler",
    "start_at_tier": "Tier",
    "stop_at_tier": "Tier",
}

# Global allowlist — files where bare literals are legitimate regardless of
# which enum matched: the enum-defining modules themselves, TOML-schema dicts
# (keys ARE the documented config schema), LLM-output parsers (keys must match
# the prompt JSON), and the MCP JSON-RPC protocol layer.
ALLOWLIST_SUFFIXES: list[str] = [
    "core/config/_core.py",
    "core/config/_configs.py",
    "core/config/keys.py",
    "compilation/_two_phase_core.py",  # parses extraction JSON from LLM
    "mcp/server.py",  # JSON-RPC protocol keys
    "llm/__init__.py",  # lazy_loader submod_attrs table: keys are module names
]

# Known-generic enum values to skip even outside allowlisted files.
# These collide with ordinary English words too often to enforce reliably.
VALUE_DENYLIST: frozenset[str] = frozenset(
    {
        # Overloaded across unrelated contracts in this codebase:
        #   "source" = metadata-source OR edge-dict source-node OR query result
        #   "metadata" = CompiledWiki container OR parser-config OR node-meta
        # Policing the bare literal catches too many unrelated contracts.
        # The enum members remain available for explicit use at known sites.
        "source",
        "metadata",
        # ElementType / common words
        "text",
        "code",
        "image",
        "table",
        "heading",
        "formula",
        "list_item",
        # ChunkStrategy
        "single_pass",
        "two_level",
        "multi_resolution",
    }
)


@dataclass
class Violation:
    """One bare-literal contract-key use."""

    file: Path
    line: int
    col: int
    value: str
    enum: str  # e.g. "MetadataField.SOURCE"
    context: str  # the source line, trimmed

    def format(self) -> str:
        rel = self.file.relative_to(REPO_ROOT)
        return f'  {rel}:{self.line}:{self.col}  "{self.value}" -> {self.enum}\n    {self.context}'


@dataclass
class EnumEntry:
    """One enforced enum + the consumer modules its values are policed in."""

    module_name: str
    class_name: str
    consumers: list[str]  # POSIX suffixes; empty = repo-wide
    values: dict[str, str] = field(default_factory=dict)  # value -> qualified

    def applies_to(self, path: Path) -> bool:
        posix = path.as_posix()
        # The defining module always excluded (handled by caller).
        if posix.endswith(self.module_name.replace(".", "/") + ".py"):
            return False
        if self.consumers:
            return any(posix.endswith(c) for c in self.consumers)
        return True  # repo-wide

    def qualified(self, value: str) -> str | None:
        if value in VALUE_DENYLIST:
            return None
        return self.values.get(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enum",
        action="append",
        default=[],
        help="Restrict to a specific enum class name (repeatable).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--src", default=str(SRC_ROOT), help="Source root to scan.")
    args = parser.parse_args()

    # Make kompend importable for enum discovery.
    sys.path.insert(0, str(REPO_ROOT / "src"))

    enums = ENFORCED_ENUMS
    if args.enum:
        wanted = set(args.enum)
        enums = [e for e in ENFORCED_ENUMS if e[1] in wanted]
        if not enums:
            print(f"ERROR: no enforced enums match {wanted}", file=sys.stderr)
            return 2

    entries = load_enums(enums)
    total_values = sum(len(e.values) for e in entries)
    if not total_values:
        print("ERROR: no enum members loaded", file=sys.stderr)
        return 2

    root = Path(args.src)
    py_files = discover_py_files(root)
    require_scanned(len(py_files), [root])

    all_violations: list[Violation] = []
    for f in py_files:
        all_violations.extend(scan_file(f, entries))

    if args.json:
        payload = [
            {
                "file": str(v.file.relative_to(REPO_ROOT)),
                "line": v.line,
                "col": v.col,
                "value": v.value,
                "enum": v.enum,
            }
            for v in all_violations
        ]
        print(json.dumps(payload, indent=2))
    else:
        # Group by file for readability.
        by_file: dict[Path, list[Violation]] = defaultdict(list)
        for v in all_violations:
            by_file[v.file].append(v)
        print(f"check-field-consistency: {len(all_violations)} violation(s) in {len(by_file)} file(s), {len(py_files)} file(s) scanned")
        for f in sorted(by_file):
            print(f"\n{f.relative_to(REPO_ROOT)}:")
            for v in sorted(by_file[f], key=lambda x: (x.line, x.col)):
                print(v.format())

    return 1 if all_violations else 0


def load_enums(enums: list[tuple[str, str, list[str]]]) -> list[EnumEntry]:
    """Import each StrEnum and register its member values.

    An entry that cannot be loaded is fatal: the registry is a claim about
    what the gate enforces, so a module that moved (or a class that was
    renamed) must fail the gate rather than quietly drop that contract.
    """
    out: list[EnumEntry] = []
    broken: list[str] = []
    for module_name, class_name, consumers in enums:
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            broken.append(f"{module_name}: import failed ({exc})")
            continue
        cls = getattr(mod, class_name, None)
        if cls is None:
            broken.append(f"{module_name}.{class_name}: class not found")
            continue
        entry = EnumEntry(module_name, class_name, consumers)
        for member in cls:
            entry.values[str(member.value)] = f"{class_name}.{member.name}"
        out.append(entry)
    if broken:
        for problem in broken:
            print(f"ERROR: enforced enum unavailable — {problem}", file=sys.stderr)
        print(
            "ERROR: fix ENFORCED_ENUMS in this script (the contract it names no longer exists)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return out


def scan_file(path: Path, entries: list[EnumEntry]) -> list[Violation]:
    """Return all contract-key violations in *path*."""
    if is_globally_allowlisted(path):
        return []
    # Which enums apply to this file (per their consumer scoping)?
    applicable = [e for e in entries if e.applies_to(path)]
    # Keyword-arg contracts (e.g. role="...") apply repo-wide, so we still
    # scan even when no field-name enum is applicable to this file.
    if not applicable and not KEYWORD_ARG_ENUMS:
        return []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    out: list[Violation] = []
    lines = source.splitlines()

    def matches(val: str) -> list[str]:
        """Return qualified enum names whose member value equals *val*."""
        found: list[str] = []
        for e in applicable:
            q = e.qualified(val)
            if q is not None:
                found.append(q)
        return found

    for node in ast.walk(tree):
        # Case 1: dict literal key — {"source": x}
        if isinstance(node, ast.Dict):
            for key_node in node.keys:
                if key_node is None:  # ** unpacking
                    continue
                if _line_has_nocheck(key_node.lineno, lines):
                    continue
                val = _string_node_value(key_node)
                if val is not None:
                    for q in matches(val):
                        out.append(_make_violation(path, key_node, val, q, lines))
        # Case 2: subscript with string arg — d["source"]
        elif isinstance(node, ast.Subscript):
            if _line_has_nocheck(node.lineno, lines):
                continue
            val = _string_node_value(node.slice)
            if val is not None:
                for q in matches(val):
                    out.append(_make_violation(path, node, val, q, lines))
        # Case 3: .get("source", ...) call
        elif isinstance(node, ast.Call):
            if _line_has_nocheck(node.lineno, lines):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "get" and node.args:
                val = _string_node_value(node.args[0])
                if val is not None:
                    for q in matches(val):
                        out.append(_make_violation(path, node, val, q, lines))
            # Case 4: keyword-argument contract (e.g. role="...") — any bare
            # string literal is wrong; it must reference the enum member.
            for kw in node.keywords:
                if kw.arg in KEYWORD_ARG_ENUMS and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    if _line_has_nocheck(kw.value.lineno, lines):
                        continue
                    enum_cls = KEYWORD_ARG_ENUMS[kw.arg]
                    val = kw.value.value
                    out.append(_make_violation(path, kw.value, val, f"{enum_cls}.<{val}>", lines))

    return out


def is_globally_allowlisted(path: Path) -> bool:
    posix = path.as_posix()
    return any(posix.endswith(suf) for suf in ALLOWLIST_SUFFIXES)


def _string_node_value(node: ast.AST) -> str | None:
    """Return the literal string value of *node*, or None if not a constant str."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _line_has_nocheck(line_no: int, lines: list[str]) -> bool:
    """True if the source line carries a ``# nocheck`` comment.

    Used to exempt lines that legitimately parse an external schema
    (LLM JSON output, checkpoint files, MCP protocol payloads).
    """
    if 1 <= line_no <= len(lines):
        return "# nocheck" in lines[line_no - 1]
    return False


def _make_violation(path: Path, node: ast.AST, value: str, qualified: str, lines: list[str]) -> Violation:
    line = node.lineno
    ctx = lines[line - 1].strip() if 1 <= line <= len(lines) else ""
    return Violation(
        file=path,
        line=line,
        col=node.col_offset + 1,
        value=value,
        enum=qualified,
        context=ctx,
    )


def discover_py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


if __name__ == "__main__":
    raise SystemExit(main())
