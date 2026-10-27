#!/usr/bin/env python
"""A value that belongs to the config must not be re-typed in code.

Two rules, both born from real regressions:

1. **A config default is never repeated.** A keyword argument (or annotated
   assignment) whose *name* matches a config field and whose literal value
   equals that field's default is a stale copy waiting to drift - the
   summariser had a hardcoded `window_chars=2000`, and the agent base had a
   `max_tokens` default that shadowed the budget.
2. **A dotted config key is never a string literal.** Use the constant from
   `kompend.core.keys`; a typo there fails silently at runtime.

Mark a deliberate exception with ``# nocheck`` and say why.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

from _common import repo_root

from kompend.core import keys as key_constants
from kompend.core.config._configs import (
    BudgetConfig,
    DisplayConfig,
    LintConfig,
    LLMConfig,
    OrganizeConfig,
    SearchSynthesisConfig,
)

CONFIG_MODELS = (
    BudgetConfig,
    DisplayConfig,
    LintConfig,
    LLMConfig,
    OrganizeConfig,
    SearchSynthesisConfig,
)

#: Config sections whose keys are built or consumed there by definition, plus
#: the module that defines the dotted keys.
EXCLUDED_PARTS: tuple[str, ...] = ("core/config", "core/keys.py")


def main() -> int:
    files = _source_files()
    if not files:
        print("ERROR: no source files found - refusing to report success", file=sys.stderr)
        return 2

    defaults = _config_defaults()
    dotted = _dotted_keys()
    findings: list[str] = []
    for path in files:
        findings.extend(_scan(path, defaults, dotted))

    print(f"check-config-literals: {len(files)} file(s) scanned against {len(defaults)} config defaults and {len(dotted)} dotted keys")
    if findings:
        for finding in findings:
            print(f"  {finding}")
        print(f"\nERROR: {len(findings)} literal(s) duplicate config values.", file=sys.stderr)
        return 1
    return 0


def _config_defaults() -> dict[str, Any]:
    """Field name -> default, for every user-facing config model."""
    defaults: dict[str, Any] = {}
    for model in CONFIG_MODELS:
        for name, info in model.model_fields.items():
            if info.default is None or callable(info.default):
                continue
            defaults[name] = info.default
    return defaults


def _dotted_keys() -> set[str]:
    """Every ``section.key`` string declared in ``core/keys.py``."""
    return {value for value in vars(key_constants).values() if isinstance(value, str) and "." in value}


def _source_files() -> list[Path]:
    root = repo_root() / "src" / "kompend"
    return [path for path in sorted(root.rglob("*.py")) if not any(part in path.as_posix() for part in EXCLUDED_PARTS)]


def _scan(path: Path, defaults: dict[str, Any], dotted: set[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"{path}: cannot parse ({exc})"]

    findings: list[str] = []

    def line_of(node: ast.AST) -> str:
        lineno = getattr(node, "lineno", None)
        return lines[lineno - 1] if lineno and lineno <= len(lines) else ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg not in defaults:
                    continue
                if not isinstance(keyword.value, ast.Constant) or isinstance(keyword.value.value, bool):
                    continue
                if keyword.value.value != defaults[keyword.arg]:
                    continue
                line = line_of(keyword.value)
                if _is_exempt(line):
                    continue
                lineno = getattr(keyword.value, "lineno", 0)
                findings.append(
                    f"{path}:{lineno}: `{keyword.arg}={keyword.value.value!r}` repeats the config default "
                    f"of the same field - pass the config value instead (or mark `# nocheck: reason`)"
                )

        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in dotted:
            line = line_of(node)
            if _is_exempt(line):
                continue
            lineno = getattr(node, "lineno", 0)
            findings.append(f"{path}:{lineno}: dotted key {node.value!r} is a literal - import the constant from kompend.core.keys")

    return findings


def _is_exempt(line: str) -> bool:
    return "nocheck" in line


if __name__ == "__main__":
    sys.exit(main())
