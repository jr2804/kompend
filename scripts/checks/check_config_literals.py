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
from typing import Any, Final

from _common import repo_root, require_facts
from pydantic_core import PydanticUndefined

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


_NO_DEFAULT: Final[object] = object()


def main() -> int:
    files = _source_files()
    if not files:
        print("ERROR: no source files found - refusing to report success", file=sys.stderr)
        return 2

    defaults = _config_defaults()
    dotted = _dotted_keys()
    # Empty facts make every literal look unique, which reads as a clean run.
    try:
        require_facts(len(defaults), "config defaults", "Check CONFIG_MODELS: the config models moved or stopped exporting defaults.")
        require_facts(len(dotted), "dotted config keys", "Check core/keys.py: the dotted-key constants moved or were renamed.")
    except SystemExit as exc:
        return int(exc.code)
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


def _config_defaults() -> dict[str, tuple[Any, ...]]:
    """Field name -> every default any config model declares for it.

    A tuple per name, not a bare value: the six config models are merged into
    one map, and a field name is only unique *within* a model. Keyed by name
    alone, a second model declaring the same field with a different default
    would overwrite the first, and a literal repeating the shadowed default
    would stop being reported - the same silent drop this module exists to
    prevent, one layer down. No name is currently shared, so this is dormant;
    a tuple costs nothing and makes a future share correct instead of lossy.
    """
    defaults: dict[str, tuple[Any, ...]] = {}
    for model in CONFIG_MODELS:
        for name, info in model.model_fields.items():
            value = _comparable_default(info)
            if value is _NO_DEFAULT:
                continue
            defaults[name] = (*defaults.get(name, ()), value)
    return defaults


def _comparable_default(info: Any) -> Any:
    """The default a literal can meaningfully be compared against, if any.

    A field can declare no repeatable default in two ways, and each must be
    excluded rather than stored:

    - ``None`` - "unset" is a runtime state, not a value to repeat.
    - ``PydanticUndefined`` - the sentinel ``Field(default_factory=...)`` leaves
      on ``FieldInfo.default``. It is neither ``None`` nor callable, so an
      earlier version stored it as if it were a value, and every comparison
      against it failed silently - a dropped finding. ``LLMConfig
      .fallback_chain`` was already in that state, so the hole was live.

    A ``default_factory`` field is the case that needs care: ``fallback_chain=[]``
    in a call site IS a stale copy of ``Field(default_factory=list)``, because
    the value a caller re-types is the same even though each instance gets a
    fresh object. A factory building an EMPTY collection therefore contributes
    that empty collection as its repeatable default; a factory building
    anything else (a computed size, a timestamp) contributes nothing, since no
    literal equals it.

    Returns the default itself, or :data:`_NO_DEFAULT` when the field declares
    no value a literal can repeat.
    """
    if info.default is None or callable(info.default) or info.default is PydanticUndefined:
        factory = info.default_factory
        if factory is None:
            return _NO_DEFAULT
        # A factory that builds an EMPTY collection declares a default that
        # ``[]``/``{}``/``()`` does repeat: the object is fresh per instance,
        # but the value a caller would re-type is the same. Anything else a
        # factory builds (a computed size, a timestamp) is not a literal.
        built = factory()
        if isinstance(built, (list, dict, set, tuple)) and not built:
            return built
        return _NO_DEFAULT
    return info.default


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
                literal = _literal_value(keyword.value)
                if literal is _NO_DEFAULT or literal not in defaults[keyword.arg]:
                    continue
                line = line_of(keyword.value)
                if _is_exempt(line):
                    continue
                lineno = getattr(keyword.value, "lineno", 0)
                findings.append(
                    f"{path}:{lineno}: `{keyword.arg}={literal!r}` repeats the config default "
                    f"of the same field - pass the config value instead (or mark `# nocheck: reason`)"
                )

        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in dotted:
            line = line_of(node)
            if _is_exempt(line):
                continue
            lineno = getattr(node, "lineno", 0)
            findings.append(f"{path}:{lineno}: dotted key {node.value!r} is a literal - import the constant from kompend.core.keys")

    return findings


def _literal_value(node: ast.AST) -> Any:
    """The value a caller literally wrote, for the forms a config default takes.

    Only literals are considered: a name, a call or an f-string is a computed
    value, not a re-typed default. Empty and scalar ``list``/``dict``/``set``/
    ``tuple`` displays are included because a ``default_factory`` field's
    repeatable default is exactly that empty collection (see
    :func:`_comparable_default`).

    Returns :data:`_NO_DEFAULT` for anything that is not a literal.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or node.value is None:
            return _NO_DEFAULT
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        operand = node.operand.value
        return -operand if isinstance(operand, (int, float)) and not isinstance(operand, bool) else _NO_DEFAULT
    if isinstance(node, ast.List) and not node.elts:
        return []
    if isinstance(node, ast.Dict) and not node.keys:
        return {}
    if isinstance(node, ast.Set) and not node.elts:
        return set()
    if isinstance(node, ast.Tuple) and not node.elts:
        return ()
    return _NO_DEFAULT


def _is_exempt(line: str) -> bool:
    return "nocheck" in line


if __name__ == "__main__":
    sys.exit(main())
