"""The lint step vocabulary is one closed set (knox-dw1f).

``run_full_lint_flow`` plans its steps from a list of ``{"name", "weight",
"items"}`` dicts and dispatches on the name. The same strings also mark the
history row's ``context``, so a typo in one place files a run under a step that
never ran. :class:`LintStep` owns them; these tests pin the wire values, pin the
planned order, and fail when a dispatch site or a history context spells one
again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kompend.lint.flow import LintStep

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "kompend"
#: Modules that both consume the step vocabulary and record history rows.
_CONSUMER_FILES = (
    _SOURCE_ROOT / "lint" / "flow.py",
    _SOURCE_ROOT / "cli" / "cmds" / "quality" / "lint" / "run.py",
    _SOURCE_ROOT / "mcp" / "tools" / "quality" / "lint.py",
)


def test_the_wire_values_are_the_planned_step_names() -> None:
    assert {member.value for member in LintStep} == {"check", "format", "semantic", "gaps", "remediate", "all"}


@pytest.mark.parametrize("path", _CONSUMER_FILES, ids=lambda path: path.name)
def test_no_consumer_spells_a_step_as_a_string(path: Path) -> None:
    """A literal step name is how a history row ends up under a step that never ran."""
    offenders = [f"{path.name}:{line} uses {value!r}" for line, value in _step_literals(path)]

    assert offenders == [], "use LintStep instead of the literal:\n" + "\n".join(offenders)


def _step_literals(path: Path) -> list[tuple[int, str]]:
    """String literals equal to an enum value, outside the enum's own body."""
    steps = {member.value for member in LintStep if member is not LintStep.ALL}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    inside_enum: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LintStep":
            inside_enum.update(id(child) for child in ast.walk(node))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in inside_enum and node.value in steps
    ]
