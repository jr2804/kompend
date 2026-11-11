"""A dataclass default must not name a function or class the formatter may move below it.

``rord run`` (pyreorder) orders module sections as imports, constants, enums,
dataclasses, classes, functions. A dataclass field default is evaluated while
the class body runs, so a default that names a module-level function raises
``NameError`` once the formatter moves that function into the functions section
— and the whole package stops importing. The same holds for a module-level
CLASS: the formatter sorts plain classes into the classes section, which sits
BELOW the dataclasses, so a default that named one raised ``NameError`` after
the formatter had run (the lint package did, and with it every ``kompend``
command). Only names bound by the imports (or by constants, enums and runtime
setup, which sit above the dataclasses) are safe. Use
``field(default_factory=lambda: the_name)`` instead: the factory resolves the
name when an instance is built, long after the module is complete.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "kompend"

_DATACLASS_DECORATORS = frozenset({"dataclass"})

#: Base classes the formatter sorts into the enums section, above the
#: dataclasses — a default may name one safely.
_ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "StrFlag"})


def test_every_module_is_scanned() -> None:
    """The sweep must actually see the tree it guards."""
    modules = _modules()
    assert len(modules) > 200
    assert any(path.name == "context_manager.py" for path in modules)


def test_no_dataclass_default_names_a_module_level_function() -> None:
    """The invariant: the formatter may move any ``def`` below any dataclass."""
    offences = [offence for path in _modules() for offence in violations(path.read_text(encoding="utf-8"), where=str(path.relative_to(ROOT)))]

    assert offences == [], (
        "dataclass defaults must not name module-level functions or classes (rord moves them below dataclasses, "
        f"so the class body would raise NameError); use field(default_factory=lambda: f) — {offences}"
    )


def _modules() -> list[Path]:
    """Every module under ``src/kompend``."""
    return sorted(SRC.rglob("*.py"))


def test_the_guard_catches_the_shape_it_exists_for() -> None:
    """The scan is only worth running if it flags the original module layout."""
    broken = """
from dataclasses import dataclass

def count(text: str) -> int:
    return len(text)

@dataclass
class Strategy:
    counter: Callable[[str], int] = count
"""
    assert violations(broken) == ["source:9: Strategy.counter"]

    field_default = broken.replace("= count", "= field(default=count)")
    assert violations(field_default) == ["source:9: Strategy.counter"]

    class_default = """
from dataclasses import dataclass, field

@dataclass
class Strategy:
    counter: Callable[[str], int] = count

def count(text: str) -> int:
    return len(text)
"""
    assert violations(class_default) == ["source:6: Strategy.counter"]


def test_the_guard_catches_a_class_the_formatter_moves_below() -> None:
    """A plain class lands in the classes section, below the dataclasses."""
    source = """
from dataclasses import dataclass, field
from typing import TypedDict

@dataclass
class Finding:
    details: Details = field(default_factory=Details)

class Details(TypedDict, total=False):
    path: str
"""
    assert violations(source) == ["source:7: Finding.details"]


@pytest.mark.parametrize(
    "source",
    [
        # A factory defers the lookup past the class body.
        "@dataclass\nclass A:\n    counter: int = field(default_factory=lambda: count)\n",
        # A constant sits above the dataclasses, so it is safe.
        "COUNT = 4\n\n@dataclass\nclass A:\n    counter: int = COUNT\n",
        # Something imported is safe too.
        "from itertools import count\n\n@dataclass\nclass A:\n    counter: int = count\n",
        # Only dataclasses are affected; a plain class body evaluates in order.
        "def count(text: str) -> int:\n    return len(text)\n\nclass A:\n    counter = count\n",
        # An enum sits above the dataclasses, so naming one is safe.
        "from enum import StrEnum\n\nclass Level(StrEnum):\n    LOW = 'low'\n\n@dataclass\nclass A:\n    level: Level = Level.LOW\n",
        # A deferred factory is safe for a class too.
        "class Details:\n    pass\n\n@dataclass\nclass A:\n    details: Details = field(default_factory=lambda: Details())\n",
    ],
)
def test_safe_shapes_are_not_flagged(source: str) -> None:
    """Deferring the lookup, or naming something that is not a function, is fine."""
    assert violations("from __future__ import annotations\n\n" + source) == []


def violations(source: str, *, where: str = "source") -> list[str]:
    """Every dataclass field default in *source* that names a module-level name."""
    tree = ast.parse(source)
    functions = _module_functions(tree)
    classes = _module_classes(tree)
    found: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not _is_dataclass(node):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                continue
            if statement.value is None:
                continue
            if _names_module_level(statement.value, functions | classes):
                found.append(f"{where}:{statement.lineno}: {node.name}.{statement.target.id}")
    return found


def _module_functions(tree: ast.Module) -> set[str]:
    """Names bound by a ``def`` at module level (the functions section)."""
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}


def _module_classes(tree: ast.Module) -> set[str]:
    """Names bound by a module-level class the formatter moves below the dataclasses.

    Enums are excluded: the formatter sorts them into the enums section, which
    sits above the dataclasses, so a default may name one.
    """
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if any(_base_name(base) in _ENUM_BASES for base in node.bases):
            continue
        names.add(node.name)
    return names


def _base_name(base: ast.expr) -> str:
    """The bare name of a base class (``enum.StrEnum`` reads as ``StrEnum``)."""
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ""


def _names_module_level(expression: ast.expr, names: set[str]) -> bool:
    """True when *expression* evaluates a module-level name at class time."""
    if isinstance(expression, ast.Name):
        return expression.id in names
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == "field":
        return any(keyword.arg in {"default", "default_factory"} and _names_module_level(keyword.value, names) for keyword in expression.keywords)
    return False


def _is_dataclass(node: ast.ClassDef) -> bool:
    """True for ``@dataclass`` and any qualified or called form of it."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in _DATACLASS_DECORATORS:
            return True
    return False
