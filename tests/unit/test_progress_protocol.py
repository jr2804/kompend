"""The progress protocol's ``detail`` vocabulary is one closed set (knox-dw1f).

A progress callback receives ``detail`` as a free-form string, because every
ordinary event puts the unit being worked on there — a source file name, a
tree label, ``asset: <name>``, ``embedding <name>``. Four values are not free
form: they are sentinels that open, close, and sub-divide a phase, and a
consumer keys on them. They live in :class:`ProgressDetail`; these tests pin
the wire values, the classification they drive, and the fact that every
emitter uses the member instead of spelling the string again.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from kompend.core.pipeline.events import (
    PipelineEvent,
    PipelineEventKind,
    ProgressDetail,
)

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "kompend"
_RESERVED = frozenset(member.value for member in ProgressDetail)
_EMITTERS = frozenset({"emit", "emit_item_start", "progress_fn"})


def test_the_wire_values_are_the_published_sentinels() -> None:
    assert {member.value for member in ProgressDetail} == {"starting", "done", "vlm", "captioning"}


def test_starting_opens_a_phase_and_done_closes_it() -> None:
    assert PipelineEvent.from_emit("parse", 0, 3, ProgressDetail.STARTING).kind is PipelineEventKind.STARTED
    assert PipelineEvent.from_emit("parse", 3, 3, ProgressDetail.DONE).kind is PipelineEventKind.COMPLETED
    assert PipelineEvent.from_emit("resolve", 0, 4, ProgressDetail.VLM).kind is PipelineEventKind.PROGRESS


def test_the_sentinels_serialize_as_plain_strings() -> None:
    # ``detail`` is annotated ``str``; a StrEnum member is one, so the JSONL
    # record and every callback comparison see the same text.
    record = PipelineEvent.from_emit("summary", 2, 2, ProgressDetail.DONE).to_dict()

    assert record["detail"] == "done"
    assert json.dumps(record, sort_keys=True).count('"detail": "done"') == 1
    assert isinstance(record["detail"], str)


def test_a_unit_label_is_not_a_sentinel() -> None:
    """The closed set must not swallow ordinary details."""
    assert "doc17.pdf" not in _RESERVED
    assert "asset: figure.png" not in _RESERVED
    assert PipelineEvent.from_emit("parse", 1, 3, "doc17.pdf").kind is PipelineEventKind.PROGRESS


def _emitter_detail_nodes(tree: ast.AST) -> list[ast.expr]:
    """Every ``detail`` argument passed to an emitter call, positionally or by keyword."""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in _EMITTERS:
            continue
        for keyword in node.keywords:
            if keyword.arg == "detail":
                found.append(keyword.value)
        if len(node.args) >= 4:  # emit(phase, current, total, detail)
            found.append(node.args[3])
        elif name == "progress_fn" and len(node.args) >= 3:  # progress_fn(current, total, detail)
            found.append(node.args[2])
    return found


def test_no_emitter_spells_a_sentinel_as_a_string() -> None:
    """A literal sentinel at an emit site is how the vocabulary drifts apart."""
    offenders: list[str] = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _emitter_detail_nodes(tree):
            if isinstance(node, ast.Constant) and node.value in _RESERVED:
                relative = path.relative_to(_SOURCE_ROOT.parents[1]).as_posix()
                offenders.append(f"{relative}:{node.lineno} passes {node.value!r}")

    assert offenders == [], "use ProgressDetail instead of the literal:\n" + "\n".join(offenders)
