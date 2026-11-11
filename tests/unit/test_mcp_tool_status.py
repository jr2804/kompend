"""The MCP tool envelope: one ``status`` vocabulary, unchanged on the wire.

Every tool answers with an envelope whose ``status`` is either ``ok`` or
``error``. The values are the protocol contract an agent reads, so they are
named once in :mod:`kompend.mcp.tools._helpers` and never spelled out at a
call site.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from kompend.mcp.tools._helpers import ToolStatus

TOOLS_DIR = Path(__file__).resolve().parents[2] / "src" / "kompend" / "mcp" / "tools"

_ENVELOPE_VALUES = frozenset({"ok", "error"})


class TestToolStatus:
    """The vocabulary itself."""

    def test_wire_values(self) -> None:
        assert {str(member) for member in ToolStatus} == _ENVELOPE_VALUES
        assert ToolStatus.OK.value == "ok"
        assert ToolStatus.ERROR.value == "error"

    def test_members_are_strings(self) -> None:
        assert isinstance(ToolStatus.OK, str)
        assert json.loads(json.dumps({"status": ToolStatus.OK})) == {"status": "ok"}


class TestEnvelopeLiterals:
    """No module spells an envelope value out."""

    def test_no_bare_status_literals(self) -> None:
        offenders: list[str] = []
        for path in sorted(TOOLS_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or node.value not in _ENVELOPE_VALUES:
                    continue
                if _inside_status_enum(tree, node):
                    continue
                offenders.append(f"{path.relative_to(TOOLS_DIR)}:{node.lineno}: {node.value!r}")
        assert offenders == [], "envelope values must be ToolStatus members:\n" + "\n".join(offenders)

    def test_every_tool_module_imports_the_vocabulary(self) -> None:
        missing = [
            path.name
            for path in sorted(TOOLS_DIR.rglob("*.py"))
            if _envelopes(path) and "ToolStatus" not in path.read_text(encoding="utf-8")
        ]
        assert missing == []


class TestSerializedEnvelope:
    """The wire payload a client sees is byte-for-byte what it was."""

    def test_ok_envelope_serializes_as_before(self) -> None:
        from kompend.mcp.tools.packs import pack_list

        payload = pack_list()
        assert payload["status"] == "ok"
        assert json.loads(json.dumps(payload))["status"] == "ok"

    def test_error_envelope_serializes_as_before(self) -> None:
        from kompend.mcp.tools.config import config_get

        payload = config_get("not.a.key")
        assert payload["status"] == "error"
        assert payload["message"]
        assert json.loads(json.dumps(payload))["status"] == "error"

    @pytest.mark.parametrize("value", [ToolStatus.OK, ToolStatus.ERROR])
    def test_member_serializes_to_its_value(self, value: ToolStatus) -> None:
        assert json.dumps({"status": value}) == f'{{"status": "{value.value}"}}'


def _envelopes(path: Path) -> bool:
    """Whether ``path`` builds a ``status`` envelope at all."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Constant) and node.value == "status"
        for node in ast.walk(tree)
    )


def _inside_status_enum(tree: ast.Module, node: ast.Constant) -> bool:
    """Whether ``node`` is a member value of the ``ToolStatus`` definition."""
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or cls.name != "ToolStatus":
            continue
        if any(child is node for child in ast.walk(cls)):
            return True
    return False
