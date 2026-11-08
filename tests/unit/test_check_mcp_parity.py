"""Contract test for the MCP tool-schema snapshot gate.

The gate lives in ``scripts/checks/check_mcp_parity.py``: it owns the tool
*names* and, through ``scripts/checks/snapshots/mcp_tool_surface.json``, the
tool *shapes*. These tests prove the snapshot matches the live server and that
a renamed parameter is reported as a violation instead of passing silently.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_CHECKS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "checks"


def test_snapshot_covers_exactly_the_quality_tool_surface(gate: Any) -> None:
    snapshot = gate.load_snapshot()
    assert set(snapshot["tools"]) == gate.SNAPSHOT_TOOL_NAMES


def test_live_surface_matches_the_committed_snapshot(gate: Any) -> None:
    assert gate.compare_snapshots(gate.load_snapshot(), gate.build_snapshot()) == []


def test_a_renamed_parameter_is_a_violation(gate: Any) -> None:
    snapshot = copy.deepcopy(gate.load_snapshot())
    properties = snapshot["tools"]["doctor"]["input_schema"]["properties"]
    properties["wiki_directory"] = properties.pop("wiki_dir")

    findings = gate.compare_snapshots(snapshot, gate.build_snapshot())

    assert findings, "a renamed input parameter must fail the schema gate"
    assert any("doctor" in finding for finding in findings)


def test_a_missing_snapshot_cannot_judge(gate: Any, tmp_path: Path) -> None:
    assert gate.main(["--snapshot", str(tmp_path / "absent.json")]) == 2


@pytest.fixture(scope="module")
def gate() -> Any:
    """Load the gate script without making ``scripts/checks`` a package."""
    sys.path.insert(0, str(_CHECKS_DIR))
    spec = importlib.util.spec_from_file_location("check_mcp_parity_under_test", _CHECKS_DIR / "check_mcp_parity.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
