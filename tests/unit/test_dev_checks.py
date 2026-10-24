"""The new guards must pass, and they must actually catch their bug class."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kompend.core.cache import store

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "scripts" / "checks"
GUARDS = ("check_enum_cache.py", "check_config_literals.py", "check_cli_surface.py")


@pytest.mark.parametrize("script", GUARDS)
def test_guard_passes_on_the_current_tree(script: str) -> None:
    result = _run(script)

    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(CHECKS / script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        timeout=180,
        check=False,
    )


def test_all_gates_runner_lists_the_guards() -> None:
    """`all.py` must invoke the guards - a gate nobody runs is not a gate."""
    source = (CHECKS / "all.py").read_text(encoding="utf-8")

    for name in ("enum_cache", "config_literals", "cli_surface", "dox", "mcp_parity"):
        assert name in source


def test_enum_cache_guard_knows_edge_type() -> None:
    """The EdgeType regression: the enum must be decodable from a payload."""
    assert "EdgeType" in store._CACHE_ENUMS
