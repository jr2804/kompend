"""The new guards must pass, and they must actually catch their bug class."""

from __future__ import annotations

import ast
import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kompend.core.cache import store

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "scripts" / "checks"
GUARDS = ("check_enum_cache.py", "check_config_literals.py", "check_cli_surface.py")


@pytest.mark.parametrize("script", GUARDS)
def test_guard_passes_on_the_current_tree(script: str) -> None:
    result = _run(script)

    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"


def test_all_gates_runner_lists_the_guards() -> None:
    """`all.py` must invoke the guards - a gate nobody runs is not a gate."""
    source = (CHECKS / "all.py").read_text(encoding="utf-8")

    for name in ("enum_cache", "config_literals", "cli_surface", "dox", "mcp_parity"):
        assert name in source


def test_enum_cache_guard_knows_edge_type() -> None:
    """The EdgeType regression: the enum must be decodable from a payload."""
    assert "EdgeType" in store._CACHE_ENUMS


def test_pip_install_guard_rejects_a_comment() -> None:
    """The guard must fail on an install instruction that bypasses uv."""
    sample_dir = ROOT / "tmp" / "check-antipatterns-pip"
    sample_dir.mkdir(parents=True, exist_ok=True)
    forbidden = " ".join(("uv", "pip", "install"))
    (sample_dir / "sample.py").write_text(f"VALUE = 1  # install with {forbidden} foo\n", encoding="utf-8")
    try:
        result = _run("check_antipatterns.py", "--src", str(sample_dir))

        assert result.returncode == 1
        assert "PIP-INSTALL" in result.stdout
    finally:
        shutil.rmtree(sample_dir, ignore_errors=True)


def test_noqa_scan_finds_every_spelling_with_a_real_column() -> None:
    """``#noqa`` / ``#  noqa`` are suppressions too, and the column is real."""
    module = _antipatterns()

    for line in ("x = 1  # noqa: PLC0415", "x = 1  #noqa: PLC0415", "x = 1  #  noqa: PLC0415"):
        _, comments = module._split_comments(line)
        found = module.scan_noqa_bans(Path("a.py"), comments)

        assert [v.line for v in found] == [1], line
        assert found[0].col == 8, f"column must point at the '#', not -1 (got {found[0].col})"


def test_noqa_scan_ignores_prose_and_strings() -> None:
    """A docstring quoting the rule is documentation, not a suppression."""
    module = _antipatterns()
    docstring = 'def f() -> None:\n    """Use `# noqa: PLC0415` as a last resort."""\n'

    _, comments = module._split_comments(docstring)

    assert module.scan_noqa_bans(Path("a.py"), comments) == []


def test_noqa_codes_split_on_whitespace_as_ruff_does() -> None:
    """Ruff accepts space-separated codes, so the ban must see them."""
    module = _antipatterns()

    assert module._collect_noqa_codes("# noqa: PLC0415 B008") == {"PLC0415", "B008"}
    assert module._collect_noqa_codes("# noqa: PLC0415, B008") == {"PLC0415", "B008"}


def test_magic_crops_honours_a_bare_noqa_and_ignores_string_data() -> None:
    """A crop in a string is data; a bare ``# noqa`` is a valid opt-out."""
    module = _antipatterns()

    def crops(source: str) -> int:
        code, comments = module._split_comments(source)
        return len(module.scan_magic_crops(Path("a.py"), code, comments))

    assert crops("title = text[:120]  # noqa") == 0
    assert crops('x = "text[:120]"') == 0
    assert crops('x = f"{value[:3]}"') == 1, "an f-string expression is still code"
    assert crops("title = text[:120]") == 1


def test_import_error_guard_ignores_an_import_inside_the_handler() -> None:
    """Only an import in the guarded body makes it an import guard."""
    module = _antipatterns()

    handler_only = ast.parse("try:\n    risky()\nexcept ImportError:\n    import json\n")
    real_guard = ast.parse("try:\n    import optional_pkg\nexcept ImportError:\n    optional_pkg = None\n")

    assert module.scan_import_error_guards(Path("a.py"), handler_only) == []
    assert len(module.scan_import_error_guards(Path("a.py"), real_guard)) == 1


def test_an_empty_source_root_is_not_a_pass(tmp_path: Path) -> None:
    """Scanning zero Python files must abort, not report success."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _run("check_antipatterns.py", "--src", str(empty))

    assert result.returncode == 2, f"vacuous scan must fail loudly:\n{result.stdout}\n{result.stderr}"
    assert "nothing to scan" in result.stderr


def test_an_unparseable_file_aborts_the_scan(tmp_path: Path) -> None:
    """A file the gate cannot parse is a blind spot, not a clean bill."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "bad.py").write_text("def f(:\n", encoding="utf-8")

    result = _run("check_antipatterns.py", "--src", str(broken))

    assert result.returncode == 2
    assert "cannot parse" in result.stderr


def test_a_fact_provider_that_cannot_read_aborts_rather_than_emptying(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A rule must not lose its facts silently.

    ``_path_filename_constants`` and ``_vocab_enum_values`` decide their rules
    by reading files outside the one being scanned. Returning an empty set on
    failure reads as "this project has no such constants", which disarms the
    rule without a word — the same class as an empty source root.
    """
    module = _antipatterns()
    missing = tmp_path / "nope.py"

    monkeypatch.setattr(module, "_PATHS_MODULE", missing)
    module._path_filename_constants.cache_clear()
    with pytest.raises(module.ScanError, match="path constants"):
        module._path_filename_constants()
    module._path_filename_constants.cache_clear()

    monkeypatch.setattr(module, "SRC_ROOT", tmp_path)
    module._vocab_enum_values.cache_clear()
    (tmp_path / "bad.py").write_text("class X(StrEnum\\n", encoding="utf-8")
    with pytest.raises(module.ScanError, match="enum values"):
        module._vocab_enum_values()
    module._vocab_enum_values.cache_clear()


# ── check_antipatterns: the scanner must not lie in either direction ────────


def _antipatterns() -> Any:
    """Import the checker module (it lives outside the package tree)."""
    sys.path.insert(0, str(CHECKS))
    try:
        return importlib.import_module("check_antipatterns")
    finally:
        sys.path.pop(0)


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(CHECKS / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        timeout=180,
        check=False,
    )
