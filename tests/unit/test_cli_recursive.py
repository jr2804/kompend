"""``paths.recursive`` resolution for every command that discovers source files.

``--recursive/--no-recursive`` is tri-state: unset follows ``paths.recursive``
(config file, ``KOMP_PATHS_RECURSIVE``, or ``--set``), and an explicit flag wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kompend.capture.ingest import ingest
from kompend.cli.app import app
from tests.helpers.cli import CliRunner

_cli_runner = CliRunner()


class _Captured(Exception):
    """Raised by the ``resolve_files`` stub to stop the command early."""


@pytest.mark.parametrize(
    ("config_recursive", "flags", "expected"),
    [
        (True, [], True),
        (False, [], False),
        (None, [], True),
        (False, ["--recursive"], True),
        (True, ["--no-recursive"], False),
        (None, ["--no-recursive"], False),
    ],
)
def test_parse_follows_paths_recursive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_recursive: bool | None,
    flags: list[str],
    expected: bool,
) -> None:
    base = _vault(tmp_path, recursive=config_recursive)
    captured = _capture_resolve(monkeypatch, "kompend.cli.cmds.pipeline.pipeline")

    with pytest.raises(_Captured):
        _cli_runner.invoke(
            app,
            ["--wiki-dir", str(base), "parse", *flags, str(base / "sources")],
            catch_exceptions=False,
        )

    assert _recursive_arg(captured) is expected


@pytest.mark.parametrize(
    ("config_recursive", "flags", "expected"),
    [
        (True, [], True),
        (False, [], False),
        (None, [], True),
        (False, ["--recursive"], True),
        (True, ["--no-recursive"], False),
    ],
)
def test_write_follows_paths_recursive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_recursive: bool | None,
    flags: list[str],
    expected: bool,
) -> None:
    base = _vault(tmp_path, recursive=config_recursive)
    captured = _capture_resolve(monkeypatch, "kompend.cli.cmds.knowledge.capture")

    with pytest.raises(_Captured):
        _cli_runner.invoke(
            app,
            ["--wiki-dir", str(base), "write", "Some Page", *flags, "--source", str(base / "sources")],
            catch_exceptions=False,
        )

    assert _recursive_arg(captured) is expected


def test_write_source_resolution_stays_inside_sources_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, recursive=True)
    captured = _capture_resolve(monkeypatch, "kompend.cli.cmds.knowledge.capture")

    with pytest.raises(_Captured):
        _cli_runner.invoke(
            app,
            ["--wiki-dir", str(base), "write", "Some Page", "--source", "top.md"],
            catch_exceptions=False,
        )

    assert captured["kwargs"]["sources_dir"] == base / "sources"


def _vault(tmp_path: Path, *, recursive: bool | None) -> Path:
    """Build a vault with a nested source; *recursive* seeds ``[paths]``."""
    sources = tmp_path / "sources"
    (sources / "nested").mkdir(parents=True)
    (sources / "top.md").write_text("# Top\n", encoding="utf-8")
    (sources / "nested" / "deep.md").write_text("# Deep\n", encoding="utf-8")
    kompend_dir = tmp_path / ".kompend"
    kompend_dir.mkdir()
    config = "" if recursive is None else f"[paths]\nrecursive = {str(recursive).lower()}\n"
    (kompend_dir / "config.toml").write_text(config, encoding="utf-8")
    return tmp_path


def test_capture_ingest_defaults_to_recursive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_resolve(monkeypatch, "kompend.capture.ingest")

    with pytest.raises(_Captured):
        ingest(str(tmp_path), sources_dir=tmp_path / "sources")

    assert _recursive_arg(captured) is True


def test_capture_ingest_forwards_recursive_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_resolve(monkeypatch, "kompend.capture.ingest")

    with pytest.raises(_Captured):
        ingest(str(tmp_path), sources_dir=tmp_path / "sources", recursive=False)

    assert _recursive_arg(captured) is False


def _capture_resolve(monkeypatch: pytest.MonkeyPatch, module: str) -> dict[str, Any]:
    """Replace ``resolve_files`` in *module* with a recorder that aborts the run."""
    captured: dict[str, Any] = {}

    def _fake(*patterns: str, **kwargs: Any) -> list[Path]:
        captured["patterns"] = patterns
        captured["kwargs"] = kwargs
        raise _Captured

    monkeypatch.setattr(f"{module}.resolve_files", _fake)
    return captured


def _recursive_arg(captured: dict[str, Any]) -> bool:
    """Return the ``recursive`` keyword the command passed to ``resolve_files``."""
    assert captured["kwargs"] is not None
    return captured["kwargs"]["recursive"]
