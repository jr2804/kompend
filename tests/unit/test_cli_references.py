"""Tests for ``knox references`` CLI command (Phase 4).

Covers:
- ``references list`` — listing stored references
- ``references export --format bib`` — BibTeX export
- ``references export --format json`` — CSL JSON export
- No references / empty store handling
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.models.references import Author, Reference
from knox.core.pipeline.references import ReferenceStore


class TestReferencesList:
    """Tests for ``knox references list``."""

    def test_list_all(self, project_with_refs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(project_with_refs)
        runner = CliRunner()
        result = runner.invoke(app, ["references", "list"])
        assert result.exit_code == 0
        assert "Smith 2020" in result.output or "smith2020" in result.output
        assert "Jones 2019" in result.output or "jones2019" in result.output

    def test_list_no_references(self, empty_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(empty_project)
        runner = CliRunner()
        result = runner.invoke(app, ["references", "list"])
        assert result.exit_code == 0
        assert "No references found" in result.output

    def test_list_with_wiki_dir(self, project_with_refs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(project_with_refs)
        runner = CliRunner()
        result = runner.invoke(app, ["--wiki-dir", str(project_with_refs), "references", "list"])
        assert result.exit_code == 0
        assert "Deep Learning" in result.output


class TestReferencesExport:
    """Tests for ``knox references export``."""

    def test_export_bibtex(self, project_with_refs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(project_with_refs)
        runner = CliRunner()
        out_file = project_with_refs / "refs.bib"
        result = runner.invoke(app, ["references", "export", "-o", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "@article{smith2020," in content
        assert "@article{jones2019," in content
        assert "10.1/xyz" in content

    def test_export_json(self, project_with_refs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(project_with_refs)
        runner = CliRunner()
        out_file = project_with_refs / "refs.json"
        result = runner.invoke(app, ["references", "export", "-o", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 2
        assert any(d.get("id") == "smith2020" for d in data)

    def test_export_no_references(self, empty_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(empty_project)
        runner = CliRunner()
        result = runner.invoke(app, ["references", "export", "-o", str(empty_project / "refs.bib")])
        assert result.exit_code == 1
        assert "No references" in result.output

    def test_export_unknown_format(self, project_with_refs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(project_with_refs)
        runner = CliRunner()
        result = runner.invoke(app, ["references", "export", "-o", str(project_with_refs / "refs.txt")])
        assert result.exit_code == 1
        assert "Unknown format" in result.output

    def test_export_bibtex_contains_all_fields(self, project_with_refs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(project_with_refs)
        runner = CliRunner()
        out_file = project_with_refs / "refs.bib"
        result = runner.invoke(app, ["references", "export", "-o", str(out_file)])
        assert result.exit_code == 0
        content = out_file.read_text(encoding="utf-8")
        # Smith entry should have author, title, year, journal, doi
        smith_entry = [part for part in content.split("@") if "smith2020" in part]
        assert smith_entry
        smith_text = smith_entry[0]
        assert "author" in smith_text
        assert "title" in smith_text
        assert "year" in smith_text
        assert "journal" in smith_text
        assert "doi" in smith_text


class TestReferencesHelp:
    """Tests for CLI registration and help."""

    def test_references_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["references", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "export" in result.output

    def test_references_no_args_shows_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["references"])
        assert result.exit_code != 0  # no_args_is_help


@pytest.fixture
def project_with_refs(tmp_path: Path) -> Path:
    """Create a minimal project directory with a references.json."""
    knox_dir = tmp_path / ".knox"
    knox_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "sources").mkdir(exist_ok=True)
    (tmp_path / "wiki").mkdir(exist_ok=True)

    store = ReferenceStore(knox_dir / "references.json")
    store.add_source(
        "sources/paper.pdf",
        [
            _make_ref(id="smith2020", title="Deep Learning for NLP", family="Smith", issued="2020", doi="10.1/xyz"),
            _make_ref(id="jones2019", title="Graph Networks", family="Jones", issued="2019"),
        ],
    )
    store.save()
    return tmp_path


def _make_ref(
    *,
    id: str = "smith2020",
    title: str = "Deep Learning",
    family: str = "Smith",
    issued: str = "2020",
    doi: str = "",
    type: str = "article-journal",
) -> Reference:
    return Reference(
        id=id,
        type=type,
        title=title,
        author=[Author(family=family)],
        issued=issued,
        container_title="Journal of AI",
        doi=doi,
        raw=f"Smith, J. ({issued}). {title}.",
    )


@pytest.fixture
def empty_project(tmp_path: Path) -> Path:
    """Create a project dir without references.json."""
    knox_dir = tmp_path / ".knox"
    knox_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "sources").mkdir(exist_ok=True)
    (tmp_path / "wiki").mkdir(exist_ok=True)
    return tmp_path
