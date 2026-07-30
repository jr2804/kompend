"""Tests for reference DOI/URL resolution (Phase 4 ``fetch``).

Covers:
- ``ReferenceStore.unresolved_indices`` / ``update_reference``
- ``resolve_unresolved_references`` flow (with a stub agent)
- ``knox references fetch`` CLI wiring (offline, stubbed agent)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.cli.cmds.knowledge import references as references_cmd
from knox.compilation.flows.references import resolve_unresolved_references
from knox.core.models.references import (
    Author,
    Reference,
    ReferenceResolutionOutput,
)
from knox.core.pipeline.references import ReferenceStore


class _StubResolver:
    """Stand-in for ReferenceResolverAgent that always resolves."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        self.calls: list = []

    def run(self, deps: object, *, message_history: object | None = None) -> ReferenceResolutionOutput:
        self.calls.append(deps)
        return ReferenceResolutionOutput(
            resolved=True,
            doi="10.1000/stub",
            url="https://example.org/stub",
            note="stubbed",
        )


class TestStoreResolutionHelpers:
    def test_unresolved_indices_excludes_resolved(self, store_with_unresolved: ReferenceStore) -> None:
        pending = store_with_unresolved.unresolved_indices()
        assert pending == [("sources/paper.pdf", 1), ("sources/paper.pdf", 2)]

    def test_unresolved_indices_filtered_by_source(self, store_with_unresolved: ReferenceStore) -> None:
        pending = store_with_unresolved.unresolved_indices("sources/paper.pdf")
        assert len(pending) == 2

    def test_update_reference_fills_empty(self, store_with_unresolved: ReferenceStore) -> None:
        changed = store_with_unresolved.update_reference("sources/paper.pdf", 1, doi="10.2/abc")
        assert changed
        refs = store_with_unresolved.get_source("sources/paper.pdf")
        assert refs[1].doi == "10.2/abc"

    def test_update_reference_keeps_existing(self, store_with_unresolved: ReferenceStore) -> None:
        changed = store_with_unresolved.update_reference("sources/paper.pdf", 0, doi="10.99/overwrite")
        assert not changed  # already had a doi
        refs = store_with_unresolved.get_source("sources/paper.pdf")
        assert refs[0].doi == "10.1/xyz"

    def test_update_reference_bad_index(self, store_with_unresolved: ReferenceStore) -> None:
        assert not store_with_unresolved.update_reference("sources/paper.pdf", 99, doi="x")


class TestResolveUnresolvedFlow:
    def test_resolves_and_saves(self, store_with_unresolved: ReferenceStore) -> None:
        agent = _StubResolver()
        resolved = resolve_unresolved_references(
            store_with_unresolved,
            agent,
            timeout=10.0,
        )
        assert resolved == 2
        # Reload from disk to confirm persistence.
        reloaded = ReferenceStore(store_with_unresolved._file)
        refs = reloaded.get_source("sources/paper.pdf")
        assert refs[1].doi == "10.1000/stub"
        assert refs[2].doi == "10.1000/stub"

    def test_dry_run_does_not_persist(self, store_with_unresolved: ReferenceStore, tmp_path: Path) -> None:
        agent = _StubResolver()
        resolved = resolve_unresolved_references(
            store_with_unresolved,
            agent,
            timeout=10.0,
            dry_run=True,
        )
        assert resolved == 2
        # A fresh load from disk should show no change (not saved).
        reloaded = ReferenceStore(store_with_unresolved._file)
        refs = reloaded.get_source("sources/paper.pdf")
        assert refs[1].doi == ""  # unchanged on disk

    def test_source_filter(self, store_with_unresolved: ReferenceStore) -> None:
        agent = _StubResolver()
        resolved = resolve_unresolved_references(
            store_with_unresolved,
            agent,
            timeout=10.0,
            source="sources/paper.pdf",
        )
        assert resolved == 2

    def test_agent_failure_is_non_fatal(self, store_with_unresolved: ReferenceStore) -> None:
        class _FailingResolver:
            def run(self, deps: object, *, message_history: object | None = None) -> ReferenceResolutionOutput:
                raise RuntimeError("boom")

        resolved = resolve_unresolved_references(
            store_with_unresolved,
            _FailingResolver(),  # type: ignore[arg-type]
            timeout=10.0,
        )
        assert resolved == 0


class TestFetchCli:
    @pytest.fixture
    def project(self, tmp_path: Path) -> tuple[Path, ReferenceStore]:
        knox_dir = tmp_path / ".knox"
        knox_dir.mkdir(parents=True, exist_ok=True)
        (tmp_path / "sources").mkdir(exist_ok=True)
        (tmp_path / "wiki").mkdir(exist_ok=True)
        store = ReferenceStore(knox_dir / "references.json")
        store.add_source(
            "sources/paper.pdf",
            [
                _make_ref(id="jones2019", family="Jones", issued="2019", title="Graph Networks"),
                _make_ref(id="smith2020", family="Smith", issued="2020", title="Deep Learning", doi="10.1/xyz"),
            ],
        )
        store.save()
        return tmp_path, store

    def test_no_references(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        knox_dir = tmp_path / ".knox"
        knox_dir.mkdir(parents=True, exist_ok=True)
        (tmp_path / "sources").mkdir(exist_ok=True)
        (tmp_path / "wiki").mkdir(exist_ok=True)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["references", "fetch"])
        assert result.exit_code == 0
        assert "No references found" in result.output

    def test_all_resolved(self, project: tuple[Path, ReferenceStore], monkeypatch: pytest.MonkeyPatch) -> None:
        tmp_path, store = project
        # Mark the one unresolved ref resolved so fetch short-circuits.
        store.update_reference("sources/paper.pdf", 0, doi="10.9/all")
        store.save()
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["references", "fetch"])
        assert result.exit_code == 0
        assert "All references already have a DOI or URL" in result.output

    def test_no_model_configured(self, project: tuple[Path, ReferenceStore], monkeypatch: pytest.MonkeyPatch) -> None:
        tmp_path, _ = project
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_LLM_DEFAULT_MODEL", "")
        result = CliRunner().invoke(app, ["references", "fetch"])
        assert result.exit_code == 1
        assert "No LLM model configured" in result.output

    def test_fetch_resolves(self, project: tuple[Path, ReferenceStore], monkeypatch: pytest.MonkeyPatch) -> None:
        tmp_path, _ = project
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_LLM_DEFAULT_MODEL", "fake/model")
        monkeypatch.setattr(
            references_cmd,
            "ReferenceResolverAgent",
            _StubResolver,
        )
        result = CliRunner().invoke(app, ["references", "fetch"])
        assert result.exit_code == 0
        assert "Resolved 1 reference(s)" in result.output
        # Confirm the store on disk was updated.
        store = ReferenceStore(tmp_path / ".knox" / "references.json")
        refs = store.get_source("sources/paper.pdf")
        assert refs[0].doi == "10.1000/stub"


@pytest.fixture
def store_with_unresolved(tmp_path: Path) -> ReferenceStore:
    knox_dir = tmp_path / ".knox"
    knox_dir.mkdir(parents=True, exist_ok=True)
    store = ReferenceStore(knox_dir / "references.json")
    store.add_source(
        "sources/paper.pdf",
        [
            _make_ref(id="smith2020", family="Smith", issued="2020", title="Deep Learning", doi="10.1/xyz"),
            _make_ref(id="jones2019", family="Jones", issued="2019", title="Graph Networks"),
            _make_ref(id="doe2018", family="Doe", issued="2018", title="Transformers"),
        ],
    )
    store.save()
    return store


def _make_ref(
    *,
    id: str,
    family: str,
    issued: str,
    title: str,
    doi: str = "",
    url: str = "",
    raw: str = "",
) -> Reference:
    return Reference(
        id=id,
        type="article-journal",
        title=title,
        author=[Author(family=family)],
        issued=issued,
        doi=doi,
        url=url,
        raw=raw or f"{family} ({issued}). {title}.",
    )
