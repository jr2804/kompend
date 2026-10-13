"""Regression: lint consumers read the compiled wiki from the vault store.

Previously ``run_full_lint_flow`` gated on ``.kompend/last_wiki.json`` existing, and
``_load_degree_map`` parsed that file — but nothing writes it since the store
migration, so ``kompend lint all`` raised "No compiled wiki found" even after a
successful compile, and the gap degree map was permanently empty. These tests
seed a wiki snapshot through the real writer and assert the store path is used.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from kompend.core.cache.wiki import write_wiki_cache
from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.paths import VaultPaths
from kompend.lint.flow import run_full_lint_flow
from kompend.lint.gap_analysis import load_degree_map


def test_load_degree_map_reads_store_snapshot(tmp_path: Path) -> None:
    """The degree map is built from the store's compiled-wiki snapshot.

    Fails against the old file-based implementation, which returned ``None``
    because ``.kompend/last_wiki.json`` is never written.
    """
    _seed_wiki(tmp_path)
    assert load_degree_map(tmp_path) == {"a": 1, "b": 1}


def test_load_degree_map_absent_store_is_none(tmp_path: Path) -> None:
    assert load_degree_map(tmp_path) is None


def test_run_full_lint_flow_succeeds_on_seeded_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The flow reaches its body (no bogus "No compiled wiki found") when the
    store holds a snapshot. Phases are stubbed so this tests the gate, not lint rules.
    """
    _seed_wiki(tmp_path)
    monkeypatch.setattr("kompend.lint.flow._make_lint_agent", lambda cfg: (None, None))
    monkeypatch.setattr("kompend.lint.flow._plan", lambda *args, **kwargs: [])
    result = run_full_lint_flow(VaultPaths(tmp_path), types.SimpleNamespace(raw={}), show_progress=False, json_output=True)
    assert isinstance(result, dict)


def _seed_wiki(base_dir: Path) -> None:
    wiki = CompiledWiki(
        nodes=[
            WikiNode(id="a", title="A", content="a", node_type="concept"),
            WikiNode(id="b", title="B", content="b", node_type="concept"),
        ],
        edges=[{"source": "a", "target": "b", "type": "related"}],
        toc=[],
    )
    write_wiki_cache(wiki, base_dir)


def test_run_full_lint_flow_raises_without_wiki(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_full_lint_flow(VaultPaths(tmp_path), types.SimpleNamespace(raw={}), show_progress=False, json_output=True)
