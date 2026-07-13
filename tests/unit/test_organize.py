"""Unit tests for the ``knox organize`` LLM-based file categorisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from knox.agents.categorize import CategorizeAgent
from knox.agents.categorize_models import CategoryNode as PydanticNode
from knox.agents.categorize_models import CategoryTree as PydanticTree
from knox.cli.app import app
from knox.compilation._two_phase_core import Extraction
from knox.core.config import OrganizeConfig, resolve_config
from knox.core.config.keys import (
    ORGANIZE_COLLAPSE_SINGLE_FILE,
    ORGANIZE_MAX_DEPTH,
    ORGANIZE_MIN_FILES_PER_CATEGORY,
    ORGANIZE_MIN_SUBCATEGORIES,
)
from knox.core.paths import CAPTURED_SUBDIR, DISTILLED_SUBDIR, KNOX_DIR, ORGANIZE_REPORT_FILE
from knox.organize import (
    CategoryNode,
    CategoryTree,
    OrganizeResult,
    build_link_remap,
    flatten_sources,
    load_all_extractions,
    materialize_tree,
    move_files,
    organize_sources,
    propose_categories,
    update_wiki_links,
    validate_tree,
)
from knox.organize.categorize import _slugify_category, _tree_from_json
from knox.organize.engine import resolve_organize_config
from knox.organize.moves import FileMove

# ── Helpers ──────────────────────────────────────────────────────────────


def _fake_agent(payload: dict[str, Any] | None) -> CategorizeAgent:
    """Return a CategorizeAgent that yields the given CategoryTree once."""
    agent = MagicMock(spec=CategorizeAgent)
    agent._timeout = 120
    if payload is None:
        agent.run.side_effect = Exception("LLM failed")
    else:
        cats = []
        for c in payload.get("categories", []):
            subs = [PydanticNode(name=s["name"], files=s.get("files", [])) for s in c.get("subcategories", [])]
            cats.append(PydanticNode(name=c["name"], files=c.get("files", []), subcategories=subs))
        tree = PydanticTree(categories=cats, unfiled=payload.get("unfiled", []))
        agent.run.return_value = tree
    return agent


def _fake_llm(payload: dict[str, Any] | None) -> CategorizeAgent:
    """Legacy alias for _fake_agent."""
    return _fake_agent(payload)


def _write_extraction(base_dir: Path, source: str, summary: str = "test summary") -> None:
    cache = base_dir / KNOX_DIR / "cache" / "extract"
    cache.mkdir(parents=True, exist_ok=True)
    entry = {
        "key": "x",
        "layer": "extract",
        "deps": {},
        "value": {"source": source, "summary": summary, "entities": [], "concepts": [], "tags": []},
    }
    out = cache / f"{source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entry), encoding="utf-8")


# ── Slug helper ──────────────────────────────────────────────────────────


class TestSlugify:
    def test_lowercase(self) -> None:
        assert _slugify_category("MyCategory") == "mycategory"

    def test_spaces_to_hyphens(self) -> None:
        assert _slugify_category("hello world") == "hello-world"

    def test_strips_invalid_chars(self) -> None:
        assert _slugify_category("foo/bar:baz") == "foo-bar-baz"

    def test_collapses_runs(self) -> None:
        assert _slugify_category("a  -  b") == "a-b"

    def test_empty_falls_back(self) -> None:
        assert _slugify_category("") == "misc"
        assert _slugify_category("///") == "misc"


# ── CategoryNode helpers ────────────────────────────────────────────────


class TestCategoryNode:
    def test_all_files_flat(self) -> None:
        node = CategoryNode(name="a", files=["x.md", "y.md"])
        assert node.all_files() == ["x.md", "y.md"]

    def test_all_files_nested(self) -> None:
        node = CategoryNode(
            name="a",
            files=["x.md"],
            subcategories=[CategoryNode(name="b", files=["y.md", "z.md"])],
        )
        assert node.all_files() == ["x.md", "y.md", "z.md"]


# ── load_all_extractions ────────────────────────────────────────────────


class TestLoadExtractions:
    def test_returns_matching_sources(self, tmp_path: Path) -> None:
        base = tmp_path
        sources = base / "sources"
        sources.mkdir()
        (sources / "a.md").write_text("a")
        (sources / "b.md").write_text("b")
        _write_extraction(base, "a.md", "about a")
        _write_extraction(base, "b.md", "about b")

        result = load_all_extractions(base, sources)
        assert set(result) == {"a.md", "b.md"}

    def test_drops_missing_files(self, tmp_path: Path) -> None:
        base = tmp_path
        sources = base / "sources"
        sources.mkdir()
        _write_extraction(base, "ghost.md", "no file")

        result = load_all_extractions(base, sources)
        assert result == {}

    def test_drops_files_without_summary(self, tmp_path: Path) -> None:
        base = tmp_path
        sources = base / "sources"
        sources.mkdir()
        (sources / "a.md").write_text("a")
        _write_extraction(base, "a.md", summary="")

        result = load_all_extractions(base, sources)
        assert result == {}

    def test_skips_outside_sources(self, tmp_path: Path) -> None:
        base = tmp_path
        sources = base / "sources"
        sources.mkdir()
        other = base / "other"
        other.mkdir()
        (other / "x.md").write_text("x")
        cache = base / KNOX_DIR / "cache" / "extract"
        cache.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": "k",
            "layer": "extract",
            "deps": {},
            "value": {"source": "../other/x.md", "summary": "s", "entities": [], "concepts": [], "tags": []},
        }
        (cache / "x.json").write_text(json.dumps(entry), encoding="utf-8")

        result = load_all_extractions(base, sources)
        assert result == {}

    def test_skips_protected_extractions(self, tmp_path: Path) -> None:
        base = tmp_path
        sources = base / "sources"
        (sources / CAPTURED_SUBDIR).mkdir(parents=True)
        (sources / CAPTURED_SUBDIR / "a.md").write_text("a")
        (sources / "b.md").write_text("b")
        # Extraction cache entry for a file in _captured/
        _write_extraction(base, f"{CAPTURED_SUBDIR}/a.md", "captured note")
        # Normal extraction
        _write_extraction(base, "b.md", "normal source")

        result = load_all_extractions(base, sources)
        # Protected file excluded; normal file included
        assert f"{CAPTURED_SUBDIR}/a.md" not in result
        assert "b.md" in result


# ── validate_tree ───────────────────────────────────────────────────────


class TestValidateTree:
    def test_max_depth_collapses_children(self) -> None:
        tree = CategoryTree(
            categories=[
                CategoryNode(
                    name="a",
                    files=["sibling.md"],
                    subcategories=[
                        CategoryNode(
                            name="b",
                            subcategories=[CategoryNode(name="c", files=["deep.md"])],
                        ),
                    ],
                ),
            ],
        )
        validated = validate_tree(
            tree,
            OrganizeConfig(max_depth=2, min_subcategories=2, min_files_per_category=1),
        )
        # b at depth 2, so its children collapse into b's files
        assert validated.categories[0].name == "a"
        b = validated.categories[0].subcategories[0]
        assert b.name == "b"
        # c collapsed into b's files
        assert "deep.md" in b.files

    def test_single_file_collapse(self) -> None:
        tree = CategoryTree(
            categories=[
                CategoryNode(
                    name="parent",
                    subcategories=[CategoryNode(name="only", files=["solo.md"])],
                ),
            ],
        )
        validated = validate_tree(tree, OrganizeConfig(max_depth=3, min_subcategories=2))
        # single-file child collapsed into parent
        assert validated.categories[0].subcategories == []
        assert "solo.md" in validated.categories[0].files

    def test_min_subcategories_promotes(self) -> None:
        # 1 subcategory (which itself has subs) when 2 are required — promote all files up
        tree = CategoryTree(
            categories=[
                CategoryNode(
                    name="parent",
                    subcategories=[
                        CategoryNode(
                            name="only",
                            subcategories=[CategoryNode(name="deep", files=["a.md", "b.md"])],
                        ),
                    ],
                ),
            ],
        )
        validated = validate_tree(tree, OrganizeConfig(max_depth=3, min_subcategories=2))
        assert validated.categories[0].subcategories == []
        assert sorted(validated.categories[0].files) == ["a.md", "b.md"]

    def test_disabled_collapse_keeps_single(self) -> None:
        tree = CategoryTree(
            categories=[
                CategoryNode(
                    name="parent",
                    subcategories=[CategoryNode(name="only", files=["solo.md"])],
                ),
            ],
        )
        validated = validate_tree(
            tree,
            OrganizeConfig(max_depth=3, min_subcategories=2, collapse_single_file=False),
        )
        assert len(validated.categories[0].subcategories) == 1

    def test_min_files_dissolves_sparse_leaf(self) -> None:
        # A leaf category with fewer than min_files gets merged into its parent.
        tree = CategoryTree(
            categories=[
                CategoryNode(
                    name="parent",
                    files=["p1.md", "p2.md"],
                    subcategories=[
                        CategoryNode(name="sparse", files=["lonely.md"]),
                        CategoryNode(name="full", files=["a.md", "b.md"]),
                    ],
                ),
            ],
        )
        validated = validate_tree(
            tree,
            OrganizeConfig(max_depth=3, min_subcategories=2, min_files_per_category=2),
        )
        names = {s.name for s in validated.categories[0].subcategories}
        assert "full" in names  # kept (2 files)
        assert "sparse" not in names  # dissolved (1 file < 2)
        assert "lonely.md" in validated.categories[0].files  # promoted up

    def test_min_files_top_level_goes_to_unfiled(self) -> None:
        # A top-level leaf below the minimum has no parent — files go to unfiled.
        tree = CategoryTree(
            categories=[
                CategoryNode(name="big", files=["a.md", "b.md", "c.md"]),
                CategoryNode(name="tiny", files=["solo.md"]),
            ],
        )
        validated = validate_tree(
            tree,
            OrganizeConfig(max_depth=3, min_subcategories=2, min_files_per_category=2),
        )
        names = {c.name for c in validated.categories}
        assert "big" in names  # kept (3 files)
        assert "tiny" not in names  # dissolved (1 file < 2)
        assert "solo.md" in validated.unfiled

    def test_min_files_clamped_to_corpus_size(self) -> None:
        # A minimum larger than the total document count degrades gracefully:
        # effective minimum is clamped, so a single all-encompassing category survives.
        tree = CategoryTree(
            categories=[CategoryNode(name="only", files=["a.md", "b.md"])],
        )
        validated = validate_tree(
            tree,
            OrganizeConfig(max_depth=3, min_subcategories=2, min_files_per_category=100),
        )
        # clamped to 2 → the 2-file category survives instead of dissolving
        assert len(validated.categories) == 1
        assert "a.md" in validated.categories[0].files


# ── propose_categories ──────────────────────────────────────────────────


class TestProposeCategories:
    def test_empty_extractions(self) -> None:
        tree = propose_categories({}, OrganizeConfig(), _fake_llm(None))
        assert tree == CategoryTree()

    def test_parses_payload(self) -> None:
        payload = {
            "categories": [
                {"name": "papers", "files": ["a.md"], "subcategories": []},
                {
                    "name": "notes",
                    "files": [],
                    "subcategories": [{"name": "meetings", "files": ["m.md"], "subcategories": []}],
                },
            ],
            "unfiled": [],
        }
        tree = propose_categories(
            {"a.md": MagicMock(summary="x"), "m.md": MagicMock(summary="y")},
            OrganizeConfig(),
            _fake_llm(payload),
        )
        assert tree.categories[0].name == "papers"
        assert tree.categories[1].subcategories[0].name == "meetings"

    def test_llm_failure_returns_unfiled(self) -> None:
        a = Extraction(source="a.md", summary="x")
        tree = propose_categories({"a.md": a}, OrganizeConfig(), _fake_llm(None))
        assert tree.unfiled == ["a.md"]
        assert tree.categories == []


# ── _tree_from_json edge cases ──────────────────────────────────────────


class TestTreeFromJson:
    def test_drops_unknown_files(self) -> None:
        payload = {
            "categories": [
                {"name": "a", "files": ["keep.md", "ghost.md"], "subcategories": []},
            ],
            "unfiled": [],
        }
        tree = _tree_from_json(payload, ["keep.md"])
        assert tree.categories[0].files == ["keep.md"]

    def test_collects_omitted_into_unfiled(self) -> None:
        payload = {"categories": [], "unfiled": []}
        tree = _tree_from_json(payload, ["x.md", "y.md"])
        assert sorted(tree.unfiled) == ["x.md", "y.md"]

    def test_non_dict_payload(self) -> None:
        tree = _tree_from_json("not a dict", ["x.md"])
        assert tree.unfiled == ["x.md"]


# ── move.py: flatten / materialize / move ───────────────────────────────


class TestFlattenSources:
    def test_flattens_nested(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "x.md").write_text("x")
        (nested / "y.md").write_text("y")

        moves = flatten_sources(tmp_path)

        assert (tmp_path / "x.md").exists()
        assert (tmp_path / "y.md").exists()
        assert len(moves) == 2
        # All moves go to top level
        for m in moves:
            assert "/" not in m.new

    def test_handles_collision(self, tmp_path: Path) -> None:
        sub = tmp_path / "a"
        sub.mkdir()
        (tmp_path / "x.md").write_text("first")
        (sub / "x.md").write_text("second")

        moves = flatten_sources(tmp_path)

        # Only the nested file is moved (top-level files are skipped).
        # The destination gets a unique name to avoid collision.
        assert len(moves) == 1
        assert moves[0].old == "a/x.md"
        assert moves[0].new == "x-2.md"

    def test_missing_dir(self, tmp_path: Path) -> None:
        assert flatten_sources(tmp_path / "nope") == []

    def test_skips_protected_subdirs(self, tmp_path: Path) -> None:
        sources = tmp_path / "sources"
        (sources / CAPTURED_SUBDIR).mkdir(parents=True)
        (sources / DISTILLED_SUBDIR).mkdir(parents=True)
        (sources / "sub").mkdir()
        (sources / CAPTURED_SUBDIR / "a.md").write_text("a")
        (sources / DISTILLED_SUBDIR / "b.md").write_text("b")
        (sources / "sub" / "c.md").write_text("c")

        moves = flatten_sources(sources)

        # Protected files stay in place
        assert (sources / CAPTURED_SUBDIR / "a.md").is_file()
        assert (sources / DISTILLED_SUBDIR / "b.md").is_file()
        # Normal nested file was flattened
        assert (sources / "c.md").is_file()
        assert not (sources / "sub" / "c.md").exists()
        # Only the normal file appears in the move list
        assert len(moves) == 1
        assert moves[0].old == "sub/c.md"

    def test_remap_files_after_flatten(self, tmp_path: Path) -> None:
        """Regression: tree file paths must be rewritten after flatten_sources.

        Previously the flatten return value was discarded, so move_files
        looked for pre-flatten paths (e.g. 'sub/foo.pdf') and warned
        'File in tree but missing on disk'.
        """
        sources = tmp_path / "sources"
        (sources / "sub").mkdir(parents=True)
        (sources / "sub" / "foo.pdf").write_text("x")
        (sources / "sub" / "bar.pdf").write_text("y")

        # Simulate the engine flow: build tree from pre-flatten paths, then
        # flatten, then remap.
        tree = CategoryTree(
            categories=[
                CategoryNode(name="papers", files=["sub/foo.pdf", "sub/bar.pdf"]),
            ],
        )
        flat_moves = flatten_sources(sources)
        flatten_remap = {m.old: m.new for m in flat_moves}
        assert flatten_remap == {"sub/foo.pdf": "foo.pdf", "sub/bar.pdf": "bar.pdf"}
        tree.remap_files(flatten_remap)

        # Tree now references post-flatten top-level paths
        assert tree.categories[0].files == ["foo.pdf", "bar.pdf"]

    def test_move_files_with_nested_subcategories(self, tmp_path: Path) -> None:
        """Regression: subcategory recursion must resolve src from sources_dir,
        not from the parent category dir.

        Previously _move_node used ``target_dir.parent`` to resolve the source
        path. For nested categories that resolved to ``sources/cat/sub/file``
        instead of ``sources/file``, causing every subcategory file to warn
        'File in tree but missing on disk'.
        """
        sources = tmp_path / "sources"
        sources.mkdir()
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (sources / name).write_text("x")

        tree = CategoryTree(
            categories=[
                CategoryNode(
                    name="cat",
                    subcategories=[
                        CategoryNode(name="sub", files=["a.pdf", "b.pdf"]),
                    ],
                ),
            ],
        )

        moves = move_files(sources, tree)

        moved_names = {m.new for m in moves}
        assert "cat/sub/a.pdf" in moved_names
        assert "cat/sub/b.pdf" in moved_names
        assert (sources / "cat" / "sub" / "a.pdf").is_file()
        assert (sources / "cat" / "sub" / "b.pdf").is_file()
        assert not (sources / "a.pdf").exists()


class TestMaterializeTree:
    def test_creates_dirs(self, tmp_path: Path) -> None:
        tree = CategoryTree(
            categories=[
                CategoryNode(
                    name="papers",
                    files=["a.md"],
                    subcategories=[CategoryNode(name="ml", files=["b.md"])],
                ),
            ],
        )
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")

        materialize_tree(tmp_path, tree)

        assert (tmp_path / "papers").is_dir()
        assert (tmp_path / "papers" / "ml").is_dir()

    def test_skips_empty_subcategories(self, tmp_path: Path) -> None:
        tree = CategoryTree(
            categories=[
                CategoryNode(
                    name="papers",
                    subcategories=[CategoryNode(name="empty", subcategories=[CategoryNode(name="deeper", subcategories=[])])],
                ),
            ],
        )
        materialize_tree(tmp_path, tree)
        # papers created (has subcategories), but empty/ and deeper/ not created
        assert (tmp_path / "papers").is_dir()
        assert not (tmp_path / "papers" / "empty").exists()


class TestMoveFiles:
    def test_moves_to_categories(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")
        tree = CategoryTree(
            categories=[
                CategoryNode(name="papers", files=["a.md", "b.md"], subcategories=[]),
            ],
        )

        moves = move_files(tmp_path, tree)

        assert (tmp_path / "papers" / "a.md").exists()
        assert (tmp_path / "papers" / "b.md").exists()
        assert not (tmp_path / "a.md").exists()
        assert len(moves) == 2
        assert {m.old for m in moves} == {"a.md", "b.md"}

    def test_collision_suffixed(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("first")
        (tmp_path / "papers" / "a.md").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "papers" / "a.md").write_text("second")
        tree = CategoryTree(
            categories=[CategoryNode(name="papers", files=["a.md"], subcategories=[])],
        )

        moves = move_files(tmp_path, tree)

        assert any(m.new.endswith("a-2.md") for m in moves)

    def test_unfiled_files_unchanged(self, tmp_path: Path) -> None:
        (tmp_path / "loose.md").write_text("loose")
        tree = CategoryTree(categories=[], unfiled=["loose.md"])
        assert move_files(tmp_path, tree) == []
        assert (tmp_path / "loose.md").exists()


# ── links.py: build_link_remap / update_wiki_links ──────────────────────


class TestBuildLinkRemap:
    def test_basename_map(self) -> None:
        remap = build_link_remap([("papers/x.md", "docs/x-renamed.md"), ("y.md", "y.md")])
        # Basename changed -> remap contains entry
        assert remap.get("x.md") == "x-renamed.md"
        # Basename unchanged -> no entry
        assert remap.get("y.md") is None

    def test_multiple_targets_warns(self) -> None:
        remap = build_link_remap(
            [
                ("a/x.md", "b/x-1.md"),
                ("c/x.md", "d/x-2.md"),
            ],
        )
        # First wins, second is dropped (with warning)
        assert remap.get("x.md") == "x-1.md"


class TestUpdateWikiLinks:
    def test_updates_inline_citations(self, tmp_path: Path) -> None:
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        page = wiki / "concept.md"
        page.write_text(
            "Some claim [old.md:1-2] here.\nAnother [old.md:5-7] ref.",
            encoding="utf-8",
        )

        remap = build_link_remap([("sources/old.md", "sources/new.md")])
        counts = update_wiki_links(tmp_path, wiki, remap)

        assert counts["wiki_files"] == 2
        assert page.read_text(encoding="utf-8") == ("Some claim [new.md:1-2] here.\nAnother [new.md:5-7] ref.")

    def test_updates_provenance(self, tmp_path: Path) -> None:
        prov_dir = tmp_path / KNOX_DIR / "provenance"
        prov_dir.mkdir(parents=True)
        prov_file = prov_dir / "node.json"
        prov_file.write_text(
            json.dumps(
                {
                    "node_id": "node",
                    "title": "T",
                    "citations": ["[old.md:1-2]"],
                },
            ),
            encoding="utf-8",
        )

        wiki = tmp_path / "wiki"
        wiki.mkdir()

        remap = build_link_remap([("sources/old.md", "sources/new.md")])
        counts = update_wiki_links(tmp_path, wiki, remap)

        assert counts["provenance_files"] == 1
        data = json.loads(prov_file.read_text(encoding="utf-8"))
        assert data["citations"] == ["[new.md:1-2]"]

    def test_updates_wiki_node_metadata(self, tmp_path: Path) -> None:
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        cache_dir = tmp_path / KNOX_DIR
        cache_dir.mkdir()
        cache = cache_dir / "last_wiki.json"
        cache.write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "n1",
                            "title": "T",
                            "content": "x",
                            "metadata": {"source_file": "sources/old.md"},
                        },
                    ],
                    "edges": [],
                    "toc": [],
                },
            ),
            encoding="utf-8",
        )

        remap = build_link_remap([("sources/old.md", "sources/new.md")])
        counts = update_wiki_links(tmp_path, wiki, remap)

        assert counts["wiki_node_metadata"] == 1
        data = json.loads(cache.read_text(encoding="utf-8"))
        assert data["nodes"][0]["metadata"]["source_file"] == "sources/new.md"

    def test_updates_tier_state(self, tmp_path: Path) -> None:
        state_dir = tmp_path / KNOX_DIR
        state_dir.mkdir()
        state = state_dir / "compile_state.json"
        state.write_text(
            json.dumps(
                {
                    "entries": {
                        "a": {"source_file": "sources/old.md", "stem": "a"},
                    },
                },
            ),
            encoding="utf-8",
        )

        wiki = tmp_path / "wiki"
        wiki.mkdir()

        remap = build_link_remap([("sources/old.md", "sources/new.md")])
        counts = update_wiki_links(tmp_path, wiki, remap)

        assert counts["tier_state"] == 1
        data = json.loads(state.read_text(encoding="utf-8"))
        assert data["entries"]["a"]["source_file"] == "sources/new.md"

    def test_empty_remap_is_noop(self, tmp_path: Path) -> None:
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        counts = update_wiki_links(tmp_path, wiki, build_link_remap([]))
        assert counts == {
            "wiki_files": 0,
            "wiki_node_metadata": 0,
            "provenance_files": 0,
            "tier_state": 0,
        }


# ── engine: organize_sources ────────────────────────────────────────────


class TestOrganizeSourcesDryRun:
    def test_dry_run_does_not_move(self, tmp_path: Path) -> None:
        base = tmp_path
        sources = base / "sources"
        sources.mkdir()
        output = base / "wiki"
        (sources / "a.md").write_text("a")
        _write_extraction(base, "a.md", "summary")

        llm = _fake_llm({"categories": [{"name": "papers", "files": ["a.md"], "subcategories": []}], "unfiled": []})

        result = organize_sources(
            base_dir=base,
            sources_dir=sources,
            output_dir=output,
            categorize_agent=llm,
            config=OrganizeConfig(),
            dry_run=True,
        )

        assert result.dry_run
        assert not (sources / "papers").exists()
        assert (sources / "a.md").exists()

    def test_no_extractions_returns_empty(self, tmp_path: Path) -> None:
        base = tmp_path
        sources = base / "sources"
        sources.mkdir()
        output = base / "wiki"

        result = organize_sources(
            base_dir=base,
            sources_dir=sources,
            output_dir=output,
            categorize_agent=_fake_llm(None),
            dry_run=False,
        )

        assert result.tree == CategoryTree()
        assert result.moves == []


class TestOrganizeSourcesApply:
    def test_full_run_moves_and_writes_report(self, tmp_path: Path) -> None:
        base = tmp_path
        sources = base / "sources"
        sources.mkdir()
        output = base / "wiki"
        (sources / "a.md").write_text("a")
        (sources / "b.md").write_text("b")
        _write_extraction(base, "a.md", "about a")
        _write_extraction(base, "b.md", "about b")

        llm = _fake_llm(
            {
                "categories": [
                    {
                        "name": "papers",
                        "files": ["a.md", "b.md"],
                        "subcategories": [],
                    },
                ],
                "unfiled": [],
            },
        )

        result = organize_sources(
            base_dir=base,
            sources_dir=sources,
            output_dir=output,
            categorize_agent=llm,
            config=OrganizeConfig(),
            dry_run=False,
        )

        assert (sources / "papers" / "a.md").exists()
        assert (sources / "papers" / "b.md").exists()
        assert len(result.moves) == 2
        # Report written
        report = base / KNOX_DIR / ORGANIZE_REPORT_FILE
        assert report.is_file()
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["dry_run"] is False
        assert len(data["moves"]) == 2


class TestOrganizeReportPath:
    def test_uses_path_constant(self) -> None:
        assert ORGANIZE_REPORT_FILE == "last_organize.json"


# ── Config precedence for [organize] ────────────────────────────────────


class TestOrganizeConfig:
    def test_defaults(self) -> None:
        config = resolve_organize_config()
        assert config.max_depth == 3
        assert config.min_subcategories == 2
        assert config.min_files_per_category == 2
        assert config.collapse_single_file is True

    def test_cli_override(self) -> None:
        config = resolve_organize_config(
            {
                ORGANIZE_MAX_DEPTH: 4,
                ORGANIZE_MIN_SUBCATEGORIES: 3,
                ORGANIZE_MIN_FILES_PER_CATEGORY: 4,
                ORGANIZE_COLLAPSE_SINGLE_FILE: False,
            },
        )
        assert config.max_depth == 4
        assert config.min_subcategories == 3
        assert config.min_files_per_category == 4
        assert config.collapse_single_file is False

    def test_min_files_below_one_rejected(self, tmp_path: Any, monkeypatch: Any) -> None:
        """min_files_per_category < 1 is invalid and must raise."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="min_files_per_category"):
            resolve_organize_config({ORGANIZE_MIN_FILES_PER_CATEGORY: 0})
        with pytest.raises(ValueError, match="min_files_per_category"):
            resolve_organize_config({ORGANIZE_MIN_FILES_PER_CATEGORY: -3})

    def test_env_override(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_ORGANIZE_MAX_DEPTH", "2")
        cfg = resolve_config()
        assert cfg.get(ORGANIZE_MAX_DEPTH) == 2


# ── CLI overrides for [organize] ────────────────────────────────────────


class TestOrganizeCliCommand:
    def test_organize_command_registered(self) -> None:
        command_names = {cmd.name for cmd in app.registered_commands if cmd.name}
        assert "organize" in command_names


# ── Organized result dataclass ──────────────────────────────────────────


class TestOrganizeResult:
    def test_moved_count(self) -> None:
        result = OrganizeResult(
            tree=CategoryTree(),
            moves=[FileMove(old="a.md", new="papers/a.md"), FileMove(old="b.md", new="papers/b.md")],
        )
        assert result.moved_count == 2
