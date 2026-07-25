"""Tests for the ReferenceStore JSON-backed reference storage."""

from __future__ import annotations

import json
from pathlib import Path

from knox.core.models.references import Author, Reference
from knox.core.pipeline.references import ReferenceStore


def _make_ref(
    id: str = "smith2020",
    title: str = "A Paper",
    family: str = "Smith",
    issued: str = "2020",
) -> Reference:
    return Reference(
        id=id,
        type="article-journal",
        title=title,
        author=[Author(family=family)],
        issued=issued,
        raw=f"{family}, J. ({issued}). {title}.",
    )


# ---------------------------------------------------------------------------
# Load / save round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_empty_store_has_no_file(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        assert store.total_count() == 0
        assert store.source_count() == 0

    def test_round_trip(self, tmp_path: Path) -> None:
        ref_file = tmp_path / "refs.json"
        store = ReferenceStore(ref_file)
        store.add_source("paper.pdf", [_make_ref()])
        store.save()

        assert ref_file.exists()
        data = json.loads(ref_file.read_text())
        assert "paper.pdf" in data
        assert data["paper.pdf"][0]["id"] == "smith2020"

        store2 = ReferenceStore(ref_file)
        refs = store2.get_source("paper.pdf")
        assert len(refs) == 1
        assert refs[0].title == "A Paper"
        assert refs[0].author[0].family == "Smith"

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        ref_file = tmp_path / "sub" / "dir" / "refs.json"
        store = ReferenceStore(ref_file)
        store.add_source("x.md", [_make_ref()])
        store.save()
        assert ref_file.exists()

    def test_replace_on_add(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("s.pdf", [_make_ref(id="a"), _make_ref(id="b")])
        assert store.total_count() == 2
        # Second add replaces, not appends
        store.add_source("s.pdf", [_make_ref(id="c")])
        refs = store.get_source("s.pdf")
        assert len(refs) == 1
        assert refs[0].id == "c"


# ---------------------------------------------------------------------------
# Corrupt / missing files
# ---------------------------------------------------------------------------


class TestCorruptFile:
    def test_missing_file_starts_empty(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "nonexistent.json")
        assert store.total_count() == 0

    def test_corrupt_json_starts_empty(self, tmp_path: Path) -> None:
        ref_file = tmp_path / "refs.json"
        ref_file.write_text("{ this is not valid json }}}")
        store = ReferenceStore(ref_file)
        assert store.total_count() == 0

    def test_non_dict_json_starts_empty(self, tmp_path: Path) -> None:
        ref_file = tmp_path / "refs.json"
        ref_file.write_text("[1, 2, 3]")
        store = ReferenceStore(ref_file)
        assert store.total_count() == 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_total_count_across_sources(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("a.pdf", [_make_ref(id="r1"), _make_ref(id="r2")])
        store.add_source("b.pdf", [_make_ref(id="r3")])
        assert store.total_count() == 3
        assert store.source_count() == 2

    def test_source_count_excludes_empty(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("a.pdf", [_make_ref()])
        store.add_source("b.pdf", [])
        assert store.source_count() == 1
        assert store.total_count() == 1

    def test_all_references_deduped_by_label(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        # Same label "Smith 2020" in two sources → deduped to one
        store.add_source("a.pdf", [_make_ref(id="r1", family="Smith", issued="2020")])
        store.add_source("b.pdf", [_make_ref(id="r2", family="Smith", issued="2020")])
        all_refs = store.all_references()
        assert len(all_refs) == 1

    def test_all_references_different_labels_not_deduped(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("a.pdf", [_make_ref(id="r1", family="Smith", issued="2020")])
        store.add_source("b.pdf", [_make_ref(id="r2", family="Doe", issued="2021")])
        all_refs = store.all_references()
        assert len(all_refs) == 2

    def test_to_dict(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("a.pdf", [_make_ref()])
        d = store.to_dict()
        assert "a.pdf" in d
        assert d["a.pdf"][0]["id"] == "smith2020"


# ---------------------------------------------------------------------------
# Malformed entries
# ---------------------------------------------------------------------------


class TestMalformedEntries:
    def test_get_source_skips_malformed(self, tmp_path: Path) -> None:
        ref_file = tmp_path / "refs.json"
        data = {
            "s.pdf": [
                {"id": "good", "type": "book", "title": "T"},
                {"this": "has no required fields and wrong types", "type": 12345},
            ]
        }
        ref_file.write_text(json.dumps(data))
        store = ReferenceStore(ref_file)
        refs = store.get_source("s.pdf")
        # The malformed entry is silently skipped
        assert len(refs) == 1
        assert refs[0].id == "good"
