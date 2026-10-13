"""Tests for fuzzy reference dedup and global consolidation.

Covers Phase 2 features:
- ``merge_similar_references`` — exact-key + embedding-based fuzzy merge
- ``consolidate_references`` — public entry point for global OKF output
- ``ReferenceStore.deduplicate`` — in-place store dedup
- ``DuplicateReferencesRule`` with ``embedding_model`` set
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from kompend.core.models import CompiledWiki
from kompend.core.models.references import Author, Reference
from kompend.core.pipeline.references import (
    ReferenceStore,
    consolidate_references,
    merge_similar_references,
)
from kompend.lint.base import LintRuleOutcome
from kompend.lint.structural import DuplicateReferencesRule

_EMPTY_WIKI = CompiledWiki(nodes=[], edges=[], toc=[])


class _FakeProvider:
    """Deterministic fake embedding provider for testing.

    Embeddings are based on character-level hashing so that similar
    strings produce similar vectors.
    """

    def __init__(self) -> None:
        self._dim = 16

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for ch in text.lower():
            vec[ord(ch) % self._dim] += 1.0
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


# ---------------------------------------------------------------------------
# merge_similar_references — exact-key dedup
# ---------------------------------------------------------------------------


class TestExactDedup:
    def test_same_doi_merged(self) -> None:
        refs = [
            _ref(id="a", doi="10.1234/test", source="s1.pdf"),
            _ref(id="b", doi="10.1234/test", source="s2.pdf"),
        ]
        result = merge_similar_references(refs, embedding_model="")
        assert len(result) == 1

    def test_same_label_merged(self) -> None:
        refs = [
            _ref(id="a", family="Smith", issued="2020", source="s1.pdf"),
            _ref(id="b", family="Smith", issued="2020", source="s2.pdf"),
        ]
        result = merge_similar_references(refs, embedding_model="")
        assert len(result) == 1

    def test_different_refs_not_merged(self) -> None:
        refs = [
            _ref(id="a", family="Smith", title="Paper A", source="s1.pdf"),
            _ref(id="b", family="Doe", title="Paper B", source="s2.pdf"),
        ]
        result = merge_similar_references(refs, embedding_model="")
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert merge_similar_references([], embedding_model="") == []

    def test_single_ref(self) -> None:
        refs = [_ref(id="a")]
        result = merge_similar_references(refs, embedding_model="")
        assert len(result) == 1

    def test_merged_ref_unions_sources(self) -> None:
        refs = [
            _ref(id="a", doi="10.1/x", source="s1.pdf"),
            _ref(id="b", doi="10.1/x", source="s2.pdf"),
        ]
        result = merge_similar_references(refs, embedding_model="")
        assert len(result) == 1
        merged = result[0]
        assert "s1.pdf" in merged.citing_source
        assert "s2.pdf" in merged.citing_source

    def test_merged_ref_unions_in_text_keys(self) -> None:
        r1 = _ref(id="a", doi="10.1/x")
        r1.in_text_keys = ["[1]", "[2]"]
        r2 = _ref(id="b", doi="10.1/x")
        r2.in_text_keys = ["[2]", "[3]"]
        result = merge_similar_references([r1, r2], embedding_model="")
        assert len(result) == 1
        merged = result[0]
        assert sorted(merged.in_text_keys) == ["[1]", "[2]", "[3]"]

    def test_merged_ref_picks_richest(self) -> None:
        """The reference with the most populated fields wins as base."""
        r1 = _ref(id="a", doi="10.1/x", title="T", family="Alpha", issued="2020")
        r2 = _ref(id="b", doi="10.1/x", title="T", family="Beta", issued="2021", container_title="Journal")
        result = merge_similar_references([r1, r2], embedding_model="")
        assert len(result) == 1
        assert result[0].container_title == "Journal"


# ---------------------------------------------------------------------------
# merge_similar_references — fuzzy embedding merge
# ---------------------------------------------------------------------------


class TestFuzzyMerge:
    def test_fuzzy_merge_similar_titles(self) -> None:
        refs = [
            _ref(id="a", title="Deep Learning for NLP", family="Alpha", issued="2020"),
            _ref(id="b", title="Deep Learning for Natural Language Processing", family="Beta", issued="2021"),
        ]
        with patch("kompend.core.pipeline.references.make_provider", return_value=_FakeProvider()):
            result = merge_similar_references(refs, embedding_model="fake-model", threshold=0.5)
        assert len(result) == 1

    def test_fuzzy_merge_disabled_without_model(self) -> None:
        refs = [
            _ref(id="a", title="Deep Learning for NLP", family="Alpha", issued="2020"),
            _ref(id="b", title="Deep Learning for Natural Language Processing", family="Beta", issued="2021"),
        ]
        result = merge_similar_references(refs, embedding_model="")
        assert len(result) == 2

    def test_fuzzy_merge_disabled_with_hash(self) -> None:
        refs = [
            _ref(id="a", title="Deep Learning for NLP", family="Alpha", issued="2020"),
            _ref(id="b", title="Deep Learning for Natural Language Processing", family="Beta", issued="2021"),
        ]
        result = merge_similar_references(refs, embedding_model="hash")
        assert len(result) == 2

    def test_fuzzy_merge_respects_threshold(self) -> None:
        """Very different titles should not merge even with embeddings."""
        refs = [
            _ref(id="a", title="Quantum Mechanics abc", family="Z"),
            _ref(id="b", title="Macroeconomics xyz", family="Y"),
        ]
        with patch("kompend.core.pipeline.references.make_provider", return_value=_FakeProvider()):
            result = merge_similar_references(refs, embedding_model="fake-model", threshold=0.99)
        assert len(result) == 2

    def test_fuzzy_merge_provider_failure_returns_exact_only(self) -> None:
        refs = [
            _ref(id="a", title="Paper A", family="Smith"),
            _ref(id="b", title="Paper B", family="Doe"),
        ]
        with patch("kompend.core.pipeline.references.make_provider", side_effect=RuntimeError("no model")):
            result = merge_similar_references(refs, embedding_model="fake-model")
        assert len(result) == 2

    def test_fuzzy_merge_embedding_failure_returns_exact_only(self) -> None:
        refs = [
            _ref(id="a", title="Paper A", family="Smith"),
            _ref(id="b", title="Paper B", family="Doe"),
        ]

        class _BrokenProvider:
            @property
            def dim(self) -> int:
                return 16

            def embed(self, text: str) -> list[float]:
                raise RuntimeError("embed failed")

        with patch("kompend.core.pipeline.references.make_provider", return_value=_BrokenProvider()):
            result = merge_similar_references(refs, embedding_model="fake-model")
        assert len(result) == 2

    def test_fuzzy_merge_combines_exact_and_fuzzy(self) -> None:
        """Exact dedup runs first, then fuzzy on remaining unique refs."""
        refs = [
            # Exact match by DOI
            _ref(id="a", doi="10.1/x", title="Paper X", family="Alpha", issued="2020", source="s1"),
            _ref(id="b", doi="10.1/x", title="Paper X", family="Beta", issued="2021", source="s2"),
            # Fuzzy match
            _ref(id="c", title="Attention Is All You Need", family="Vaswani", issued="2017"),
            _ref(id="d", title="Attention is All You Need", family="Other", issued="2018"),
        ]
        with patch("kompend.core.pipeline.references.make_provider", return_value=_FakeProvider()):
            result = merge_similar_references(refs, embedding_model="fake-model", threshold=0.9)
        # 4 → 2 (one exact pair + one fuzzy pair)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# consolidate_references
# ---------------------------------------------------------------------------


class TestConsolidateReferences:
    def test_consolidate_returns_flat_list(self) -> None:
        refs = [
            _ref(id="a", doi="10.1/x", source="s1"),
            _ref(id="b", doi="10.1/x", source="s2"),
            _ref(id="c", family="Doe", source="s3"),
        ]
        result = consolidate_references(refs, embedding_model="")
        assert len(result) == 2

    def test_consolidate_empty(self) -> None:
        assert consolidate_references([], embedding_model="") == []


# ---------------------------------------------------------------------------
# ReferenceStore.deduplicate
# ---------------------------------------------------------------------------


class TestStoreDeduplicate:
    def test_dedup_removes_exact_dupes(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("s1.pdf", [_ref(id="a", doi="10.1/x", family="Alpha", issued="2020")])
        store.add_source("s2.pdf", [_ref(id="b", doi="10.1/x", family="Beta", issued="2021")])
        removed = store.deduplicate()
        assert removed == 1
        assert store.total_count() == 1

    def test_dedup_no_op_without_dupes(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("s1.pdf", [_ref(id="a", family="Smith")])
        store.add_source("s2.pdf", [_ref(id="b", family="Doe")])
        removed = store.deduplicate()
        assert removed == 0

    def test_dedup_empty_store(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        removed = store.deduplicate()
        assert removed == 0

    def test_dedup_single_ref(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("s1.pdf", [_ref(id="a")])
        removed = store.deduplicate()
        assert removed == 0

    def test_dedup_preserves_sources_after_merge(self, tmp_path: Path) -> None:
        store = ReferenceStore(tmp_path / "refs.json")
        store.add_source("s1.pdf", [_ref(id="a", doi="10.1/x", family="Alpha", issued="2020", source="s1.pdf")])
        store.add_source("s2.pdf", [_ref(id="b", doi="10.1/x", family="Beta", issued="2021", source="s2.pdf")])
        store.deduplicate()
        merged_refs = store.all_references()
        assert len(merged_refs) == 1
        assert "s1.pdf" in merged_refs[0].citing_source
        assert "s2.pdf" in merged_refs[0].citing_source


# ---------------------------------------------------------------------------
# DuplicateReferencesRule with fuzzy matching
# ---------------------------------------------------------------------------


class TestLintRuleFuzzy:
    def test_lint_rule_has_embedding_model_attr(self) -> None:
        rule = DuplicateReferencesRule()
        assert rule.embedding_model == ""
        assert rule.fuzzy_threshold == 0.85

    def test_lint_rule_skip_no_sources_dir(self) -> None:
        rule = DuplicateReferencesRule()
        results = rule.check(_EMPTY_WIKI)
        assert len(results) == 1
        assert results[0].outcome == LintRuleOutcome.SKIP

    def test_lint_rule_skip_no_ref_file(self, tmp_path: Path) -> None:
        rule = DuplicateReferencesRule()
        results = rule.check(_EMPTY_WIKI, sources_dir=str(tmp_path / "src" / "doc.md"))
        assert len(results) == 1
        assert results[0].outcome == LintRuleOutcome.SKIP

    def test_lint_rule_pass_no_dupes(self, tmp_path: Path) -> None:
        kompend_dir = tmp_path / "src" / ".kompend"
        kompend_dir.mkdir(parents=True)
        store = ReferenceStore(kompend_dir / "references.json")
        store.add_source("s1.pdf", [_ref(id="a", family="Smith")])
        store.add_source("s2.pdf", [_ref(id="b", family="Doe")])
        store.save()

        src_dir = tmp_path / "src" / "doc.md"
        src_dir.write_text("test")

        rule = DuplicateReferencesRule()
        results = rule.check(_EMPTY_WIKI, sources_dir=str(src_dir))
        assert len(results) == 1
        assert results[0].outcome == LintRuleOutcome.PASS

    def test_lint_rule_fail_exact_dupes(self, tmp_path: Path) -> None:
        kompend_dir = tmp_path / "src" / ".kompend"
        kompend_dir.mkdir(parents=True)
        store = ReferenceStore(kompend_dir / "references.json")
        store.add_source("s1.pdf", [_ref(id="a", doi="10.1/x", family="Smith")])
        store.add_source("s2.pdf", [_ref(id="b", doi="10.1/x", family="Smith")])
        store.save()

        src_dir = tmp_path / "src" / "doc.md"
        src_dir.write_text("test")

        rule = DuplicateReferencesRule()
        results = rule.check(_EMPTY_WIKI, sources_dir=str(src_dir))
        assert len(results) == 1
        assert results[0].outcome == LintRuleOutcome.FAIL
        assert len(results[0].details.get("duplicates", [])) > 0

    def test_lint_rule_fuzzy_finds_near_dupes(self, tmp_path: Path) -> None:
        kompend_dir = tmp_path / "src" / ".kompend"
        kompend_dir.mkdir(parents=True)
        store = ReferenceStore(kompend_dir / "references.json")
        store.add_source(
            "s1.pdf",
            [
                _ref(id="a", title="Attention Is All You Need", family="Vaswani", issued="2017"),
            ],
        )
        store.add_source(
            "s2.pdf",
            [
                _ref(id="b", title="Attention is All You Need", family="Other", issued="2018"),
            ],
        )
        store.save()

        src_dir = tmp_path / "src" / "doc.md"
        src_dir.write_text("test")

        rule = DuplicateReferencesRule()
        rule.embedding_model = "fake-model"
        rule.fuzzy_threshold = 0.5

        with patch("kompend.core.pipeline.references.make_provider", return_value=_FakeProvider()):
            results = rule.check(_EMPTY_WIKI, sources_dir=str(src_dir))

        assert len(results) == 1
        assert results[0].outcome == LintRuleOutcome.FAIL
        near_dupes = results[0].details.get("near_duplicates", [])
        assert len(near_dupes) > 0
        assert near_dupes[0]["type"] == "fuzzy"


def _ref(  # noqa: PLR0913, PLR0917
    id: str = "smith2020",
    title: str = "A Paper",
    family: str = "Smith",
    issued: str = "2020",
    doi: str = "",
    source: str = "paper.pdf",
    container_title: str = "",
) -> Reference:
    return Reference(
        id=id,
        type="article-journal",
        title=title,
        author=[Author(family=family)],
        issued=issued,
        doi=doi,
        container_title=container_title,
        raw=f"{family}, J. ({issued}). {title}.",
        citing_source=source,
    )
