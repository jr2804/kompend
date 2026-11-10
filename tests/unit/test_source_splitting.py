"""Page-range splitting: plan invariants, decline paths, and the stage wiring.

The invariant under test is the one the design calls non-negotiable: a plan is a
partition of the document's pages — every page exactly once, in order, no gap,
no overlap — and a plan that violates it raises instead of being repaired. The
second invariant is the fail-closed one: any judge error, oversized page,
unpaged source or unusable plan leaves the source whole, and the ledger says
why.
"""

# ruff: noqa: S608 — SQL is assembled only from the imported table-name constant.

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import _output
from pydantic_ai.models import decision as decision_grammar

from kompend.chunking.hierarchical import HierarchicalChunker
from kompend.core.cache.store import CacheStore
from kompend.core.config.models.decision import DecisionConfig
from kompend.core.config.models.splitting import SplittingConfig
from kompend.core.models import (
    ChunkTree,
    DocumentElement,
    ElementType,
    ParsedDocument,
)
from kompend.core.pipeline.config import Compiler, EmbeddingConfig, LLMConfig, PipelineConfig
from kompend.core.pipeline.ingest import PARSE_CACHE_PARSER_KEY
from kompend.core.pipeline.parse_cache import cache_key, write_document
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.pipeline.stages.factory import build_pipeline
from kompend.core.pipeline.stages.parse_sources import _process_single_source  # noqa: PLC2701
from kompend.core.pipeline.state import TierState, TierStateStore
from kompend.core.reporting.ledger import stage_ledger
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import STAGES_TABLE
from kompend.core.templates import get_template_loader
from kompend.core.types import ReasonCode, StageOutcome
from kompend.llm.context_window import ContextSource, ContextWindow
from kompend.preprocessing.host_memory import HeavyParseGuard
from kompend.preprocessing.splitting import (
    LEDGER_DETAIL_CHARS,
    DecisionModelSplitJudge,
    NeverSplit,
    PageText,
    SourceSplitter,
    SplitDecision,
    SplitJudgeDeps,
    SplitJudgeUnavailable,
    SplitMetadata,
    SplitPlan,
    SplitPlanError,
    SplitReason,
    SplitRule,
    SplitRulesError,
    SplitSegment,
    attachment_owner_stem,
    boundary_field_name,
    boundary_verdict_model,
    build_splitter,
    child_source_path,
    load_rules,
    page_range,
    page_texts,
    parent_source_of,
    parse_rules,
    partition_document,
    plan_from_verdict,
    render_pages,
    render_rules,
    spec_name,
    validate_plan,
)
from kompend.types import JSONDict, JSONList
from tests.conftest import make_tmp_vault

_PAGE_MARKER = re.compile(r"page (\d+) text")


class _FixedJudge:
    """Judge double returning a canned plan (or raising a canned error)."""

    def __init__(self, segments: list[SplitSegment] | None = None, *, error: Exception | None = None) -> None:
        self.segments = segments
        self.error = error
        self.calls: list[Sequence[PageText]] = []

    def plan_pages(self, page_texts: Sequence[PageText], rules: Sequence[SplitRule]) -> SplitPlan:  # noqa: ARG002
        self.calls.append(list(page_texts))
        if self.error is not None:
            raise self.error
        assert self.segments is not None
        return SplitPlan(segments=tuple(self.segments), decided_by="fixed")


# ── Stage wiring: one tree per document ────────────────────────────────


class _StubParser:
    """Parser facade: reports the identity the seeded parse cache carries."""

    def __init__(self, identity: str) -> None:
        self._identity = identity

    def backend_identity(self) -> str:
        return self._identity

    def route(self, source_file: Path) -> None:  # noqa: ARG002 — nothing to parse in these tests
        return None


class _StubRuntime:
    """Minimal PipelineRuntime covering the chunk/cache/enrich path."""

    def __init__(self, base: Path, *, parser_identity: str | None = None) -> None:
        self.base_directory = base
        self.config = SimpleNamespace(
            no_cache=False,
            reparse=False,
            force=False,
            privacy_filter=False,
            output=SimpleNamespace(rumdl_config=None, flavor=None),
            search=SimpleNamespace(contextual=SimpleNamespace(enabled=False)),
        )
        self.parser = _StubParser(parser_identity) if parser_identity else None
        self._parser_identity = parser_identity
        self.chunker = HierarchicalChunker(max_tokens=512)
        self.enricher = None
        self.state = TierState()
        self.cache_store = CacheStore(base / "cache", base)
        self.state_store = TierStateStore(base / "compile_state.json", base)

    def rel_file(self, file: Path) -> str:
        return str(file)

    def cache_directory(self) -> Path:
        return self.base_directory

    def apply_tags_to_tree(self, tree: ChunkTree, source_file: Path) -> None:  # noqa: ARG002
        return


# ── The verdict the judge is asked for ─────────────────────────────────


class _Grammar:
    """The decision grammar's own view of a verdict model (no endpoint)."""

    LIMITS = decision_grammar._Limits(choice_options=None, score_levels=None, requires_instructions=False, questions=None)

    @classmethod
    def questions(cls, model: type[BaseModel]) -> dict[str, Any]:
        tool = cls.tool(model)
        assert decision_grammar._expressible(tool, None, cls.LIMITS) is True
        return dict(decision_grammar._questions(decision_grammar._fields(tool), tool, None, cls.LIMITS, label=None))

    @staticmethod
    def tool(model: type[BaseModel]) -> Any:
        toolset = _output.OutputToolset.build(cast("list[Any]", [model]))
        assert toolset is not None
        return toolset._tool_defs[0]


# ── validate_plan: the partition invariant ─────────────────────────────


def test_validate_plan_accepts_a_full_partition() -> None:
    plan = SplitPlan(segments=(SplitSegment(1, 4), SplitSegment(5, 7), SplitSegment(8, 9)))
    validate_plan(plan, first_page=1, last_page=9)


def test_validate_plan_rejects_a_gap() -> None:
    plan = SplitPlan(segments=(SplitSegment(1, 4), SplitSegment(6, 9)))
    with pytest.raises(SplitPlanError, match="skips page 5"):
        validate_plan(plan, first_page=1, last_page=9)


def test_validate_plan_rejects_an_overlap() -> None:
    plan = SplitPlan(segments=(SplitSegment(1, 5), SplitSegment(5, 9)))
    with pytest.raises(SplitPlanError, match="overlaps page 6"):
        validate_plan(plan, first_page=1, last_page=9)


def test_validate_plan_rejects_a_short_tail() -> None:
    plan = SplitPlan(segments=(SplitSegment(1, 4), SplitSegment(5, 8)))
    with pytest.raises(SplitPlanError, match="stops at page 8"):
        validate_plan(plan, first_page=1, last_page=9)


def test_validate_plan_rejects_out_of_range_and_inverted_segments() -> None:
    with pytest.raises(SplitPlanError, match="outside 1..9"):
        validate_plan(SplitPlan(segments=(SplitSegment(1, 10),)), first_page=1, last_page=9)
    with pytest.raises(SplitPlanError, match="ends before it starts"):
        validate_plan(SplitPlan(segments=(SplitSegment(4, 2),)), first_page=1, last_page=9)


def test_validate_plan_rejects_an_empty_plan() -> None:
    with pytest.raises(SplitPlanError, match="no segments"):
        validate_plan(SplitPlan(segments=()), first_page=1, last_page=9)


def test_validate_plan_accepts_a_single_page_range() -> None:
    validate_plan(SplitPlan(segments=(SplitSegment(3, 3),)), first_page=3, last_page=3)


# ── Page assembly ──────────────────────────────────────────────────────


def test_page_range_is_none_when_any_element_is_unpaged() -> None:
    document = _document(3)
    document.elements.append(_element(None, "preamble"))
    assert page_range(document) is None


def test_page_range_is_none_without_elements() -> None:
    assert page_range(ParsedDocument(source_file=Path("sources/empty.pdf"), elements=[])) is None


def test_page_texts_are_contiguous_and_carry_the_full_page() -> None:
    document = _document(3)
    long_text = "x" * 5000
    document.elements.append(_element(2, long_text))
    texts = page_texts(document, first_page=1, last_page=3)

    assert [page.page_number for page in texts] == [1, 2, 3]
    # Never cropped: the oversized page arrives whole, so the caller can refuse.
    assert long_text in texts[1].text


def test_page_texts_include_empty_pages() -> None:
    document = ParsedDocument(source_file=Path("sources/packet.pdf"), elements=[_element(1, "a"), _element(3, "c")])
    texts = page_texts(document, first_page=1, last_page=3)
    assert [page.text for page in texts] == ["a", "", "c"]


# ── Child identity ─────────────────────────────────────────────────────


def test_child_source_path_keeps_the_range_in_the_stem() -> None:
    child = child_source_path(Path("sources/packet.pdf"), SplitSegment(5, 9))
    assert child.name == "packet#p005-009.pdf"
    assert child.stem == "packet#p005-009"
    assert child.parent.as_posix() == "sources"


def test_children_of_one_packet_have_distinct_stems() -> None:
    parent = Path("sources/packet.pdf")
    stems = {child_source_path(parent, segment).stem for segment in (SplitSegment(1, 4), SplitSegment(5, 9))}
    assert len(stems) == 2
    assert all(stem.startswith("packet#") for stem in stems)


def test_partition_assigns_every_element_exactly_once_and_in_order() -> None:
    document = _document(9)
    segments = (SplitSegment(1, 4), SplitSegment(5, 6), SplitSegment(7, 9))

    children = partition_document(document, segments, parent_source=document.source_file, decided_by="fixed", rules_sha256="abc")

    pages = [[element.page for element in child.elements] for child in children]
    assert pages == [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
    assert sum(len(child.elements) for child in children) == len(document.elements)


def test_partition_stamps_parent_provenance_on_every_child() -> None:
    document = _document(4)
    children = partition_document(
        document,
        (SplitSegment(1, 2, category="standard"), SplitSegment(3, 4, category="amendment")),
        parent_source=document.source_file,
        decided_by="jev",
        rules_sha256="sha-rules",
    )

    first, second = children
    assert first.metadata[SplitMetadata.PARENT_STEM.value] == "packet"
    assert first.metadata[SplitMetadata.PARENT_SOURCE.value] == "sources/packet.pdf"
    assert first.metadata[SplitMetadata.PAGES.value] == "p001-002"
    assert first.metadata[SplitMetadata.CATEGORY.value] == "standard"
    assert second.metadata[SplitMetadata.CATEGORY.value] == "amendment"
    assert first.metadata[SplitMetadata.DECIDED_BY.value] == "jev"
    assert first.metadata[SplitMetadata.RULES_SHA256.value] == "sha-rules"
    # Page count is the child's own range, and every element points at the child.
    assert (first.page_count, second.page_count) == (2, 2)
    assert all(element.source_file == first.source_file for element in first.elements)


def test_partition_scopes_the_parent_title_by_page_range() -> None:
    children = partition_document(
        _document(4),
        (SplitSegment(1, 2), SplitSegment(3, 4)),
        parent_source=Path("sources/packet.pdf"),
        decided_by="fixed",
        rules_sha256="",
    )
    assert children[0].title == "Packet [p001-002]"


def test_partition_leaves_the_parent_document_untouched() -> None:
    """Cutting must not repoint the parent's (or a sibling's) elements.

    Children are cut from one parsed document, so in-place rehoming would leave
    every element pointing at whichever child was built last — the parent would
    then cite pages it no longer owns.
    """
    document = _document(4)
    parent_source = document.source_file
    originals = list(document.elements)
    original_metadata = [dict(element.metadata) for element in originals]

    children = partition_document(
        document,
        (SplitSegment(1, 2), SplitSegment(3, 4)),
        parent_source=parent_source,
        decided_by="fixed",
        rules_sha256="",
    )

    assert document.elements == originals
    assert [element.source_file for element in document.elements] == [None] * len(originals)
    assert [element.metadata for element in document.elements] == original_metadata
    # Each child's elements point at *its* source, and no element is shared.
    for child in children:
        assert {element.source_file for element in child.elements} == {child.source_file}
    ids = [id(element) for child in children for element in child.elements]
    assert len(set(ids)) == len(ids)


def test_partition_refuses_to_drop_elements() -> None:
    """A segment list that does not cover the document is a bug, not a trim."""
    document = _document(4)
    with pytest.raises(SplitPlanError, match="partition lost"):
        partition_document(
            document,
            (SplitSegment(1, 2),),
            parent_source=document.source_file,
            decided_by="fixed",
            rules_sha256="",
        )


# ── decide(): every decline path is explicit ────────────────────────────


def test_decide_declines_an_unpaged_document() -> None:
    document = _document(8)
    document.elements.append(_element(None, "frontmatter"))

    decision = SourceSplitter(judge=_FixedJudge([SplitSegment(1, 4), SplitSegment(5, 8)])).decide(document)

    assert decision.reason is SplitReason.UNPAGED
    assert not decision.applies


def test_decide_declines_below_the_page_floor_without_calling_the_judge() -> None:
    judge = _FixedJudge([SplitSegment(1, 2), SplitSegment(3, 4)])
    decision = SourceSplitter(judge=judge, min_pages=6).decide(_document(4))

    assert decision.reason is SplitReason.TOO_FEW_PAGES
    assert judge.calls == []


def test_decide_declines_an_oversized_page_without_cropping_it() -> None:
    document = _document(8)
    document.elements.append(_element(3, "y" * 64))
    judge = _FixedJudge([SplitSegment(1, 4), SplitSegment(5, 8)])

    decision = SourceSplitter(judge=judge, min_pages=2, max_page_chars=32).decide(document)

    assert decision.reason is SplitReason.PAGE_OVERSIZED
    assert "page 3" in decision.detail
    assert judge.calls == []


def test_decide_declines_when_the_judge_raises() -> None:
    judge = _FixedJudge(error=RuntimeError("model unreachable"))
    decision = SourceSplitter(judge=judge, min_pages=2).decide(_document(8))

    assert decision.reason is SplitReason.JUDGE_FAILED
    assert "model unreachable" in decision.detail


def test_decide_declines_a_plan_that_is_not_a_partition() -> None:
    judge = _FixedJudge([SplitSegment(1, 4), SplitSegment(6, 8)])
    decision = SourceSplitter(judge=judge, min_pages=2).decide(_document(8))

    assert decision.reason is SplitReason.PLAN_INVALID
    assert "skips page 5" in decision.detail


def test_decide_reports_no_boundary_for_a_single_segment_plan() -> None:
    decision = SourceSplitter(judge=NeverSplit(), min_pages=2).decide(_document(8))

    assert decision.reason is SplitReason.NO_BOUNDARY
    assert decision.decided_by == "NeverSplit"


def test_decide_applies_a_valid_plan_and_summarises_it() -> None:
    judge = _FixedJudge([SplitSegment(1, 4, category="standard"), SplitSegment(5, 8, category="amendment")])

    decision = SourceSplitter(judge=judge, min_pages=2).decide(_document(8))

    assert decision.applies
    assert decision.reason is SplitReason.APPLIED
    assert decision.summary() == "1->2: p001-004 standard, p005-008 amendment"


def test_ledger_detail_fits_the_columns_budget() -> None:
    """A many-document packet must not lose its boundaries to the ledger's cut.

    ``stages.detail`` keeps 240 characters and cuts the rest, so the split has
    to compress its own account: first and last segment stay named, the middle
    is counted, and the full list remains available for the run log.
    """
    segments = tuple(SplitSegment(start, start + 9) for start in range(1, 200, 10))
    decision = SourceSplitter(judge=_FixedJudge(list(segments)), min_pages=2).decide(_document(200))

    detail = decision.ledger_detail()

    assert len(detail) <= LEDGER_DETAIL_CHARS < len(decision.summary())
    assert detail.startswith("1->20: p001-010")
    assert "p191-200" in detail
    assert "18 between" in detail


def test_ledger_detail_lists_a_small_cut_in_full() -> None:
    judge = _FixedJudge([SplitSegment(1, 4, category="standard"), SplitSegment(5, 8, category="amendment")])

    decision = SourceSplitter(judge=judge, min_pages=2).decide(_document(8))

    assert decision.ledger_detail() == "1->2: p001-004 standard, p005-008 amendment"


def test_ledger_detail_names_the_decline_and_its_reason() -> None:
    decision = SourceSplitter(judge=_FixedJudge(error=RuntimeError("judge down")), min_pages=2).decide(_document(8))

    detail = decision.ledger_detail()

    assert detail.startswith("judge_failed: _FixedJudge: RuntimeError: judge down")
    assert detail == f"judge_failed: {decision.detail}"


def test_ledger_detail_cuts_a_long_decline_on_a_word_boundary() -> None:
    reason = "word " * 100
    decision = SplitDecision(reason=SplitReason.PLAN_INVALID, detail=reason.strip())

    detail = decision.ledger_detail()

    assert len(detail) <= LEDGER_DETAIL_CHARS
    assert detail.endswith("…")
    assert not detail[:-1].endswith(" ")
    assert detail.endswith("word…")


def test_decide_never_sends_a_cropped_page_to_the_judge() -> None:
    document = _document(3)
    long_page = "z" * 4000
    document.elements.append(_element(2, long_page))
    judge = _FixedJudge([SplitSegment(1, 2), SplitSegment(3, 3)])

    SourceSplitter(judge=judge, min_pages=2, max_page_chars=10_000).decide(document)

    assert long_page in judge.calls[0][1].text


def test_no_splitter_returns_exactly_one_tree(tmp_path: Path) -> None:
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)

    with stage_ledger(vault_db_path(base)) as ledger:
        trees = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain")

    assert len(trees) == 1
    assert trees[0].source_file == source


def test_no_splitter_writes_no_split_ledger_row(tmp_path: Path) -> None:
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)

    with stage_ledger(vault_db_path(base)) as ledger:
        _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain")

    assert _ledger_rows(base, stage="split") == []


def test_injected_splitter_produces_one_tree_per_child(tmp_path: Path) -> None:
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)
    splitting = SourceSplitter(judge=_FixedJudge([SplitSegment(1, 5), SplitSegment(6, 8)]), min_pages=2)

    with stage_ledger(vault_db_path(base)) as ledger:
        trees = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)

    assert [tree.source_file.name if tree.source_file else "" for tree in trees] == [
        "packet#p001-005.md",
        "packet#p006-008.md",
    ]
    # Each child carries only its own pages, and its provenance names the parent.
    assert _pages(trees[0]) == [1, 2, 3, 4, 5]
    assert _pages(trees[1]) == [6, 7, 8]


def test_split_is_recorded_in_the_ledger_with_its_boundary(tmp_path: Path) -> None:
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)
    splitting = SourceSplitter(judge=_FixedJudge([SplitSegment(1, 5), SplitSegment(6, 8)]), min_pages=2)

    with stage_ledger(vault_db_path(base)) as ledger:
        _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)

    rows = _ledger_rows(base, stage="split")
    assert len(rows) == 1
    assert rows[0]["outcome"] == StageOutcome.OK.value
    assert "p001-005" in rows[0]["detail"]
    assert "p006-008" in rows[0]["detail"]


def test_declined_split_leaves_the_source_whole_and_says_why(tmp_path: Path) -> None:
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)
    splitting = SourceSplitter(judge=_FixedJudge(error=RuntimeError("judge down")), min_pages=2)

    with stage_ledger(vault_db_path(base)) as ledger:
        trees = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)

    assert [tree.source_file for tree in trees] == [source]
    assert _pages(trees[0]) == list(range(1, 9))
    rows = _ledger_rows(base, stage="split")
    assert len(rows) == 1
    assert rows[0]["outcome"] == StageOutcome.DEGRADED.value
    assert rows[0]["reason_code"] == ReasonCode.JUDGE_DEGRADED.value
    assert "judge down" in rows[0]["detail"]


def test_normal_decline_is_skipped_not_degraded(tmp_path: Path) -> None:
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)
    splitting = SourceSplitter(judge=NeverSplit(), min_pages=2)

    with stage_ledger(vault_db_path(base)) as ledger:
        _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)

    rows = _ledger_rows(base, stage="split")
    assert len(rows) == 1
    assert rows[0]["outcome"] == StageOutcome.SKIPPED.value
    assert "no_boundary" in rows[0]["detail"]


def test_children_get_independent_chunk_cache_entries(tmp_path: Path) -> None:
    """Siblings share the parent's content hash; their identities must not collide."""
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)
    splitting = SourceSplitter(judge=_FixedJudge([SplitSegment(1, 4), SplitSegment(5, 8)]), min_pages=2)

    with stage_ledger(vault_db_path(base)) as ledger:
        first = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)
        second = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)

    assert _pages(first[0]) != _pages(first[1])
    # The second run is a cache hit: same documents, same order.
    assert [tree.source_file for tree in second] == [tree.source_file for tree in first]
    assert _pages(second[0]) == _pages(first[0])

    rows = _ledger_rows(base, stage="chunk")
    assert [row["outcome"] for row in rows[-2:]] == [StageOutcome.CACHED.value, StageOutcome.CACHED.value]


def test_skip_path_returns_an_empty_list(tmp_path: Path) -> None:
    base = tmp_path / "vault"
    base.mkdir(parents=True)
    source = base / "unparsedable.zzz"
    source.write_text("bytes", encoding="utf-8")
    runtime = _StubRuntime(base)

    with stage_ledger(vault_db_path(base)) as ledger:
        assert _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, None) == []


def _pages(tree: ChunkTree) -> list[int]:
    """Page numbers of every page marker in the tree, in reading order.

    Chunks carry no page field, so the fixture's "page N text" markers are the
    only page evidence available downstream of chunking.
    """
    numbers: list[int] = []
    for leaf in tree.leaves():
        if leaf.chunk is None:
            continue
        numbers.extend(int(found) for found in _PAGE_MARKER.findall(leaf.chunk.text))
    return numbers


def _ledger_rows(base: Path, *, stage: str) -> list[Any]:
    with open_store(vault_db_path(base)) as conn:
        return conn.execute(f"SELECT * FROM {STAGES_TABLE} WHERE stage = ? ORDER BY id", (stage,)).fetchall()


def test_the_verdict_asks_one_yes_no_question_per_candidate_cut_point() -> None:
    """Seven pages have six seams, and the decision grammar carries each as a bool."""
    questions = _Grammar.questions(boundary_verdict_model(7))

    assert list(questions) == [boundary_field_name(index) for index in range(1, 7)]
    assert {question.type for question in questions.values()} == {"noul"}


def test_the_first_page_is_never_a_candidate() -> None:
    """A packet's first page starts its first document, so there is no seam before it."""
    assert list(_Grammar.questions(boundary_verdict_model(2))) == [boundary_field_name(1)]


def test_a_page_count_with_no_seam_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two pages"):
        boundary_verdict_model(1)


def test_a_batched_list_of_verdicts_would_be_refused_by_the_grammar() -> None:
    """Why the verdict is one bool per cut point and not a list of segment models."""

    class _BatchedSegments(BaseModel):
        """One model per segment."""

        segments: list[SplitSegment] = Field(description="One model per segment.")

    tool = _Grammar.tool(_BatchedSegments)
    assert decision_grammar._expressible(tool, None, _Grammar.LIMITS) is False
    with pytest.raises(Exception, match="list must be of two or more string options"):
        decision_grammar._questions(decision_grammar._fields(tool), tool, None, _Grammar.LIMITS, label=None)


def test_plan_from_verdict_closes_a_segment_at_every_answered_seam() -> None:
    answers = {boundary_field_name(index): index in {2, 4} for index in range(1, 6)}
    verdict = boundary_verdict_model(6)(**answers)

    plan = plan_from_verdict(verdict, page_numbers=[1, 2, 3, 4, 5, 6], decided_by="jev", rules_sha256="abc")

    assert [(segment.start_page, segment.end_page) for segment in plan.segments] == [(1, 2), (3, 4), (5, 6)]
    assert plan.decided_by == "jev"
    assert plan.rules_sha256 == "abc"


def test_an_unanswered_seam_keeps_the_document_whole() -> None:
    """A missing answer is not a boundary: the safest reading is continuation."""
    verdict = boundary_verdict_model(4).model_construct(None, **{boundary_field_name(2): True})

    plan = plan_from_verdict(verdict, page_numbers=[1, 2, 3, 4], decided_by="jev", rules_sha256="")

    assert [(segment.start_page, segment.end_page) for segment in plan.segments] == [(1, 2), (3, 4)]


def test_a_plan_from_the_verdict_always_partitions_the_packet() -> None:
    verdict = boundary_verdict_model(5)(**{boundary_field_name(index): index % 2 == 0 for index in range(1, 5)})

    plan = plan_from_verdict(verdict, page_numbers=[1, 2, 3, 4, 5], decided_by="jev", rules_sha256="")

    validate_plan(plan, first_page=1, last_page=5)


def test_the_judge_sees_every_page_under_its_own_marker() -> None:
    pages = [PageText(page_number=number, text=f"text of {number}") for number in (1, 2, 3)]

    rendered = render_pages(pages)

    assert "<<< page 1 >>>\ntext of 1" in rendered
    assert "<<< page 3 >>>\ntext of 3" in rendered


def test_the_rules_are_rendered_as_the_categories_the_judge_may_use() -> None:
    rules = (SplitRule(id="contract", description="An agreement."), SplitRule(id="amendment", description="Changes a contract."))

    assert render_rules(rules).splitlines() == ["- contract: An agreement.", "- amendment: Changes a contract."]
    assert render_rules(()) == "(no categories configured)"


def test_the_spec_is_named_by_the_number_of_questions_it_asks() -> None:
    """Materialised agent classes are cached per name, so the name carries the shape."""
    assert spec_name(9) == spec_name(9)
    assert spec_name(9) != spec_name(10)


# ── The judge templates ────────────────────────────────────────────────


def test_the_judge_templates_resolve_and_render() -> None:
    """The judge lives outside ``kompend.agents``, so its templates need their own guard."""
    loader = get_template_loader()
    deps = SplitJudgeDeps(source_id="split-judge", pages="<<< page 1 >>>\nbody", rules="- contract: An agreement.")

    body = loader.render("decision/split_boundary.md.j2", pages=deps.pages, rules=deps.rules)

    assert "<<< page 1 >>>" in body
    assert "- contract: An agreement." in body
    assert "one typed question per page transition" in body
    assert loader.get_body("decision/split_boundary_system.md.j2").strip()


def test_the_judge_templates_declare_only_what_the_questions_carry() -> None:
    meta = get_template_loader().get_meta("decision/split_boundary.md.j2")
    assert {variable["name"] for variable in meta.variables} == {"pages", "rules"}


# ── Rules ──────────────────────────────────────────────────────────────


def test_the_built_in_rules_load_from_the_shipped_resource() -> None:
    rules = load_rules()

    ids = [rule.id for rule in rules]
    assert "contract" in ids
    assert "amendment" in ids
    assert len(set(ids)) == len(ids)
    assert all(rule.description for rule in rules)


def test_rules_from_an_explicit_file_win(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("categories:\n  - id: memo\n    description: An internal memo.\n", encoding="utf-8")

    assert [rule.id for rule in load_rules(path)] == ["memo"]


def test_an_unreadable_rules_file_is_an_error_not_an_empty_list(tmp_path: Path) -> None:
    with pytest.raises(SplitRulesError, match="Cannot read splitting rules"):
        load_rules(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ("categories: []", "non-empty 'categories' list"),
        ("options: []", "unknown key\\(s\\): options"),
        ("categories:\n  - description: no id\n", "needs an 'id'"),
        ("categories:\n  - id: Bad Id\n    description: x\n", "needs an 'id'"),
        ("categories:\n  - id: memo\n    description: x\n  - id: memo\n    description: y\n", "repeats the id 'memo'"),
        ("categories:\n  - id: memo\n    description: '   '\n", "needs a non-empty 'description'"),
        ("categories:\n  - id: memo\n    description: x\n    colour: red\n", "unknown key\\(s\\): colour"),
    ],
)
def test_a_rules_file_that_is_half_understood_is_refused(document: str, match: str) -> None:
    with pytest.raises(SplitRulesError, match=match):
        parse_rules(yaml.safe_load(document), origin="inline")


# ── Building the splitter from settings ────────────────────────────────


def test_splitting_is_off_until_the_block_says_otherwise() -> None:
    assert build_splitter(SplittingConfig(), DecisionConfig(), base_dir=Path("vault")) is None


def test_an_enabled_block_without_a_model_still_wires_a_judge_that_declines() -> None:
    """An operator who turned splitting on learns per source why nothing happened."""
    splitter = build_splitter(SplittingConfig(enabled=True), DecisionConfig(), base_dir=Path("vault"))

    assert splitter is not None
    decision = splitter.decide(_document(8))
    assert decision.reason is SplitReason.JUDGE_FAILED
    assert "splitting.judge_model" in decision.detail
    with pytest.raises(SplitJudgeUnavailable):
        splitter.judge.plan_pages(page_texts(_document(8), first_page=1, last_page=8), ())


def _document(pages: int, *, base: Path = Path("sources/packet.pdf"), text: str = "body") -> ParsedDocument:
    elements = [_element(page, f"{text} p{page}") for page in range(1, pages + 1)]
    return ParsedDocument(source_file=base, elements=elements, title="Packet", page_count=pages)


# ── Fixtures: documents ────────────────────────────────────────────────


def _element(page: int | None, text: str = "", *, kind: ElementType = ElementType.TEXT) -> DocumentElement:
    return DocumentElement(type=kind, content=text, page=page, reading_order=page)


def test_the_judge_model_falls_back_to_the_decision_section() -> None:
    splitter = build_splitter(
        SplittingConfig(enabled=True),
        DecisionConfig(enabled=True, model="typesafe/jev-latest"),
        base_dir=Path("vault"),
    )

    assert splitter is not None
    assert isinstance(splitter.judge, DecisionModelSplitJudge)
    assert splitter.judge.model_string == "typesafe/jev-latest"


def test_a_judge_model_of_its_own_wins_over_the_decision_section() -> None:
    splitter = build_splitter(
        SplittingConfig(enabled=True, judge_model="ollama/clef"),
        DecisionConfig(enabled=True, model="typesafe/jev-latest"),
        base_dir=Path("vault"),
    )

    assert splitter is not None
    assert isinstance(splitter.judge, DecisionModelSplitJudge)
    assert splitter.judge.model_string == "ollama/clef"


def test_the_settings_reach_the_splitter(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("categories:\n  - id: memo\n    description: An internal memo.\n", encoding="utf-8")

    splitter = build_splitter(
        SplittingConfig(enabled=True, judge_model="ollama/clef", rules_file="rules.yaml", min_pages=4, max_page_chars=900),
        DecisionConfig(),
        base_dir=tmp_path,
    )

    assert splitter is not None
    assert (splitter.min_pages, splitter.max_page_chars) == (4, 900)
    assert [rule.id for rule in splitter.rules] == ["memo"]


# ── Factory wiring ─────────────────────────────────────────────────────


def test_the_factory_leaves_splitting_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _built_pipeline(monkeypatch, PipelineConfig(**_factory_kwargs()))

    assert pipeline.splitter is None


def test_the_factory_hands_an_enabled_splitter_to_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PipelineConfig(**_factory_kwargs(splitting=SplittingConfig(enabled=True, judge_model="ollama/clef")))

    pipeline = _built_pipeline(monkeypatch, config)

    assert isinstance(pipeline.splitter, SourceSplitter)
    assert pipeline.splitter.judge.model_string == "ollama/clef"


def _factory_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "source_dir": make_tmp_vault(),
        "output_dir": make_tmp_vault(),
        "llm": LLMConfig(default_model="ollama/gpt-oss:20b", vlm_model="vlm"),
        "embedding": EmbeddingConfig(model="hash"),
        "compiler": Compiler.GRAPH,
    }
    kwargs.update(overrides)
    return kwargs


def _built_pipeline(monkeypatch: pytest.MonkeyPatch, config: PipelineConfig) -> Any:
    """Build through the real factory without touching a model."""
    monkeypatch.setattr(
        "kompend.core.pipeline.stages.factory.discover_context_window",
        lambda *a, **k: ContextWindow(max_input=8192, max_output=2048, source=ContextSource.FALLBACK),
    )
    return build_pipeline(config)


# ── Provenance a split stamps onto its children ────────────────────────


def test_a_child_names_the_parent_whose_attachments_it_keeps() -> None:
    child = Path("sources/packet#p005-009.pdf")
    stamped = {SplitMetadata.PARENT_STEM.value: "packet", SplitMetadata.PARENT_SOURCE.value: "sources/packet.pdf"}

    assert attachment_owner_stem(child, stamped) == "packet"
    assert parent_source_of(stamped) == Path("sources/packet.pdf")


def test_an_ordinary_source_owns_its_own_attachments() -> None:
    assert attachment_owner_stem(Path("sources/report.pdf"), {}) == "report"
    assert attachment_owner_stem(None, {}) is None
    assert parent_source_of({}) is None
    assert parent_source_of(None) is None


def test_a_child_chunk_carries_the_child_path_and_the_parent_link(tmp_path: Path) -> None:
    """Extraction and export read the chunk, so the provenance must reach it."""
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)
    splitting = SourceSplitter(judge=_FixedJudge([SplitSegment(1, 5), SplitSegment(6, 8)]), min_pages=2)

    with stage_ledger(vault_db_path(base)) as ledger:
        trees = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)

    first = trees[0]
    leaves = [leaf for leaf in first.leaves() if leaf.chunk is not None]
    assert leaves
    for leaf in leaves:
        assert leaf.chunk is not None
        # The figure of a child's page must match a provenance entry, and
        # provenance is keyed by the child's own stem.
        assert leaf.chunk.source_file == first.source_file
        assert leaf.chunk.source_file is not None
        assert leaf.chunk.source_file.name == "packet#p001-005.md"
        assert leaf.chunk.metadata[SplitMetadata.PARENT_STEM.value] == "packet"
        assert leaf.chunk.metadata[SplitMetadata.PARENT_SOURCE.value] == source.as_posix()
        assert leaf.chunk.metadata[SplitMetadata.PAGES.value] == "p001-005"


def test_an_unsplit_source_gains_no_split_metadata(tmp_path: Path) -> None:
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)

    with stage_ledger(vault_db_path(base)) as ledger:
        trees = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain")

    keys = {str(key.value) for key in SplitMetadata}
    for leaf in trees[0].leaves():
        if leaf.chunk is not None:
            assert not keys & set(leaf.chunk.metadata)


def test_a_child_keeps_its_provenance_through_the_chunk_cache(tmp_path: Path) -> None:
    """The second run rebuilds the tree from cache; the provenance must survive."""
    base = tmp_path / "vault"
    runtime, source = _split_runtime(base)
    splitting = SourceSplitter(judge=_FixedJudge([SplitSegment(1, 5), SplitSegment(6, 8)]), min_pages=2)

    with stage_ledger(vault_db_path(base)) as ledger:
        _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)
        cached = _process_single_source(cast("PipelineRuntime", runtime), source, {}, HeavyParseGuard(), ledger, "test-chain", splitting=splitting)

    leaves = [leaf for leaf in cached[0].leaves() if leaf.chunk is not None]
    assert leaves
    assert all(leaf.chunk.metadata[SplitMetadata.PARENT_STEM.value] == "packet" for leaf in leaves if leaf.chunk is not None)


def _split_runtime(base: Path) -> tuple[Any, Path]:
    runtime = _StubRuntime(base, parser_identity="test-chain")
    source, _ = _seed(base, 8)
    return runtime, source


def _seed(base: Path, pages: int, *, stem: str = "packet") -> tuple[Path, str]:
    """Create the source file and a current parse-cache entry with *pages* pages."""
    base.mkdir(parents=True, exist_ok=True)
    source = base / f"{stem}.md"
    source.write_text("# packet\n", encoding="utf-8")
    content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    elements = [
        {
            "type": str(ElementType.TEXT),
            "content": f"page {page} text",
            "page": page,
            "reading_order": page,
            "heading_level": None,
            "metadata": {},
        }
        for page in range(1, pages + 1)
    ]
    payload: JSONDict = {
        "source_file": str(source),
        PARSE_CACHE_PARSER_KEY: "test-chain",
        "title": "Packet",
        "author": "",
        "language": "en",
        "page_count": pages,
        "metadata": {},
        "elements": cast("JSONList", elements),
    }
    write_document(vault_db_path(base), cache_key(source, content_hash), payload)
    return source, content_hash
