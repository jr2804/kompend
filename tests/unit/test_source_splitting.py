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

from kompend.chunking.hierarchical import HierarchicalChunker
from kompend.core.cache.store import CacheStore
from kompend.core.models import (
    ChunkTree,
    DocumentElement,
    ElementType,
    ParsedDocument,
)
from kompend.core.pipeline.ingest import PARSE_CACHE_PARSER_KEY
from kompend.core.pipeline.parse_cache import cache_key, write_document
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.pipeline.stages.parse_sources import _process_single_source  # noqa: PLC2701
from kompend.core.pipeline.state import TierState, TierStateStore
from kompend.core.reporting.ledger import stage_ledger
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import STAGES_TABLE
from kompend.core.types import ReasonCode, StageOutcome
from kompend.preprocessing.host_memory import HeavyParseGuard
from kompend.preprocessing.splitting import (
    NeverSplit,
    PageText,
    SourceSplitter,
    SplitMetadata,
    SplitPlan,
    SplitPlanError,
    SplitReason,
    SplitRule,
    SplitSegment,
    child_source_path,
    page_range,
    page_texts,
    partition_document,
    validate_plan,
)
from kompend.types import JSONDict, JSONList

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


def test_decide_never_sends_a_cropped_page_to_the_judge() -> None:
    document = _document(3)
    long_page = "z" * 4000
    document.elements.append(_element(2, long_page))
    judge = _FixedJudge([SplitSegment(1, 2), SplitSegment(3, 3)])

    SourceSplitter(judge=judge, min_pages=2, max_page_chars=10_000).decide(document)

    assert long_page in judge.calls[0][1].text


def _document(pages: int, *, base: Path = Path("sources/packet.pdf"), text: str = "body") -> ParsedDocument:
    elements = [_element(page, f"{text} p{page}") for page in range(1, pages + 1)]
    return ParsedDocument(source_file=base, elements=elements, title="Packet", page_count=pages)


# ── Fixtures: documents ────────────────────────────────────────────────


def _element(page: int | None, text: str = "", *, kind: ElementType = ElementType.TEXT) -> DocumentElement:
    return DocumentElement(type=kind, content=text, page=page, reading_order=page)


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
