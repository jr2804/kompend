"""Tests for source search during generation.

Covers the lexical tier (BM25 over the run's sources), the judge tier (a
TypeSafe decision model answering one yes/no per candidate, fused by rank),
the semantic tier (cosine over the matched snippets, only when an embedding
provider is live), window paging, and the two generation tools that expose
them. No LLM or embedding endpoint is contacted: providers and the judge
agent are fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import patch

from pydantic_ai import RunContext

from kompend.agents.types import ToolGenerationDeps
from kompend.compilation.gen_tools import read_source_window, search_sources
from kompend.compilation.source_search import (
    ChunkRelevanceDecision,
    ChunkVerdict,
    SourceIndex,
    SourceSearchJudgeDeps,
)
from kompend.core.config.models.decision import DecisionConfig

_ALPHA = "The transcoder maps timestamps onto the output clock. Latency is bounded per frame."
_BETA = "Listening quality is scored by the POLQA algorithm. The transcoder feeds it with speech frames."


_DECISION = DecisionConfig(enabled=True, model="typesafe/jev-latest")

#: Six sources that all match "transcoder", with falling term frequency, so
#: the lexical order is source-5 … source-0 and the judge has something to
#: promote from the tail.
_JUDGE_SOURCES: list[tuple[str, str]] = [(f"source-{number}.md", "transcoder " * (number + 1) + "latency is bounded per frame") for number in range(6)]


class _MappingEmbedder:
    """Embeds by table lookup so a test controls the cosine exactly.

    Unknown strings embed to the zero vector, which scores 0.0 — a test
    that forgets a string fails on the ranking, not on a KeyError.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vectors.get(text, [0.0, 0.0])


class _BrokenEmbedder:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding endpoint is down")


class _DepsCtx:
    """Minimal stand-in for ``RunContext`` (the tools only read ``deps``)."""

    def __init__(self, deps: ToolGenerationDeps) -> None:
        self.deps = deps


class TestLexicalTier:
    def test_a_source_that_mentions_the_query_is_found(self) -> None:
        index = _index(("alpha.md", _ALPHA), ("beta.md", _BETA))
        hits = index.search("transcoder")
        assert {hit.stem for hit in hits} == {"alpha.md", "beta.md"}

    def test_a_source_that_does_not_mention_the_query_is_not_returned(self) -> None:
        index = _index(("alpha.md", _ALPHA), ("beta.md", _BETA))
        hits = index.search("POLQA")
        assert [hit.stem for hit in hits] == ["beta.md"]

    def test_a_rare_term_outranks_a_common_one(self) -> None:
        # "transcoder" appears in both sources, "POLQA" only in beta: a query
        # naming both must put the source that holds the rare term first.
        index = _index(("alpha.md", _ALPHA), ("beta.md", _BETA))
        hits = index.search("POLQA transcoder")
        assert hits[0].stem == "beta.md"

    def test_an_absent_term_returns_no_hits(self) -> None:
        index = _index(("alpha.md", _ALPHA))
        assert index.search("quantum chromodynamics") == []

    def test_an_empty_query_returns_no_hits(self) -> None:
        index = _index(("alpha.md", _ALPHA))
        assert index.search("  ") == []

    def test_k_bounds_the_reply(self) -> None:
        index = _index(("alpha.md", _ALPHA), ("beta.md", _BETA))
        assert len(index.search("transcoder", k=1)) == 1

    def test_a_hit_names_where_its_preview_starts(self) -> None:
        # The offset is what read_source_window() takes, so the match has to
        # sit inside the window it names.
        index = _index(("beta.md", _BETA))
        hit = index.search("POLQA")[0]
        assert hit.offset + len(hit.snippet) > _BETA.index("POLQA") >= hit.offset
        assert hit.total_chars == len(_BETA)

    def test_the_snippet_carries_the_query_term(self) -> None:
        index = _index(("beta.md", _BETA))
        assert "POLQA" in index.search("POLQA")[0].snippet

    def test_segments_of_one_source_concatenate_back_into_it(self) -> None:
        index = SourceIndex.from_segments([[("alpha.md", _ALPHA[:20]), ("alpha.md", _ALPHA[20:]), ("beta.md", _BETA)]])
        assert len(index) == 2
        assert index.stems() == ["alpha.md", "beta.md"]
        assert index.total_chars("alpha.md") == len(_ALPHA)
        assert index.read_window("alpha.md")[0] == _ALPHA

    def test_empty_text_is_not_indexed(self) -> None:
        assert len(_index(("empty.md", ""))) == 0


class TestReadWindow:
    def test_a_window_names_the_next_offset(self) -> None:
        index = _index(("alpha.md", _ALPHA))
        window, next_offset = index.read_window("alpha.md", 0, limit=10)
        assert window == _ALPHA[:10]
        assert next_offset == 10

    def test_the_last_window_reports_the_end_of_the_source(self) -> None:
        index = _index(("alpha.md", _ALPHA))
        window, next_offset = index.read_window("alpha.md", len(_ALPHA) - 5, limit=10)
        assert window == _ALPHA[-5:]
        assert next_offset == len(_ALPHA)

    def test_paging_yields_the_whole_source(self) -> None:
        index = _index(("alpha.md", _ALPHA))
        parts: list[str] = []
        offset = 0
        while True:
            window, offset = index.read_window("alpha.md", offset, limit=7)
            if not window:
                break
            parts.append(window)
        assert "".join(parts) == _ALPHA

    def test_an_unknown_source_returns_nothing(self) -> None:
        index = _index(("alpha.md", _ALPHA))
        assert index.read_window("missing.md") == ("", 0)

    def test_an_offset_past_the_end_returns_nothing(self) -> None:
        index = _index(("alpha.md", _ALPHA))
        assert index.read_window("alpha.md", 10_000) == ("", len(_ALPHA))


class TestSemanticTier:
    def test_the_semantic_tier_can_reorder_lexical_candidates(self) -> None:
        """A source the embedding model judges closer wins over the lexical order."""
        index = _index(("alpha.md", _ALPHA), ("beta.md", _BETA))
        lexical = [hit.stem for hit in index.search("transcoder", k=2)]
        snippets = {hit.stem: hit.snippet for hit in index.search("transcoder", k=2)}
        query_vector, other = [1.0, 0.0], [0.0, 1.0]
        embedder = _MappingEmbedder(
            {
                "transcoder": query_vector,
                snippets[lexical[1]]: query_vector,
                snippets[lexical[0]]: other,
            }
        )
        with patch("kompend.compilation.source_search.make_provider", return_value=embedder):
            rescored = _index(("alpha.md", _ALPHA), ("beta.md", _BETA), embedding_model="fake/model").search("transcoder", k=2)
        assert rescored[0].stem == lexical[1]

    def test_the_semantic_tier_embeds_each_snippet_once(self) -> None:
        embedder = _MappingEmbedder({"transcoder": [1.0, 0.0]})
        with patch("kompend.compilation.source_search.make_provider", return_value=embedder):
            index = _index(("alpha.md", _ALPHA), ("beta.md", _BETA), embedding_model="fake/model")
            index.search("transcoder", k=2)
            first = len(embedder.calls)
            index.search("transcoder", k=2)
        assert first > 1
        assert len(embedder.calls) == first

    def test_no_embedding_model_keeps_the_search_lexical(self) -> None:
        with patch("kompend.compilation.source_search.make_provider") as factory:
            index = _index(("alpha.md", _ALPHA), embedding_model="")
            assert index.search("transcoder")
        factory.assert_not_called()

    def test_the_hash_placeholder_does_not_embed(self) -> None:
        with patch("kompend.compilation.source_search.make_provider") as factory:
            index = _index(("alpha.md", _ALPHA), embedding_model="hash")
            assert index.search("transcoder")
        factory.assert_not_called()

    def test_an_unusable_embedding_provider_keeps_the_lexical_ranking(self) -> None:
        with patch("kompend.compilation.source_search.make_provider", side_effect=RuntimeError("no provider")):
            index = _index(("alpha.md", _ALPHA), ("beta.md", _BETA), embedding_model="fake/model")
        assert [hit.stem for hit in index.search("POLQA")] == ["beta.md"]

    def test_a_dead_embedder_keeps_the_lexical_ranking(self) -> None:
        with patch("kompend.compilation.source_search.make_provider", return_value=_BrokenEmbedder()):
            index = _index(("alpha.md", _ALPHA), ("beta.md", _BETA), embedding_model="fake/model")
            hits = index.search("POLQA")
        assert [hit.stem for hit in hits] == ["beta.md"]


class TestSearchSourcesTool:
    def test_the_reply_names_the_source_offset_and_snippet(self) -> None:
        deps = ToolGenerationDeps(source_id="seg", source_index=_index(("beta.md", _BETA)))
        reply = _call(search_sources, deps, "POLQA")
        assert "beta.md" in reply
        assert "preview at offset 0" in reply
        assert "POLQA" in reply
        assert "read_source_window" in reply

    def test_a_missing_term_says_so(self) -> None:
        deps = ToolGenerationDeps(source_id="seg", source_index=_index(("alpha.md", _ALPHA)))
        assert "No source of this run mentions" in _call(search_sources, deps, "POLQA")

    def test_a_run_without_an_index_says_so(self) -> None:
        deps = ToolGenerationDeps(source_id="seg")
        assert "not available" in _call(search_sources, deps, "POLQA")

    def test_k_is_clamped_to_the_reply_budget(self) -> None:
        deps = ToolGenerationDeps(source_id="seg", source_index=_index(("alpha.md", _ALPHA), ("beta.md", _BETA)))
        assert "2 source(s) match" in _call(search_sources, deps, "transcoder", 99)


class TestReadSourceWindowTool:
    def test_a_window_names_the_next_offset(self) -> None:
        deps = ToolGenerationDeps(source_id="seg", source_index=_index(("alpha.md", _ALPHA)))
        reply = _call(read_source_window, deps, "alpha.md", 0)
        assert _ALPHA in reply or "end of window" in reply

    def test_a_long_source_is_paged_with_the_next_call(self) -> None:
        long_source = "frame " * 4000
        deps = ToolGenerationDeps(source_id="seg", source_index=_index(("long.md", long_source)))
        reply = _call(read_source_window, deps, "long.md", 0)
        assert f"of {len(long_source)} characters read" in reply
        assert "offset=6000" in reply

    def test_the_end_of_a_source_carries_no_hint(self) -> None:
        deps = ToolGenerationDeps(source_id="seg", source_index=_index(("alpha.md", _ALPHA)))
        reply = _call(read_source_window, deps, "alpha.md", 0)
        assert "end of window" not in reply
        assert reply == _ALPHA

    def test_an_unknown_source_lists_what_is_available(self) -> None:
        deps = ToolGenerationDeps(source_id="seg", source_index=_index(("alpha.md", _ALPHA)))
        reply = _call(read_source_window, deps, "missing.md", 0)
        assert "Unknown source" in reply
        assert "alpha.md" in reply

    def test_a_long_source_list_becomes_a_count_and_a_search_hint(self) -> None:
        sources = [(f"source-{number}.md", _ALPHA) for number in range(25)]
        deps = ToolGenerationDeps(source_id="seg", source_index=_index(*sources))
        reply = _call(read_source_window, deps, "missing.md", 0)
        assert "25 sources" in reply
        assert "search_sources" in reply
        assert "source-24.md" not in reply

    def test_a_run_without_an_index_says_so(self) -> None:
        deps = ToolGenerationDeps(source_id="seg")
        assert "not available" in _call(read_source_window, deps, "alpha.md", 0)


class TestJudgeTier:
    def test_the_judge_promotes_a_chunk_the_lexical_order_put_last(self) -> None:
        index = _judged_index()
        lexical = _lexical_baseline()
        assert lexical[-1] == "source-0.md"
        judge = _FakeJudge(relevant={"source-0.md"})
        with patch("kompend.compilation.source_search.build_agent", return_value=judge):
            judged = [hit.stem for hit in index.search("transcoder", k=len(_JUDGE_SOURCES))]
        assert judged[0] == "source-0.md"
        assert sorted(judged) == sorted(lexical)

    def test_a_failing_judge_falls_back_to_the_lexical_order(self) -> None:
        index = _judged_index()
        lexical = _lexical_baseline()
        judge = _FakeJudge(fail=TimeoutError("decision model timed out"))
        with patch("kompend.compilation.source_search.build_agent", return_value=judge):
            judged = [hit.stem for hit in index.search("transcoder", k=len(_JUDGE_SOURCES))]
        assert judged == lexical

    def test_rank_fusion_is_stable_when_every_verdict_agrees(self) -> None:
        """fast-jev-compaction#154: probability scale shifts, order does not.

        A boolean verdict has no scale to shift; the property that matters is
        that a board-wide verdict changes nothing: all-relevant and
        all-irrelevant both keep the BM25 order, so a judge that is merely
        uncertain can never scramble the ranking.
        """
        index = _judged_index()
        lexical = _lexical_baseline()
        with patch("kompend.compilation.source_search.build_agent", return_value=_FakeJudge(relevant=set(lexical))):
            assert [hit.stem for hit in index.search("transcoder", k=len(_JUDGE_SOURCES))] == lexical
        # A fresh index: the first search filled the verdict cache, and a
        # cached verdict must not stand in for this second question.
        other = _judged_index()
        with patch("kompend.compilation.source_search.build_agent", return_value=_FakeJudge()):
            assert [hit.stem for hit in other.search("transcoder", k=len(_JUDGE_SOURCES))] == lexical

    def test_a_repeated_query_does_not_ask_the_judge_again(self) -> None:
        index = _judged_index()
        judge = _FakeJudge(relevant={"source-0.md"})
        with patch("kompend.compilation.source_search.build_agent", return_value=judge):
            index.search("transcoder")
            first = len(judge.calls)
            index.search("transcoder")
        assert first == 1
        assert len(judge.calls) == 1

    def test_a_verdict_for_a_stem_that_was_not_sent_is_dropped(self) -> None:
        index = _judged_index()
        lexical = _lexical_baseline()
        judge = _FakeJudge(extra={"ghost.md"})
        with patch("kompend.compilation.source_search.build_agent", return_value=judge):
            judged = [hit.stem for hit in index.search("transcoder", k=len(_JUDGE_SOURCES))]
        assert "ghost.md" not in judged
        assert judged == lexical

    def test_a_missing_verdict_keeps_the_lexical_rank(self) -> None:
        index = _judged_index()
        lexical = _lexical_baseline()
        # The judge promotes the last candidate but says nothing about the
        # others: they keep their BM25 order behind it.
        judge = _FakeJudge(relevant={"source-0.md"}, skip=set(lexical) - {"source-0.md"})
        with patch("kompend.compilation.source_search.build_agent", return_value=judge):
            judged = [hit.stem for hit in index.search("transcoder", k=len(_JUDGE_SOURCES))]
        assert judged[0] == "source-0.md"
        assert judged[1:] == lexical[:-1]

    def test_the_judge_is_not_built_until_a_search_needs_it(self) -> None:
        with patch("kompend.compilation.source_search.build_agent") as factory:
            index = _judged_index()
            assert index.search("  ") == []
        factory.assert_not_called()

    def test_a_search_inside_a_running_loop_skips_the_judge(self) -> None:
        index = _judged_index()
        lexical = _lexical_baseline()

        async def scenario() -> list[str]:
            return [hit.stem for hit in index.search("transcoder", k=len(_JUDGE_SOURCES))]

        with patch("kompend.compilation.source_search.build_agent") as factory:
            assert asyncio.run(scenario()) == lexical
        factory.assert_not_called()

    def test_no_judge_and_no_embedding_model_still_searches(self) -> None:
        index = _index(("beta.md", _BETA))
        assert [hit.stem for hit in index.search("POLQA")] == ["beta.md"]


class _FakeJudge:
    """Stands in for the materialised judge agent.

    Answers ``relevant`` for the listed stems, stays silent about ``skip``
    candidates (an unjudged candidate), appends ``extra`` echo lines, or
    raises ``fail`` — all without touching a model.
    """

    def __init__(
        self,
        relevant: set[str] | None = None,
        *,
        skip: set[str] | None = None,
        extra: set[str] | None = None,
        fail: Exception | None = None,
    ) -> None:
        self._relevant = relevant or set()
        self._skip = skip or set()
        self._extra = extra or set()
        self._fail = fail
        self.calls: list[SourceSearchJudgeDeps] = []

    def run(self, deps: SourceSearchJudgeDeps) -> ChunkRelevanceDecision:
        self.calls.append(deps)
        if self._fail is not None:
            raise self._fail
        verdicts = [ChunkVerdict(stem=stem, relevant=stem in self._relevant) for stem, _ in deps.candidates if stem not in self._skip]
        verdicts.extend(ChunkVerdict(stem=stem, relevant=True) for stem in self._extra)
        return ChunkRelevanceDecision(verdicts=verdicts)


def _judged_index() -> SourceIndex:
    return SourceIndex(_JUDGE_SOURCES, decision=_DECISION)


def _lexical_baseline() -> list[str]:
    """The same corpus without a judge — the BM25 order a test compares against."""
    return [hit.stem for hit in SourceIndex(_JUDGE_SOURCES).search("transcoder", k=len(_JUDGE_SOURCES))]


def _index(*sources: tuple[str, str], embedding_model: str = "") -> SourceIndex:
    return SourceIndex(sources, embedding_model=embedding_model)


def _call(tool: Any, deps: ToolGenerationDeps, *args: Any) -> str:
    """Run one gen tool against *deps* the way the agent framework would."""
    return asyncio.run(tool(cast("RunContext[ToolGenerationDeps]", _DepsCtx(deps)), *args))
