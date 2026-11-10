"""Tests for source search during generation.

Covers the lexical tier (BM25 over the run's sources), the judge tier (a
TypeSafe decision model answering one yes/no per candidate, fused by rank),
the semantic tier (cosine over the matched snippets, only when an embedding
provider is live), window paging, and the two generation tools that expose
them. No LLM or embedding endpoint is contacted: providers and the judge
agent are fakes.

The verdict schema is checked against the real decision-model machinery, not
a stub: a result type the decision grammar refuses must fail here, because a
stubbed judge once hid exactly that.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, _output
from pydantic_ai.models import decision as decision_grammar

from kompend.agents.types import ToolGenerationDeps
from kompend.compilation import source_search
from kompend.compilation.gen_tools import read_source_window, search_sources
from kompend.compilation.source_search import (
    CandidateVerdict,
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
        judge = _JudgePolicy(relevant={"source-0.md"})
        with judge.installed():
            judged = [hit.stem for hit in index.search("transcoder", k=len(_JUDGE_SOURCES))]
        assert judged[0] == "source-0.md"
        assert judged[1:] == lexical[:-1]
        assert sorted(judged) == sorted(lexical)

    def test_one_failing_candidate_is_unjudged_and_keeps_its_lexical_rank(self, caplog: pytest.LogCaptureFixture) -> None:
        """A single dead request is an unjudged candidate, not a dead tier."""
        lexical = _lexical_baseline()
        judge = _JudgePolicy(relevant={"source-0.md"}, fail_for=set(lexical) - {"source-0.md"})
        with judge.installed(), caplog.at_level(logging.INFO, logger="kompend.compilation.source_search"):
            judged = [hit.stem for hit in _judged_index().search("transcoder", k=len(_JUDGE_SOURCES))]
        assert judged[0] == "source-0.md"
        assert judged[1:] == lexical[:-1]
        assert "left 5 of 6 candidate(s) unjudged" in caplog.text
        assert "failed for every candidate" not in caplog.text

    def test_a_failing_judge_keeps_the_lexical_order_and_names_the_cause(self, caplog: pytest.LogCaptureFixture) -> None:
        lexical = _lexical_baseline()
        judge = _JudgePolicy(fail=TimeoutError("decision model timed out"))
        with judge.installed(), caplog.at_level(logging.WARNING, logger="kompend.compilation.source_search"):
            judged = [hit.stem for hit in _judged_index().search("transcoder", k=len(_JUDGE_SOURCES))]
        assert judged == lexical
        assert "judge failed for every candidate" in caplog.text
        assert "decision model timed out" in caplog.text

    def test_a_dead_judge_falls_through_to_the_semantic_tier(self) -> None:
        """Total judge failure hands the ranking back to today's cosine tier."""
        vectors = _semantic_vectors()
        with patch("kompend.compilation.source_search.make_provider", return_value=_MappingEmbedder(vectors)):
            semantic = SourceIndex(_JUDGE_SOURCES, embedding_model="fake/model")
            expected = [hit.stem for hit in semantic.search("transcoder", k=len(_JUDGE_SOURCES))]
        judge = _JudgePolicy(fail=RuntimeError("judge endpoint is down"))
        embedder = _MappingEmbedder(vectors)
        with (
            judge.installed(),
            patch("kompend.compilation.source_search.make_provider", return_value=embedder),
        ):
            index = SourceIndex(_JUDGE_SOURCES, embedding_model="fake/model", decision=_DECISION)
            judged = [hit.stem for hit in index.search("transcoder", k=len(_JUDGE_SOURCES))]
        assert judged == expected
        assert embedder.calls, "the semantic tier must have been asked"

    def test_rank_fusion_is_stable_when_every_verdict_agrees(self) -> None:
        """fast-jev-compaction#154: probability scale shifts, order does not.

        A boolean verdict has no scale to shift; the property that matters is
        that a board-wide verdict changes nothing: all-relevant and
        all-irrelevant both keep the BM25 order, so a judge that is merely
        uncertain can never scramble the ranking.
        """
        lexical = _lexical_baseline()
        with _JudgePolicy(relevant=set(lexical)).installed():
            assert [hit.stem for hit in _judged_index().search("transcoder", k=len(_JUDGE_SOURCES))] == lexical
        # A fresh index: the first search filled the verdict cache, and a
        # cached verdict must not stand in for this second question.
        with _JudgePolicy().installed():
            assert [hit.stem for hit in _judged_index().search("transcoder", k=len(_JUDGE_SOURCES))] == lexical

    def test_the_judge_is_asked_once_per_candidate_with_one_chunk(self) -> None:
        judge = _JudgePolicy(relevant={"source-0.md"})
        with judge.installed():
            _judged_index().search("transcoder", k=len(_JUDGE_SOURCES))
        assert len(judge.calls) == len(_JUDGE_SOURCES)
        assert {call.stem for call in judge.calls} == {stem for stem, _ in _JUDGE_SOURCES}
        assert all(call.query == "transcoder" for call in judge.calls)
        assert all(isinstance(call.candidate, str) and call.candidate for call in judge.calls)

    def test_a_repeated_query_does_not_ask_the_judge_again(self) -> None:
        index = _judged_index()
        judge = _JudgePolicy(relevant={"source-0.md"})
        with judge.installed():
            index.search("transcoder")
            first = len(judge.calls)
            index.search("transcoder")
        assert first == len(_JUDGE_SOURCES)
        assert len(judge.calls) == first

    def test_concurrent_judging_yields_the_same_verdicts_as_a_serial_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pool width changes latency, never the ranking."""
        relevant = {"source-0.md", "source-2.md"}
        with _JudgePolicy(relevant=relevant).installed():
            concurrent = [hit.stem for hit in _judged_index().search("transcoder", k=len(_JUDGE_SOURCES))]
        monkeypatch.setattr(source_search, "JUDGE_CONCURRENCY", 1)
        serial_judge = _JudgePolicy(relevant=relevant)
        with serial_judge.installed():
            serial = [hit.stem for hit in _judged_index().search("transcoder", k=len(_JUDGE_SOURCES))]
        assert serial_judge.calls
        assert len(serial_judge.calls) == len(_JUDGE_SOURCES)
        assert concurrent == serial
        # Rank fusion, not score fusion: the two relevant chunks lead in
        # BM25 order (source-2 before source-0), the rest follow unchanged.
        assert concurrent == ["source-2.md", "source-0.md", "source-5.md", "source-4.md", "source-3.md", "source-1.md"]

    def test_each_worker_thread_builds_its_own_judge_agent(self) -> None:
        """``run()`` keeps per-run state on the instance, so instances are not shared."""
        judge = _JudgePolicy(relevant={"source-0.md"})
        with judge.installed():
            _judged_index().search("transcoder", k=len(_JUDGE_SOURCES))
        assert judge.builds, "the judge was never materialised"
        assert len(judge.builds) <= source_search.JUDGE_CONCURRENCY
        assert len({id(agent) for agent in judge.builds}) == len(judge.builds)
        assert len(judge.threads) <= source_search.JUDGE_CONCURRENCY

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


class TestVerdictSchemaIsAccepted:
    """The verdict type is checked against the real decision grammar.

    This is the gap that let a dead judge ship: a stub answered a shape the
    decision model refuses before sending anything. These tests build the
    request the model would build, with no endpoint involved.
    """

    _LIMITS = decision_grammar._Limits(choice_options=None, score_levels=None, requires_instructions=False, questions=None)

    def test_the_single_bool_verdict_becomes_a_yes_no_question(self) -> None:
        tool = self._output_tool(CandidateVerdict)
        assert decision_grammar._expressible(tool, None, self._LIMITS) is True
        questions = decision_grammar._questions(decision_grammar._fields(tool), tool, None, self._LIMITS, label=None)
        assert list(questions) == ["relevant"]
        question = questions["relevant"]
        assert question.type == "noul"
        assert "answers the query" in str(question.instructions)

    def test_a_list_of_verdict_models_is_refused(self) -> None:
        class _BatchedVerdicts(BaseModel):
            """One verdict per candidate chunk."""

            verdicts: list[CandidateVerdict] = Field(description="One verdict per candidate.")

        tool = self._output_tool(_BatchedVerdicts)
        assert decision_grammar._expressible(tool, None, self._LIMITS) is False
        with pytest.raises(Exception, match="list must be of two or more string options"):
            decision_grammar._questions(decision_grammar._fields(tool), tool, None, self._LIMITS, label=None)

    @staticmethod
    def _output_tool(model: type[BaseModel]) -> Any:
        """The tool definition the decision model reads for this output type."""
        toolset = _output.OutputToolset.build(cast("list[Any]", [model]))
        assert toolset is not None
        return toolset._tool_defs[0]


class _JudgePolicy:
    """What the fake judge answers, shared by the per-thread agent instances.

    ``build_agent`` is patched to hand out one agent per call (the tier keeps
    one per worker thread), so the policy — the answer set, the failure set,
    and the call log — lives here, behind a lock, instead of on the agent.
    """

    def __init__(
        self,
        relevant: set[str] | None = None,
        *,
        fail: Exception | None = None,
        fail_for: set[str] | None = None,
    ) -> None:
        self._relevant = relevant or set()
        self._fail = fail
        self._fail_for = fail_for or set()
        self.calls: list[SourceSearchJudgeDeps] = []
        self.builds: list[Any] = []
        self.threads: set[int] = set()
        self._lock = threading.Lock()

    def installed(self) -> Any:
        """Patch ``build_agent`` for the ``with`` block."""
        return patch("kompend.compilation.source_search.build_agent", side_effect=self._build)

    def answer(self, deps: SourceSearchJudgeDeps) -> CandidateVerdict:
        with self._lock:
            self.calls.append(deps)
            self.threads.add(threading.get_ident())
        if self._fail is not None:
            raise self._fail
        if deps.stem in self._fail_for:
            raise RuntimeError(f"judge request for {deps.stem} failed")
        return CandidateVerdict(relevant=deps.stem in self._relevant)

    def _build(self, spec: Any, **kwargs: Any) -> _FakeJudge:
        agent = _FakeJudge(self)
        with self._lock:
            self.builds.append(agent)
        return agent


class _FakeJudge:
    """Stands in for one materialised judge agent; answers through the policy."""

    def __init__(self, policy: _JudgePolicy) -> None:
        self._policy = policy

    def run(self, deps: SourceSearchJudgeDeps) -> CandidateVerdict:
        return self._policy.answer(deps)


def _semantic_vectors() -> dict[str, list[float]]:
    """Vectors under which the cosine tier reorders the lexical candidates."""
    lexical = _lexical_baseline()
    snippets = {hit.stem: hit.snippet for hit in _index(*_JUDGE_SOURCES).search("transcoder", k=len(_JUDGE_SOURCES))}
    query_vector, other = [1.0, 0.0], [0.0, 1.0]
    return {
        "transcoder": query_vector,
        snippets[lexical[-1]]: query_vector,
        snippets[lexical[0]]: other,
    }


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
