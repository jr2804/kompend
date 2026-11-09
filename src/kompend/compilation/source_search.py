"""Search over the sources of one compile run.

A generation agent works one segment at a time: it sees the concepts of its
segment and the content of its segment. This module gives it reach beyond
that — the agent asks a question, the index names the sources that mention
it and where, and
:func:`kompend.compilation.gen_tools.read_source_window` serves the matching
source in windows that name the next offset. A hit names a source and an
offset, so the text is paged through, never shortened.

Three tiers, the same split :class:`kompend.compilation.flows.generation.segments._CoverageCredit`
uses for page titles:

- **Lexical** (always): BM25 over the tokens of each source. Rare query
  terms dominate, repeated terms saturate, long sources do not win by
  length alone.
- **Judge** (when ``[decision]`` is configured): a TypeSafe decision model
  answers one typed yes/no per top lexical candidate — does this chunk
  contain material that answers the query? The verdicts are fused by rank
  (judge-relevant first, lexical order within each group), never by
  thresholding a probability: fast-jev-compaction#154 measured that
  probabilities shift with request shape while their order holds. When the
  judge is live it replaces the semantic tier below; verdicts are cached
  per ``(query, snippet)`` so a repeated query in a run costs nothing.
- **Semantic** (when ``[embedding] model`` is configured and no judge is):
  cosine between the query and the matched snippet of the best lexical
  candidates. The snippets are cached per string, so a segment pays for
  the strings it actually asks about.

Judge and embedding failures leave the lexical ranking in place: each tier
is an enhancement over a working ranking, and a dead provider must not
take the source search with it.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, Field

from kompend.agents.base import AgentSpec, build_agent
from kompend.core.config._configs import DecisionConfig
from kompend.core.text.matching import cosine
from kompend.llm.decision import decision_model_string
from kompend.llm.models import build_decision_model
from kompend.logging import get_logger
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.embedding_pydantic import make_provider

#: Characters of a source served per ``read_source_window()`` call. The reply
#: names the offset of the next window, so the size of one window shortens a
#: reply, never the source.
SOURCE_WINDOW_CHARS: Final[int] = 6000

#: Characters of context shown around a match in a search reply. The reply
#: carries the match offset, so this is the preview of what is there — the
#: agent reads the rest through ``read_source_window()``.
SNIPPET_CHARS: Final[int] = 400

#: How many lexical candidates the semantic tier re-scores. Re-scoring is one
#: embedding call per candidate snippet, cached afterwards; the tail of the
#: lexical ranking is already weak, so re-scoring it would only cost time.
RERANK_CANDIDATES: Final[int] = 8

#: How many lexical candidates the judge tier reads. One batched request
#: carries every uncached candidate, so the pool width costs nothing per
#: search; beyond this the lexical tail is too weak for a verdict to lift.
JUDGE_CANDIDATES: Final[int] = 20

#: Share of the combined score the semantic tier contributes when it is live.
_SEMANTIC_WEIGHT: Final[float] = 0.5

# BM25 saturation and length normalisation, the standard values.
_K1: Final[float] = 1.5
_B: Final[float] = 0.75

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")

#: Tokens shorter than this carry no selection power ("a", "of", "to") and
#: would match every source.
_MIN_TOKEN_CHARS: Final[int] = 2

#: Characters of text kept before a match so a snippet opens on words rather
#: than mid-word.
_SNIPPET_LEAD: Final[int] = 120

logger = get_logger(__name__)


@dataclass
class SourceSearchJudgeDeps:
    """Deps of the source-search rerank judge (one batched request).

    ``candidates`` carries ``(stem, snippet)`` pairs; the model echoes each
    stem in its verdict, which is the join key back to the candidate.
    """

    source_id: str
    query: str
    candidates: list[tuple[str, str]]


class ChunkVerdict(BaseModel):
    """The judge's answer for one candidate chunk."""

    stem: str = Field(description="Stem of the candidate, copied verbatim from the request.")
    relevant: bool = Field(description="True when the chunk contains material that answers the query.")


class ChunkRelevanceDecision(BaseModel):
    """A decision model's verdict on every candidate chunk of one batch."""

    verdicts: list[ChunkVerdict] = Field(description="One verdict per candidate, echoing each stem verbatim.")


#: The judge agent, materialised lazily on the first search that has both a
#: configured decision model and uncached candidates — never at import or
#: index-build time. Declared here (not in ``kompend.agents``) so the tier
#: ships with the module that owns it; ``build_agent`` still routes the LLM
#: call through the agent machinery in ``kompend.agents.base``.
_JUDGE_SPEC = AgentSpec(
    name="source_search_judge",
    result_type=ChunkRelevanceDecision,
    deps_type=SourceSearchJudgeDeps,
    template_name="source_search_rerank.md.j2",
    system_template_name="source_search_rerank_system.md.j2",
    template_prefix="decision/",
    temperature=0.0,
    model_builder=build_decision_model,
)


@dataclass(frozen=True, slots=True)
class SourceHit:
    """One source that matched a query, and the window of it that matched.

    ``offset`` is where ``snippet`` starts inside the source, so the agent
    passes it straight to ``read_source_window()`` and reads on from the
    match it was shown.
    """

    stem: str
    offset: int
    total_chars: int
    score: float
    snippet: str


class SourceIndex:
    """Searchable view of the sources of one compile run.

    Built from the run's segments, whose ``(label, text)`` pairs carry the
    whole corpus: the pairs of one label concatenate back into that source.
    """

    def __init__(
        self,
        sources: Iterable[tuple[str, str]],
        *,
        embedding_model: str = "",
        decision: DecisionConfig | None = None,
    ) -> None:
        texts: dict[str, str] = {}
        for stem, text in sources:
            if not text:
                continue
            texts[stem] = texts.get(stem, "") + text
        self._texts: dict[str, str] = texts
        self._counts: dict[str, Counter[str]] = {stem: _tokens(text) for stem, text in texts.items()}
        self._lengths: dict[str, int] = {stem: sum(counts.values()) or 1 for stem, counts in self._counts.items()}
        self._avg_length: float = (sum(self._lengths.values()) / len(self._lengths)) if self._lengths else 0.0
        self._document_frequency: Counter[str] = Counter()
        for counts in self._counts.values():
            self._document_frequency.update(counts.keys())
        self._provider = self._provider_for(embedding_model)
        self._vectors: dict[str, list[float]] = {}
        self._decision = decision
        #: Verdict cache, ``(normalised query, snippet digest)`` → verdict.
        #: Guarded by a lock: the generation tool runs ``search()`` on a worker
        #: thread, so two searches can race. It lives per index, never global.
        self._verdict_cache: dict[tuple[str, str], bool] = {}
        self._verdict_lock = threading.Lock()
        #: The judge agent, materialised on the first search that has both a
        #: configured decision model and uncached candidates. An index without
        #: ``[decision]`` never builds it, so no key or endpoint is needed.
        self._judge: Any = None
        self._loop_warned = False

    def __len__(self) -> int:
        """Number of sources in the run."""
        return len(self._texts)

    def stems(self) -> list[str]:
        """Source labels the index can serve, in insertion order."""
        return list(self._texts)

    def total_chars(self, stem: str) -> int:
        """Length of the source named *stem* (0 when unknown)."""
        return len(self._texts.get(stem, ""))

    def search(self, query: str, k: int = 6) -> list[SourceHit]:
        """Return up to *k* sources that mention *query*, best first."""
        terms = set(_tokens(query))
        if not terms or not self._texts:
            return []
        scored: list[tuple[float, str]] = []
        for stem, counts in self._counts.items():
            score = self._lexical_score(terms, stem, counts)
            if score > 0:
                scored.append((score, stem))
        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], item[1]))
        if self._decision is not None:
            judged = self._judge_rerank(query, terms, scored[:JUDGE_CANDIDATES])
            if judged is not None:
                return [self._hit(stem, score, terms) for score, stem in judged[:k]]
        candidates = scored[: max(k, min(len(scored), RERANK_CANDIDATES))]
        scores = self._rerank(query, terms, candidates)
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [self._hit(stem, score, terms) for score, stem in scores[:k]]

    def read_window(self, stem: str, offset: int = 0, *, limit: int = SOURCE_WINDOW_CHARS) -> tuple[str, int]:
        """Return ``(window, next_offset)`` of the source named *stem*.

        *next_offset* equals the length of the source when the window reaches
        its end; otherwise it is the offset the next call starts from.
        """
        text = self._texts.get(stem)
        if text is None:
            return "", 0
        if limit <= 0 or (offset <= 0 and len(text) <= limit):
            return text, len(text)
        if offset >= len(text):
            return "", len(text)
        window = text[offset : offset + limit]
        return window, min(len(text), offset + len(window))

    @classmethod
    def from_segments(
        cls,
        segments: Iterable[Iterable[tuple[str, str]]],
        *,
        embedding_model: str = "",
        decision: DecisionConfig | None = None,
    ) -> SourceIndex:
        """Build the index from generation segments (``[(label, text), ...]`` each)."""
        return cls(
            (pair for segment in segments for pair in segment),
            embedding_model=embedding_model,
            decision=decision,
        )

    def _judge_rerank(self, query: str, terms: set[str], pool: Sequence[tuple[float, str]]) -> list[tuple[float, str]] | None:
        """Re-order the BM25-ranked *pool* by the judge's verdicts.

        Returns the reordered pool, or ``None`` when the judge cannot run —
        the caller then falls through to the semantic tier (today's
        behavior). Ordering is rank fusion, never a probability threshold:
        judge-relevant candidates first, BM25 order inside each group.
        """
        if not pool:
            return None
        # The judge call is synchronous; inside a running event loop it would
        # block the loop for the whole request. The generation tool already
        # hops to a worker thread before calling search(); this guard is
        # defence for direct callers, and it warns once, not per call.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if not self._loop_warned:
                logger.warning(
                    "Source search judge skipped: search() ran inside a running event loop "
                    "and would block it; keeping the lower tiers. Call search() from a worker "
                    "thread (the generation tool does) to use the judge."
                )
                self._loop_warned = True
            return None
        verdicts = self._judge_verdicts(query, terms, pool)
        if verdicts is None:
            return None
        relevant = [entry for entry in pool if verdicts.get(entry[1])]
        rest = [entry for entry in pool if not verdicts.get(entry[1])]
        order = ", ".join(f"{stem}={'yes' if verdicts.get(stem) else 'no' if stem in verdicts else 'unjudged'}" for _, stem in relevant + rest)
        logger.info(
            "Source search judge on %r: %d of %d candidates relevant (%s).",
            query,
            len(relevant),
            len(pool),
            order,
        )
        return relevant + rest

    def _judge_verdicts(self, query: str, terms: set[str], pool: Sequence[tuple[float, str]]) -> dict[str, bool] | None:
        """Return ``{stem: verdict}`` for the pool, or ``None`` on judge failure.

        One batched request carries every uncached candidate — not N
        concurrent requests: ``search()`` is a synchronous path, so N
        concurrent requests would need an event loop or a thread per request
        inside one tool call, while a single round trip keeps latency flat
        (~200 ms) regardless of pool size and gives every candidate the same
        question framing, which makes the verdicts comparable. The echoed
        stem is the join key: a verdict for a stem that was not sent is
        dropped (and counted), a sent candidate without a verdict stays
        unjudged and keeps its BM25 rank — a verdict is never invented.
        """
        query_key = " ".join(query.lower().split())
        verdicts: dict[str, bool] = {}
        pending: list[tuple[str, str]] = []
        keys: dict[str, tuple[str, str]] = {}
        for _, stem in pool:
            snippet = self._locate(stem, terms)[1]
            key = (query_key, hashlib.sha256(snippet.encode("utf-8")).hexdigest())
            keys[stem] = key
            with self._verdict_lock:
                cached = self._verdict_cache.get(key)
            if cached is None:
                pending.append((stem, snippet))
            else:
                verdicts[stem] = cached
        if pending:
            fresh = self._ask_judge(query, pending)
            if fresh is None:
                return None
            sent = {stem for stem, _ in pending}
            dropped = 0
            for verdict in fresh.verdicts:
                if verdict.stem not in sent or verdict.stem in verdicts:
                    dropped += 1
                    continue
                verdicts[verdict.stem] = verdict.relevant
                with self._verdict_lock:
                    self._verdict_cache[keys[verdict.stem]] = verdict.relevant
            if dropped:
                logger.info(
                    "Source search judge: %d verdict(s) dropped (stem not sent or duplicate).",
                    dropped,
                )
        return verdicts

    def _ask_judge(self, query: str, pending: list[tuple[str, str]]) -> ChunkRelevanceDecision | None:
        """Run the batched judge request; ``None`` on any failure."""
        config = self._decision
        if config is None:
            return None
        model_string = decision_model_string(config)
        if model_string is None:
            return None
        try:
            agent = self._judge_agent(model_string, config)
            return agent.run(SourceSearchJudgeDeps(source_id="source-search", query=query, candidates=pending))
        except Exception as exc:
            logger.warning("Source search judge failed, keeping the lower tiers: %s", exc)
            return None

    def _judge_agent(self, model_string: str, config: DecisionConfig) -> Any:
        """Return the judge, materialising it once per index on first use.

        The build is kept out of the lock (it resolves a model provider); the
        double check inside makes a concurrent second search reuse the winner
        instead of building a duplicate.
        """
        with self._verdict_lock:
            cached = self._judge
        if cached is not None:
            return cached
        agent = build_agent(
            _JUDGE_SPEC,
            model_string=model_string,
            timeout=config.timeout,
            retries=0,
        )
        with self._verdict_lock:
            if self._judge is None:
                self._judge = agent
            return self._judge

    def _lexical_score(self, terms: set[str], stem: str, counts: Counter[str]) -> float:
        score = 0.0
        length = self._lengths[stem]
        for term in terms:
            frequency = counts.get(term)
            if not frequency:
                continue
            document_frequency = self._document_frequency[term]
            idf = math.log(1.0 + (len(self._counts) - document_frequency + 0.5) / (document_frequency + 0.5))
            saturation = frequency * (_K1 + 1) / (frequency + _K1 * (1.0 - _B + _B * length / self._avg_length))
            score += idf * saturation
        return score

    def _rerank(self, query: str, terms: set[str], candidates: Sequence[tuple[float, str]]) -> list[tuple[float, str]]:
        """Fuse the semantic score into the lexical ranking when a provider is live."""
        if self._provider is None or not candidates:
            return list(candidates)
        query_vector = self._embed(query)
        if query_vector is None:
            return list(candidates)
        best = candidates[0][0] or 1.0
        fused: list[tuple[float, str]] = []
        for lexical, stem in candidates:
            _, snippet = self._locate(stem, terms)
            snippet_vector = self._embed(snippet)
            lexical_share = lexical / best
            if snippet_vector is None:
                fused.append((lexical_share, stem))
                continue
            semantic = max(0.0, cosine(query_vector, snippet_vector))
            fused.append(((1.0 - _SEMANTIC_WEIGHT) * lexical_share + _SEMANTIC_WEIGHT * semantic, stem))
        return fused

    def _hit(self, stem: str, score: float, terms: set[str]) -> SourceHit:
        offset, snippet = self._locate(stem, terms)
        return SourceHit(stem=stem, offset=offset, total_chars=len(self._texts[stem]), score=score, snippet=snippet)

    def _locate(self, stem: str, terms: set[str]) -> tuple[int, str]:
        """Return the offset of the best match in *stem* and its preview."""
        text = self._texts[stem]
        lowered = text.lower()
        best_offset = -1
        for term in terms:
            position = lowered.find(term)
            if position >= 0 and (best_offset < 0 or position < best_offset):
                best_offset = position
        if best_offset < 0:
            head = text[:SNIPPET_CHARS]
            return 0, head
        start = max(0, best_offset - _SNIPPET_LEAD)
        return start, text[start : start + SNIPPET_CHARS]

    def _embed(self, text: str) -> list[float] | None:
        if self._provider is None:
            return None
        cached = self._vectors.get(text)
        if cached is not None:
            return cached
        try:
            vector = self._provider.embed(text)
        except Exception as exc:
            logger.debug("Source search embed failed for %d characters of query text: %s", len(text), exc)
            return None
        self._vectors[text] = vector
        return vector

    @staticmethod
    def _provider_for(embedding_model: str) -> EmbeddingProvider | None:
        if not embedding_model or embedding_model == "hash":
            return None
        try:
            return make_provider(embedding_model)
        except Exception as exc:
            # An unreachable embedding provider leaves the lexical ranking in
            # place; the search still answers, only without the semantic tier.
            logger.debug("Source search semantic tier disabled: %s", exc)
            return None


def _tokens(text: str) -> Counter[str]:
    """Count the searchable tokens of *text*."""
    return Counter(token for token in _TOKEN_RE.findall(text.lower()) if len(token) >= _MIN_TOKEN_CHARS)
