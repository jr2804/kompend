"""NodeSummarizer — bottom-up LLM summaries for hierarchical chunk trees.

Walks a :class:`ChunkTree` bottom-up and generates concise summaries for each
internal node.  These summaries serve as bounded context for parent and sibling
nodes during retrieval (Phase 4) and compilation (Phase 5), allowing the tree
to represent arbitrarily large documents in a compact, multi-resolution form.

Key design decisions:
- SINGLE_PASS trees are skipped entirely (the whole document fits in one
  context window, so summaries add overhead without value).
- Leaf nodes are never summarised — their ``.chunk`` text or ``.structured``
  data is the source of truth.
- Internal nodes with a single child are optionally skipped
  (``skip_single_child``) — a single-child wrapper adds no information.
- LLM calls go through the Jinja2 template ``chunking/node_summary.md.j2``.
- When no LLM client is provided, an extractive fallback is used (first
  sentence per child) so the pipeline still functions without an LLM.
- Summaries are cached by content hash + model name so unchanged subtrees
  are not re-summarised across runs. ``cache_ignore_model`` drops the model
  from the key (``compile.cache_ignore_model``): after a model switch the
  previous model's summaries are reused instead of re-derived.
- Agentic sessions are bounded per batch (``SummarizerSessionPolicy``), so a
  session that exhausts its request budget or ends without recording anything
  only loses its own batch; the remaining batches still run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import cast

from pydantic_ai.exceptions import UsageLimitExceeded

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.summarizer import SummarizerAgent
from kompend.agents.summarizer_agentic import AgenticSummarizerAgent
from kompend.agents.types import SummarizerDeps, SummaryNode
from kompend.core.cache.store import CacheStore
from kompend.core.config._configs import BudgetConfig
from kompend.core.models import ChunkNode, ChunkStrategy, ChunkTree
from kompend.core.models.structured import structured_to_text
from kompend.core.templates import get_template_loader
from kompend.core.text.windows import split_text_windows
from kompend.logging import get_logger

_TEMPLATE = "chunking/node_summary.md.j2"
_CACHE_LAYER = "summary"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

logger = get_logger(__name__)


@dataclass(frozen=True)
class SummarizerSessionPolicy:
    """Batching and retry policy for one agentic summarizer tree.

    Both values come from ``compile.budget`` (``summary_session_batch_size``
    and ``summary_session_stall_retries``). No default lives here, so the
    config schema stays the single source of truth for the numbers.

    Attributes:
        batch_size: Maximum nodes handed to one agentic session.
        stall_retries: Extra attempts a node gets after a session that recorded
            nothing for its payload. ``0`` abandons it on the first stall.
        payload_chars: Characters of node content one session payload may
            carry. It is derived from the model's window
            (``ContextWindow.input_budget``), not from a configured ceiling:
            windows beyond it stay pending and are asked in the next session,
            so nothing is dropped.
    """

    batch_size: int
    stall_retries: int
    payload_chars: int


@dataclass(frozen=True)
class SummarizerContentPolicy:
    """Content shaping policy for one summarizer run.

    Every value comes from ``compile.budget``. No default lives here, so the
    config schema stays the single source of truth for the numbers.

    Attributes:
        max_content_chars: Characters of child content per summary call in the
            plain (non-agentic) path. Longer content is split into several
            calls — it is never truncated.
        window_chars: Characters per payload window in the agentic path. One
            window costs one summarize call, so a bigger window means fewer
            calls (faster) — it never drops content.
        target_chars: Target length of the produced summary, in characters.
        skip_single_child: When ``True``, an internal node with exactly one
            child is skipped — the child's content already represents it.
    """

    max_content_chars: int
    window_chars: int
    target_chars: int
    skip_single_child: bool

    @classmethod
    def from_config(cls, budget: BudgetConfig) -> SummarizerContentPolicy:
        """Build the policy from the ``compile.budget`` config section."""
        return cls(
            max_content_chars=budget.summary_max_content_chars,
            window_chars=budget.summary_window_chars,
            target_chars=budget.summary_target_chars,
            skip_single_child=budget.summary_skip_single_child,
        )


@dataclass
class _WindowState:
    """Per-node window bookkeeping for one agentic summarization run.

    A node whose content exceeds the payload window is asked for in several
    windows. The state remembers which payload ids a node has (all of them,
    even the ones already answered) and the summary each window produced, so
    a resumed session only asks for what is still missing and the node
    summary is always the join of *all* its windows.
    """

    entry_ids: dict[str, list[str]] = field(default_factory=dict)
    summaries: dict[str, dict[str, str]] = field(default_factory=dict)

    def ids_for(self, node_id: str) -> list[str]:
        """All payload ids of ``node_id`` (empty when the node has none)."""
        return self.entry_ids.get(node_id, [])

    def done_for(self, node_id: str) -> dict[str, str]:
        """Window summaries recorded for ``node_id``."""
        return self.summaries.setdefault(node_id, {})

    def is_complete(self, node_id: str) -> bool:
        """True when every window of ``node_id`` has a summary."""
        ids = self.ids_for(node_id)
        if not ids:
            return False
        done = self.summaries.get(node_id, {})
        return all(entry_id in done and done[entry_id].strip() for entry_id in ids)


class NodeSummarizer:
    """Generate bottom-up summaries for internal nodes of a :class:`ChunkTree`.

    Parameters
    ----------
    llm_client:
        LLM client for abstractive summaries.  When ``None``, an extractive
        fallback is used.
    cache_store:
        Optional cache for summary persistence across runs.
    content_policy:
        Content shaping policy (:class:`SummarizerContentPolicy`) — the
        characters per summary call, the characters per agentic payload
        window, the target summary length, and whether single-child nodes
        are skipped. It is built from ``compile.budget``; no number is
        duplicated in this module.
    session_policy:
        Batching/retry policy for the agentic summarizer. ``None`` keeps the
        pre-batching behaviour: one session for the whole tree with no stall
        retries. The pipeline always passes the configured policy.
    cache_ignore_model:
        Drop the model from the summary cache key, so summaries produced by
        a previous model are reused after a model switch
        (``compile.cache_ignore_model``). Default off: the model is part of
        the key.
    memory_fingerprint:
        Fingerprint of the persistent agent-notebook index
        (``compile.llm.memory_store = "vault"``), or ``None`` for the
        session store. When set, cached summaries carry it: editing the
        notebook invalidates them like a model switch does.
    """

    def __init__(
        self,
        agent: SummarizerAgent | AgenticSummarizerAgent | None = None,
        cache_store: CacheStore | None = None,
        *,
        content_policy: SummarizerContentPolicy,
        session_policy: SummarizerSessionPolicy | None = None,
        cache_ignore_model: bool = False,
        memory_fingerprint: str | None = None,
    ) -> None:
        self._agent = agent
        self._cache = cache_store
        self._policy = content_policy
        self._session_policy = session_policy
        self._payload_chars = session_policy.payload_chars if session_policy else 0
        self._cache_ignore_model = cache_ignore_model
        self._memory_fingerprint = memory_fingerprint
        self._loader = get_template_loader()

    def summarize(self, tree: ChunkTree) -> ChunkTree:
        """Summarise all eligible internal nodes in ``tree`` (in-place).

        Returns the same tree object for chaining.
        Trees with ``SINGLE_PASS`` strategy are returned unchanged.
        """
        if tree.strategy == ChunkStrategy.SINGLE_PASS:
            return tree

        if isinstance(self._agent, AgenticSummarizerAgent):
            return self._summarize_agentic(tree)

        for node in self._eligible_nodes(tree):
            node.summary = self._summarize_node(tree, node)

        return tree

    def _eligible_nodes(self, tree: ChunkTree) -> list[ChunkNode]:
        """Internal nodes awaiting a summary, bottom-up.

        Single-child wrappers inherit their child's summary instead of costing
        a call; they are not eligible themselves.
        """
        eligible: list[ChunkNode] = []
        for node in reversed(list(tree.walk())):
            if node.chunk is not None or node.structured is not None:
                continue
            if self._policy.skip_single_child and len(node.children_ids) <= 1:
                if not node.children_ids:
                    continue
                child = tree.nodes.get(node.children_ids[0])
                if child is not None:
                    node.summary = child.summary or ""
                continue
            eligible.append(node)
        return eligible

    def _summarize_agentic(self, tree: ChunkTree) -> ChunkTree:
        """Summarize all eligible internal nodes in agentic session(s).

        Nodes are asked in bounded batches (``session_policy.batch_size``), so a
        session that ends without calling ``summarize_node()`` — a small model
        answering in prose instead of using its tools — strands only its own
        batch. A stalled batch is retried up to
        ``session_policy.stall_retries`` times; the rest of the tree keeps
        going either way. Each session gets its own payload and request budget
        (see ``AgenticSummarizerAgent._usage_limits_for``), and only ids in that
        session's payload count as progress, so a model that invents or mangles
        ids cannot spin the loop.
        """
        if not isinstance(self._agent, AgenticSummarizerAgent):
            raise TypeError("AgenticSummarizerAgent expected")
        agent = self._agent
        eligible = self._eligible_nodes(tree)
        if not eligible:
            return tree

        order = [node.id for node in eligible]
        pending: set[str] = set(order)
        strikes: dict[str, int] = {}
        state = _WindowState()
        batch_size = self._session_policy.batch_size if self._session_policy else None
        stall_retries = self._session_policy.stall_retries if self._session_policy else 0
        sessions = 0
        last_session_recorded = True

        while pending:
            queue = [tree.nodes[node_id] for node_id in order if node_id in pending]
            limit = batch_size or len(queue)
            for start in range(0, len(queue), limit):
                window = queue[start : start + limit]
                payload = self._agentic_payload(tree, window, state)
                payload_ids = {entry["id"] for entry in payload}
                payload_node_ids = {self._base_node_id(entry_id) for entry_id in payload_ids}
                # Cache hits were applied by ``_agentic_payload`` — no session needed.
                pending.difference_update(node.id for node in window if node.id not in payload_node_ids)
                if not payload:
                    continue

                sessions += 1
                deps = SummarizerDeps(nodes=payload, source_id=tree.root_id)
                try:
                    agent.run(deps)
                except UsageLimitExceeded as exc:
                    logger.warning(
                        "Agentic summarizer hit its request budget (%s) — resuming for the remaining node(s)",
                        exc,
                    )
                except Exception as exc:
                    logger.warning("Agentic summarizer failed: %s", exc)
                progressed = self._apply_agentic_results(window, payload, deps.results, state)

                completed = {node_id for node_id in payload_node_ids if state.is_complete(node_id)}
                if completed:
                    pending.difference_update(completed)
                if progressed:
                    last_session_recorded = True
                    # The session produced something: the nodes that are
                    # done leave the queue, the nodes that still miss a
                    # window keep their place. A session that recorded
                    # nothing is a stall — only then do the nodes lose a
                    # retry.
                    continue
                last_session_recorded = False
                for node_id in payload_node_ids:
                    strikes[node_id] = strikes.get(node_id, 0) + 1
                pending.difference_update(node_id for node_id in payload_node_ids if strikes.get(node_id, 0) > stall_retries)

        missing = [node for node in eligible if not node.summary]
        if missing:
            logger.warning(
                "Agentic summarizer left %d node(s) unsummarized after %d session(s)%s",
                len(missing),
                sessions,
                " (last session recorded nothing for its payload)" if not last_session_recorded else "",
            )
        incomplete = [node for node in eligible if node.summary and not state.is_complete(node.id)]
        if incomplete:
            logger.warning(
                "Agentic summarizer summarised %d node(s) partially — some content windows are missing",
                len(incomplete),
            )

        return tree

    def _agentic_payload(
        self,
        tree: ChunkTree,
        nodes: list[ChunkNode],
        state: _WindowState,
    ) -> list[SummaryNode]:
        """Build the per-node session payload for ``nodes``.

        Cached summaries are applied to their nodes in place and excluded from
        the payload, so a resumed session never re-asks for finished work.

        Node content is split into windows of at most ``window_chars``
        characters. Every window is asked for separately (the payload id
        carries the window index), so the model sees the whole node instead
        of a truncated head; the per-node summary is the join of its window
        summaries.
        """
        payload: list[SummaryNode] = []
        chars_used = 0
        for node in nodes:
            parts: list[str] = []
            for child_id in node.children_ids:
                child = tree.nodes.get(child_id)
                if child is None:
                    continue
                text = self._child_content(tree, child)
                if text:
                    parts.append(text)
            content = "\n\n".join(parts)
            heading_chain = " / ".join(node.heading_chain) if node.heading_chain else ""
            windows = split_text_windows(content, self._policy.window_chars) or [""]
            entry_ids = self._entry_ids(node.id, len(windows))
            state.entry_ids[node.id] = entry_ids
            done = state.done_for(node.id)
            for entry_id, window in zip(entry_ids, windows, strict=True):
                if entry_id in done:
                    continue
                cached = self._cache_lookup(window, heading_chain)
                if cached is not None:
                    done[entry_id] = cached
                    continue
                if self._payload_chars and payload and chars_used + len(window) > self._payload_chars:
                    # The payload is full: this window is asked in the next
                    # session (the node stays pending, so nothing is lost).
                    break
                payload.append(SummaryNode(id=entry_id, heading_chain=heading_chain, content=window))
                chars_used += len(window)
            if state.is_complete(node.id):
                node.summary = self._merge_window_summaries(done, entry_ids)
        return payload

    def _apply_agentic_results(
        self,
        nodes: list[ChunkNode],
        payload: list[SummaryNode],
        results: dict[str, str],
        state: _WindowState,
    ) -> set[str]:
        """Copy session summaries onto the batch's nodes and persist them in the cache.

        Returns the ids of the nodes that gained at least one window summary
        in this session — the signal that the session made progress.
        """
        content_by_id = {entry["id"]: entry["content"] for entry in payload}
        progressed: set[str] = set()
        for node in nodes:
            heading_chain = " / ".join(node.heading_chain) if node.heading_chain else ""
            done = state.done_for(node.id)
            for entry_id in state.ids_for(node.id):
                summary = results.get(entry_id, "").strip()
                if not summary:
                    continue
                done[entry_id] = summary
                progressed.add(node.id)
                content = content_by_id.get(entry_id, "")
                if content:
                    self._cache_store(content, heading_chain, summary)
            node.summary = self._merge_window_summaries(done, state.ids_for(node.id))
        return progressed

    def _summarize_node(self, tree: ChunkTree, node: ChunkNode) -> str:
        """Build content from children, then summarise via LLM or extractive fallback."""
        parts: list[str] = []
        for child_id in node.children_ids:
            child = tree.nodes.get(child_id)
            if child is None:
                continue
            text = self._child_content(tree, child)
            if text:
                parts.append(text)

        if not parts:
            return ""

        content = "\n\n".join(parts)
        heading_chain = " / ".join(node.heading_chain) if node.heading_chain else ""

        windows = split_text_windows(content, self._policy.max_content_chars)
        if not windows:
            return ""

        summaries: list[str] = []
        for window in windows:
            cached = self._cache_lookup(window, heading_chain)
            if cached is not None:
                summaries.append(cached)
                continue
            if self._agent is not None:
                summary = self._llm_summarize(window, heading_chain)
            elif len(windows) == 1:
                summary = self._extractive_summary(parts)
            else:
                summary = self._extractive_summary([window])
            self._cache_store(window, heading_chain, summary)
            summaries.append(summary)
        return "\n\n".join(part for part in summaries if part)

    def _llm_summarize(self, content: str, heading_chain: str) -> str:
        """Call LLM via the node-summary template."""
        prompt = self._loader.render(
            _TEMPLATE,
            content=content,
            heading_chain=heading_chain,
            target_chars=self._policy.target_chars,
        )
        deps = PipelineAgentDeps(text=prompt, source_id=heading_chain)
        try:
            if self._agent is None:
                raise RuntimeError("SummarizerAgent not initialized")
            result = cast("SummarizerAgent", self._agent).run(deps)
            return result.strip()
        except Exception as exc:
            logger.warning("Summarizer failed for '%s': %s", heading_chain, exc)
            return self._extractive_summary([content])

    def _cache_lookup(self, content: str, heading_chain: str) -> str | None:
        if self._cache is None:
            return None
        deps = self._cache_deps(content, heading_chain)
        result = self._cache.get(_CACHE_LAYER, deps)
        if isinstance(result, str):
            return result
        return None

    def _cache_store(self, content: str, heading_chain: str, summary: str) -> None:
        if self._cache is None:
            return
        deps = self._cache_deps(content, heading_chain)
        self._cache.set(_CACHE_LAYER, deps, summary)

    def _cache_deps(self, content: str, heading_chain: str) -> dict[str, str]:
        model = ""
        if self._agent is not None:
            model = self._agent.model_string
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        deps = {
            "content_hash": content_hash,
            "heading_chain": hashlib.sha256(heading_chain.encode()).hexdigest()[:16],
        }
        if not self._cache_ignore_model:
            deps["model"] = model  # nocheck: summarizer cache deps, not wiki metadata
        if self._memory_fingerprint is not None:
            deps["memory"] = self._memory_fingerprint
        return deps

    @staticmethod
    def _entry_ids(node_id: str, window_count: int) -> list[str]:
        """Payload ids for a node's windows.

        A node with a single window keeps its plain id, so payloads and
        caches stay identical to the un-windowed case; extra windows carry a
        ``#wN`` suffix.
        """
        if window_count <= 1:
            return [node_id]
        return [f"{node_id}#w{index + 1}" for index in range(window_count)]

    @staticmethod
    def _base_node_id(entry_id: str) -> str:
        """Node id behind a payload entry id (``id#wN`` → ``id``)."""
        return entry_id.split("#w", 1)[0]

    @staticmethod
    def _merge_window_summaries(done: dict[str, str], entry_ids: list[str]) -> str:
        """Join the window summaries of one node in window order."""
        return "\n\n".join(done[entry_id].strip() for entry_id in entry_ids if entry_id in done and done[entry_id].strip())

    @staticmethod
    def _child_content(tree: ChunkTree, child: ChunkNode) -> str:
        """Return text representation of a child node for summary input."""
        if child.summary:
            return child.summary
        if child.chunk is not None:
            return child.chunk.text
        if child.structured is not None:
            return structured_to_text(child.structured)
        return ""

    @staticmethod
    def _extractive_summary(parts: list[str]) -> str:
        """Fallback: first sentence from each child part, joined."""
        sentences: list[str] = []
        for part in parts:
            sents = _SENTENCE_SPLIT.split(part.strip())
            if sents:
                first = sents[0].strip()
                if first:
                    sentences.append(first)
            if len(sentences) >= 5:
                break
        return " ".join(sentences)


#: ``structured_to_text`` lives in ``kompend.core.models.structured`` — importing
#: it from here would pull the agent stack into the models layer and re-create
#: the chunking/agents/compilation circular import.
