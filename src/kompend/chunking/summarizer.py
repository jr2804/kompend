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
- One :class:`SummarizerAgent` session covers the eligible nodes; the
  payload is pre-windowed and cached windows are already applied.
- Summaries are cached by content hash + model name so unchanged subtrees
  are not re-summarised across runs. ``cache_ignore_model`` drops the model
  from the key (``compile.cache_ignore_model``): after a model switch the
  previous model's summaries are reused instead of re-derived.
- Sessions are bounded per batch (``SummarizerSessionPolicy``), so a session
  that exhausts its request budget or ends without recording anything only
  loses its own batch; the remaining batches still run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from pydantic_ai.exceptions import OutputTokenLimitExceeded, UsageLimitExceeded

from kompend.agents.failure import AgentFailureKind, classify_agent_failure
from kompend.agents.summarizer import SummarizerAgent
from kompend.agents.types import SummarizerDeps, SummaryNode
from kompend.core.cache.store import CacheStore
from kompend.core.config._configs import BudgetConfig
from kompend.core.models import ChunkNode, ChunkStrategy, ChunkTree
from kompend.core.models.structured import structured_to_text
from kompend.core.templates import get_template_loader
from kompend.core.text.windows import split_text_windows
from kompend.core.types import ReasonCode
from kompend.llm.overflow import stated_context_size
from kompend.llm.request_budget import tokens_to_chars
from kompend.logging import get_logger

_CACHE_LAYER = "summary"


logger = get_logger(__name__)


def _is_oversize_payload(exc: BaseException) -> bool:
    """True when *exc* means the session payload exceeded a provider limit.

    Covers both sides of the budget: a named input-window overflow
    (:data:`AgentFailureKind.CONTEXT_OVERFLOW`) and pydantic-ai's
    :class:`OutputTokenLimitExceeded` (``finish_reason == 'length'`` — the
    provider's default output ceiling, raised with no usable output). Both
    are deterministic for a byte-identical payload, so the caller splits
    the batch instead of spending a stall retry on the same request.
    """
    if classify_agent_failure(exc) is AgentFailureKind.CONTEXT_OVERFLOW:
        return True
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OutputTokenLimitExceeded):
            return True
        current = current.__cause__ or current.__context__
    return False


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
        window_chars: Characters per payload window. One window costs one
            summarize call, so a bigger window means fewer calls (faster)
            — it never drops content.
        skip_single_child: When ``True``, an internal node with exactly one
            child is skipped — the child's content already represents it.
    """

    window_chars: int
    skip_single_child: bool

    @classmethod
    def from_config(cls, budget: BudgetConfig) -> SummarizerContentPolicy:
        """Build the policy from the ``compile.budget`` config section."""
        return cls(
            window_chars=budget.summary_window_chars,
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


@dataclass
class _SplitState:
    """Mutable oversize-split budgets for one ``_summarize_tree`` run.

    A provider refusal (input window or output ceiling) is deterministic for
    a byte-identical payload, so the run halves the batch node limit and the
    payload character budget instead of spending a stall retry. ``learned``
    carries a provider-stated window converted to characters; it only ever
    lowers the budget and is announced once.
    """

    batch_limit: int | None
    payload_budget: int
    learned_chars: int | None = None
    learned_announced: bool = False


class NodeSummarizer:
    """Generate bottom-up summaries for internal nodes of a :class:`ChunkTree`.

    Parameters
    ----------
    agent:
        The summarizer agent. There is no heuristic stand-in: a node summary
        is model output, and a failed call raises.
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
    """

    def __init__(
        self,
        agent: SummarizerAgent | None,
        cache_store: CacheStore | None = None,
        *,
        content_policy: SummarizerContentPolicy,
        session_policy: SummarizerSessionPolicy | None = None,
        cache_ignore_model: bool = False,
    ) -> None:
        if agent is None:
            raise ValueError("NodeSummarizer requires an agent - summaries are model output, not a heuristic")
        self._agent = agent
        self._cache = cache_store
        self._policy = content_policy
        self._session_policy = session_policy
        self._payload_chars = session_policy.payload_chars if session_policy else 0
        self._cache_ignore_model = cache_ignore_model
        self._loader = get_template_loader()

    def summarize(self, tree: ChunkTree) -> ChunkTree:
        """Summarise all eligible internal nodes in ``tree`` (in-place).

        Returns the same tree object for chaining.
        Trees with ``SINGLE_PASS`` strategy are returned unchanged.
        """
        if tree.strategy == ChunkStrategy.SINGLE_PASS:
            return tree

        return self._summarize_tree(tree)

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

    def _summarize_tree(self, tree: ChunkTree) -> ChunkTree:
        """Summarize all eligible internal nodes in bounded session(s).

        Nodes are asked in batches (``session_policy.batch_size``), so a
        session that ends without calling ``summarize_node()`` — a small model
        answering in prose instead of using its tools — strands only its own
        batch. A stalled batch is retried up to
        ``session_policy.stall_retries`` times; the rest of the tree keeps
        going either way. Each session gets its own payload and request budget
        (see ``SummarizerAgent._usage_limits_for``), and only ids in that
        session's payload count as progress, so a model that invents or mangles
        ids cannot spin the loop.
        """
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
        # Oversize split state: halving the batch and the payload budget is the
        # remedy for a payload the provider refuses (input window or output
        # ceiling) — a byte-identical re-ask would burn another full session.
        split = _SplitState(batch_limit=batch_size, payload_budget=self._payload_chars)

        while pending:
            queue = [tree.nodes[node_id] for node_id in order if node_id in pending]
            limit = split.batch_limit or len(queue)
            for start in range(0, len(queue), limit):
                window = queue[start : start + limit]
                budget = split.payload_budget
                if split.learned_chars is not None and (not budget or split.learned_chars < budget):
                    budget = split.learned_chars
                payload = self._agentic_payload(tree, window, state, payload_chars=budget)
                payload_ids = {entry["id"] for entry in payload}
                payload_node_ids = {self._base_node_id(entry_id) for entry_id in payload_ids}
                # Cache hits were applied by ``_agentic_payload`` — no session needed.
                pending.difference_update(node.id for node in window if node.id not in payload_node_ids)
                if not payload:
                    continue

                sessions += 1
                deps = SummarizerDeps(nodes=payload, source_id=tree.root_id)
                oversize: BaseException | None = None
                try:
                    agent.run(deps)
                except UsageLimitExceeded as exc:
                    logger.warning(
                        "Agentic summarizer hit its request budget (%s) — resuming for the remaining node(s)",
                        exc,
                    )
                except Exception as exc:
                    if _is_oversize_payload(exc):
                        oversize = exc
                    else:
                        logger.warning("Agentic summarizer failed: %s", exc)
                if oversize is not None:
                    self._handle_oversize(window, payload, oversize, limit, split, pending)
                    break
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

    def _handle_oversize(
        self,
        window: list[ChunkNode],
        payload: list[SummaryNode],
        exc: BaseException,
        limit: int,
        split: _SplitState,
        pending: set[str],
    ) -> None:
        """Split the batch after a provider refusal instead of re-asking it.

        Halves the batch node limit and the payload character budget (floored
        at one window), learning a tighter budget when the provider stated
        its window. A single node that still does not fit at the floor is
        named and dropped from ``pending`` — the tail warning then counts it
        as unsummarized. No stall retry is spent: the failure is deterministic
        for a byte-identical payload.
        """
        node_ids = sorted(node.id for node in window)
        stated = stated_context_size(exc)
        if stated is not None:
            clamped = tokens_to_chars(stated)
            if split.learned_chars is None or clamped < split.learned_chars:
                split.learned_chars = clamped
                if not split.learned_announced:
                    split.learned_announced = True
                    logger.warning(
                        "Provider stated a %d-token window — clamping summarizer session payload to ~%d chars",
                        stated,
                        clamped,
                        extra={"reason_code": ReasonCode.PAYLOAD_TOO_LARGE},
                    )
        floor_chars = max(1, self._policy.window_chars)
        attempt_chars = sum(len(entry["content"]) for entry in payload)
        basis = min(split.payload_budget, attempt_chars) if split.payload_budget else attempt_chars
        new_limit = max(1, limit // 2)
        new_budget = max(floor_chars, basis // 2)
        if split.learned_chars is not None:
            new_budget = min(new_budget, split.learned_chars)
        if new_limit == limit and new_budget >= basis:
            logger.warning(
                "Agentic summarizer payload for %s exceeds the provider limit even at the split floor (batch 1, ~%d chars) — leaving unsummarized",
                node_ids,
                floor_chars,
                extra={"reason_code": ReasonCode.PAYLOAD_TOO_LARGE},
            )
            pending.difference_update(node_ids)
            return
        split.batch_limit = new_limit
        split.payload_budget = new_budget
        logger.warning(
            "Agentic summarizer payload for %s exceeds the provider limit — splitting without a stall retry (batch limit %d, payload budget ~%d chars)",
            node_ids,
            new_limit,
            new_budget,
            extra={"reason_code": ReasonCode.PAYLOAD_TOO_LARGE},
        )

    def _agentic_payload(
        self,
        tree: ChunkTree,
        nodes: list[ChunkNode],
        state: _WindowState,
        payload_chars: int | None = None,
    ) -> list[SummaryNode]:
        """Build the per-node session payload for ``nodes``.

        Cached summaries are applied to their nodes in place and excluded from
        the payload, so a resumed session never re-asks for finished work.

        Node content is split into windows of at most ``window_chars``
        characters. Every window is asked for separately (the payload id
        carries the window index), so the model sees the whole node instead
        of a truncated head; the per-node summary is the join of its window
        summaries. ``payload_chars`` caps the packed payload; it defaults to
        the session policy budget, and the oversize split passes a smaller
        budget so the next session asks for less.
        """
        budget = self._payload_chars if payload_chars is None else payload_chars
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
                if budget and payload and chars_used + len(window) > budget:
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


#: ``structured_to_text`` lives in ``kompend.core.models.structured`` — importing
#: it from here would pull the agent stack into the models layer and re-create
#: the chunking/agents/compilation circular import.
