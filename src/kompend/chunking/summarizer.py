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
import sqlite3
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic_ai.exceptions import OutputTokenLimitExceeded, UsageLimitExceeded

from kompend.agents.failure import AgentFailureKind, classify_agent_failure
from kompend.agents.summarizer import SummarizerAgent
from kompend.agents.types import SummarizerDeps, SummaryNode
from kompend.chunking.outline import OutlineNode, build_outline
from kompend.core.cache.store import CacheStore, summarizer_tree_hash
from kompend.core.config._configs import BudgetConfig
from kompend.core.models import ChunkNode, ChunkStrategy, ChunkTree
from kompend.core.models.structured import structured_to_text
from kompend.core.reporting.ledger import stage_ledger
from kompend.core.templates import get_template_loader
from kompend.core.text.windows import split_text_windows
from kompend.core.types import ReasonCode, StageOutcome
from kompend.llm.overflow import stated_context_size
from kompend.llm.request_budget import tokens_to_chars
from kompend.logging import get_logger

_CACHE_LAYER = "summary"


logger = get_logger(__name__)


class PlanMode(StrEnum):
    """What the executor does with one planned node.

    Mirrors the planner vocabulary (``kompend.agents.traversal.PlanMode``)
    without importing it: chunking speaks this vocabulary, an adapter in
    the pipeline layer converts the planner's pydantic plan.
    """

    EXPAND = "expand"
    SUMMARIZE = "summarize"
    SKIP = "skip"


class TraversalMode(StrEnum):
    """Whether the summarizer consults a traversal planner.

    ``OFF`` (the default) summarizes every eligible node — today's path,
    byte-identical. ``AUTO`` asks the planner per level top-down and
    executes its plan; a planner failure falls back to the full tree.
    """

    OFF = "off"
    AUTO = "auto"


@dataclass(frozen=True)
class PlanEntry:
    """One planned node: its mode plus the planner's focus and reason.

    ``focus`` scopes a SUMMARIZE branch (rendered as an extra heading line
    in the session prompt, never concatenated into content); ``reason``
    is the planner's fixed skip label, recorded with the skipped node.
    """

    node_id: str
    mode: PlanMode
    focus: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LevelPlan:
    """The plan for one tree level: one entry per planned child."""

    entries: tuple[PlanEntry, ...] = ()


@dataclass(frozen=True)
class PlanContext:
    """Tree identity the executor hands to every ``plan_level`` call.

    ``tree_hash`` keys the planner's own cache layer (the same hash the
    summarizer plan cache uses); ``title`` names the document the plan
    is for. The adapter owns task and budget text; the executor owns
    only what it can compute from the tree.
    """

    tree_hash: str
    title: str


@dataclass(frozen=True)
class _PlanView:
    """The executed plan, as the payload builder sees it.

    ``skip_ids`` holds directly SKIP-decided node ids (their content is
    excluded from the parent's payload; the whole subtree leaves the
    worklist). ``focus_by_node`` maps SUMMARIZE nodes to their focus.
    """

    skip_ids: frozenset[str]
    focus_by_node: dict[str, str]


@dataclass
class _SessionStats:
    """Session counters accumulated across one tree's ``_run_sessions`` calls."""

    sessions: int = 0
    windows: int = 0
    chars: int = 0
    unsummarized: int = 0


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
        traversal: ``OFF`` summarizes every eligible node; ``AUTO`` consults
            the traversal planner (requires a planner on the summarizer —
            without one the run stays on today's path).
        traversal_min_nodes: Smallest eligible-node count the planner is
            consulted for. Below it the planning overhead cannot pay for
            itself and the run stays on today's path.
    """

    window_chars: int
    skip_single_child: bool
    traversal: TraversalMode = TraversalMode.OFF
    traversal_min_nodes: int = 1

    @classmethod
    def from_config(cls, budget: BudgetConfig) -> SummarizerContentPolicy:
        """Build the policy from the ``compile.budget`` config section."""
        return cls(
            window_chars=budget.summary_window_chars,
            skip_single_child=budget.summary_skip_single_child,
            traversal=TraversalMode(budget.summary_traversal),
            traversal_min_nodes=budget.summary_traversal_min_nodes,
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


class TraversalPlanner(Protocol):
    """The executor-facing planner surface.

    ``plan_level`` receives the level's outline entries (one per child,
    in tree order), the level's depth, and the tree's :class:`PlanContext`;
    it returns the per-child decision or ``None`` when it cannot plan (the
    executor then treats the whole tree as unplanned and falls back).
    Title, purpose, task and budgets are the adapter's concern, not the
    executor's.
    """

    def plan_level(self, *, outline: Sequence[OutlineNode], depth: int, context: PlanContext) -> LevelPlan | None: ...


class _PlannerUnavailable(Exception):
    """Raised internally when the planner declines a level."""


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
    planner:
        Optional traversal planner (:class:`TraversalPlanner`). Only used
        when ``content_policy.traversal`` is ``AUTO``; a planner failure
        falls back to summarizing the full tree. ``None`` keeps today's
        path.
    ledger_db_path:
        Path of the stage-ledger SQLite file (``vault_db_path`` of the
        vault). When set, the planned path records one traversal row per
        tree (observer only — a ledger failure never affects the run).
        ``None`` (tests, off-path runs) records nothing.
    """

    def __init__(
        self,
        agent: SummarizerAgent | None,
        cache_store: CacheStore | None = None,
        *,
        content_policy: SummarizerContentPolicy,
        session_policy: SummarizerSessionPolicy | None = None,
        cache_ignore_model: bool = False,
        planner: TraversalPlanner | None = None,
        ledger_db_path: Path | None = None,
    ) -> None:
        if agent is None:
            raise ValueError("NodeSummarizer requires an agent - summaries are model output, not a heuristic")
        self._agent = agent
        self._cache = cache_store
        self._policy = content_policy
        self._session_policy = session_policy
        self._payload_chars = session_policy.payload_chars if session_policy else 0
        self._cache_ignore_model = cache_ignore_model
        self._planner = planner
        self._ledger_db_path = ledger_db_path
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

        With ``traversal=OFF`` (or no planner) every eligible node is
        summarized bottom-up. With ``traversal=AUTO`` and a planner, the
        plan is asked per level top-down first and only the planned nodes
        are summarized (:meth:`_summarize_planned`).
        """
        eligible = self._eligible_nodes(tree)
        if not eligible:
            return tree
        if self._policy.traversal is TraversalMode.AUTO and self._planner is not None and len(eligible) >= self._policy.traversal_min_nodes:
            return self._summarize_planned(tree, eligible)
        self._run_sessions(tree, eligible, plan_view=None, stats=_SessionStats())
        return tree

    def _summarize_planned(self, tree: ChunkTree, eligible: list[ChunkNode]) -> ChunkTree:
        """Execute a traversal plan: plan top-down, summarize bottom-up.

        The root is unconditional work; every level under an EXPAND branch
        is planned in turn. SKIP nodes leave the worklist and, after their
        nearest summarized ancestor exists, inherit that ancestor's summary.
        A planner failure at any point falls back to the full tree with a
        WARNING naming the tree and one FAILED ledger row; a completed plan
        records OK, or PARTIAL when nodes stayed unsummarized.
        """
        context = PlanContext(
            tree_hash=summarizer_tree_hash(tree.root_id, [(node.id, node.heading_chain) for node in tree.walk()]),
            # A tree without a source file (synthetic tests) is named by its root.
            title=tree.source_file.stem if tree.source_file is not None else tree.root_id,
        )
        try:
            work_ids, skip, focus_by_node, expand_count, planner_calls = self._plan_tree(tree, context)
        except Exception as exc:
            logger.warning(
                "Traversal planning failed for tree %s (%s) — summarizing the full tree",
                tree.root_id,
                exc,
            )
            self._run_sessions(tree, eligible, plan_view=None, stats=_SessionStats())
            self._record_traversal_row(tree, planned=0, skipped=0, expanded=0, windows=0, outcome=StageOutcome.FAILED)
            return tree

        work = [node for node in eligible if node.id in work_ids]
        stats = _SessionStats()
        plan_view = _PlanView(skip_ids=frozenset(skip), focus_by_node=focus_by_node)
        self._run_sessions(tree, work, plan_view=plan_view, stats=stats)
        holes = self._inherit_skipped(tree, eligible, work_ids)
        if holes:
            # A skipped node whose whole ancestor chain stayed summary-less
            # is summarized normally — never leave an empty summary behind.
            self._run_sessions(tree, holes, plan_view=plan_view, stats=stats)
        self._record_traversal_row(
            tree,
            planned=len(work),
            skipped=len(skip),
            expanded=expand_count,
            windows=stats.windows,
            outcome=StageOutcome.PARTIAL if stats.unsummarized else StageOutcome.OK,
        )
        logger.info(
            "tree=%s eligible=%d planned=%d expanded=%d skipped=%d sessions=%d windows=%d chars=%d planner_calls=%d",
            tree.root_id,
            len(eligible),
            len(work),
            expand_count,
            len(skip),
            stats.sessions,
            stats.windows,
            stats.chars,
            planner_calls,
        )
        return tree

    def _plan_tree(self, tree: ChunkTree, context: PlanContext) -> tuple[set[str], dict[str, str], dict[str, str], int, int]:
        """Ask the planner per level, top-down from the root.

        Returns the work set (EXPAND and SUMMARIZE nodes plus the root),
        the SKIP decisions as ``node_id -> reason``, the SUMMARIZE nodes'
        focus lines, the EXPAND count, and the number of planner calls.
        A child missing from its level's plan is treated as EXPAND — the
        safe default (work plus descend) — so a partial plan can never
        silently drop a subtree.
        """
        planner = self._planner
        if planner is None:
            raise _PlannerUnavailable("no planner attached")
        outline_by_id = {entry.node_id: entry for entry in build_outline(tree)}
        work_ids: set[str] = {tree.root_id}
        skip: dict[str, str] = {}
        focus_by_node: dict[str, str] = {}
        expand_count = 0
        planner_calls = 0
        frontier: deque[tuple[str, int]] = deque([(tree.root_id, 0)])
        while frontier:
            parent_id, parent_depth = frontier.popleft()
            parent = tree.nodes.get(parent_id)
            if parent is None:
                continue
            child_ids = [cid for cid in parent.children_ids if cid in tree.nodes and cid in outline_by_id]
            if not child_ids:
                continue
            entries = [outline_by_id[cid] for cid in child_ids]
            level = planner.plan_level(outline=entries, depth=parent_depth + 1, context=context)
            planner_calls += 1
            if level is None:
                raise _PlannerUnavailable(f"planner declined depth {parent_depth + 1} under {parent_id}")
            by_id: dict[str, PlanEntry] = {}
            for entry in level.entries:
                by_id.setdefault(entry.node_id, entry)
            for child_id in child_ids:
                entry = by_id.get(child_id)
                if entry is None or entry.mode is PlanMode.EXPAND:
                    work_ids.add(child_id)
                    expand_count += 1
                    frontier.append((child_id, parent_depth + 1))
                elif entry.mode is PlanMode.SUMMARIZE:
                    work_ids.add(child_id)
                    if entry.focus:
                        focus_by_node[child_id] = entry.focus
                else:
                    reason = entry.reason or PlanMode.SKIP.value
                    skip[child_id] = reason
                    logger.debug("Skipping %s by plan (%s)", child_id, reason)
        return work_ids, skip, focus_by_node, expand_count, planner_calls

    def _record_traversal_row(
        self,
        tree: ChunkTree,
        *,
        planned: int,
        skipped: int,
        expanded: int,
        windows: int,
        outcome: StageOutcome,
    ) -> None:
        """Write the one traversal ledger row for ``tree`` (observer only).

        The row correlates with the run through the ``current_run_id``
        ContextVar the ledger already holds. A store failure is swallowed
        — the ledger observes, it never gates.
        """
        if self._ledger_db_path is None:
            return
        source = tree.source_file.as_posix() if tree.source_file is not None else tree.root_id
        try:
            with stage_ledger(self._ledger_db_path) as ledger:
                ledger.record_traversal_plan(
                    source,
                    outcome=outcome,
                    planned=planned,
                    skipped=skipped,
                    expanded=expanded,
                    windows=windows,
                )
        except sqlite3.Error:
            logger.debug("Traversal ledger row for %s not recorded", tree.root_id, exc_info=True)

    def _run_sessions(
        self,
        tree: ChunkTree,
        eligible: list[ChunkNode],
        *,
        plan_view: _PlanView | None,
        stats: _SessionStats,
    ) -> None:
        """Summarize ``eligible`` in bounded session(s), bottom-up.

        Shared by the planned and the unplanned path: ``plan_view`` shapes
        the payload (skipped children excluded, focus lines attached);
        ``stats`` accumulates the tree's session metrics. The session
        mechanics — batching, cache application, stall strikes and the
        oversize split — are identical on both paths.

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
                payload = self._agentic_payload(tree, window, state, payload_chars=budget, plan_view=plan_view)
                payload_ids = {entry["id"] for entry in payload}
                payload_node_ids = {self._base_node_id(entry_id) for entry_id in payload_ids}
                # Cache hits were applied by ``_agentic_payload`` — no session needed.
                pending.difference_update(node.id for node in window if node.id not in payload_node_ids)
                if not payload:
                    continue

                sessions += 1
                stats.sessions += 1
                stats.windows += len(payload)
                stats.chars += sum(len(entry["content"]) for entry in payload)
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
            stats.unsummarized += len(missing)
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
        plan_view: _PlanView | None = None,
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

        ``plan_view`` (planned traversal only) excludes SKIP-decided children
        from the content and attaches a plan focus line to the node's payload
        entries; cache ids stay content-keyed, so a focus change alone cannot
        invalidate stored summaries.
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
                text = self._planned_child_content(tree, child, plan_view) if plan_view is not None else self._child_content(tree, child)
                if text:
                    parts.append(text)
            content = "\n\n".join(parts)
            heading_chain = " / ".join(node.heading_chain) if node.heading_chain else ""
            windows = split_text_windows(content, self._policy.window_chars) or [""]
            entry_ids = self._entry_ids(node.id, len(windows))
            state.entry_ids[node.id] = entry_ids
            done = state.done_for(node.id)
            focus = plan_view.focus_by_node.get(node.id) if plan_view is not None else None
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
                entry = SummaryNode(id=entry_id, heading_chain=heading_chain, content=window)
                if focus:
                    entry["focus"] = focus
                payload.append(entry)
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

    def _planned_child_content(self, tree: ChunkTree, child: ChunkNode, plan_view: _PlanView) -> str:
        """Planned-path counterpart of ``_child_content``.

        A SKIP-decided child contributes nothing to its parent's payload —
        that is what the skip means. A child with no decision (a transitive
        skip under a SUMMARIZE branch) has its undecided subtree inlined leaf
        by leaf instead, so summarizing the branch never silently loses the
        skipped-under-it content. Worked children stop at their summary, as
        on the unplanned path.
        """
        if child.id in plan_view.skip_ids:
            return ""
        if child.summary:
            return child.summary
        if child.chunk is not None:
            return child.chunk.text
        if child.structured is not None:
            return structured_to_text(child.structured)
        parts = [self._planned_child_content(tree, tree.nodes[cid], plan_view) for cid in child.children_ids if cid in tree.nodes]
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _inherit_skipped(tree: ChunkTree, eligible: list[ChunkNode], work_ids: set[str]) -> list[ChunkNode]:
        """Give every skipped node the nearest summarized ancestor's summary.

        Returns the nodes left without any ancestor summary — they are
        summarized normally instead of staying empty.
        """
        holes: list[ChunkNode] = []
        for node in eligible:
            if node.id in work_ids:
                continue
            inherited = ""
            current = node.parent_id
            while current is not None and not inherited:
                parent = tree.nodes.get(current)
                if parent is None:
                    break
                inherited = parent.summary.strip()
                current = parent.parent_id
            if inherited:
                node.summary = inherited
            else:
                holes.append(node)
        return holes

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


#: ``structured_to_text`` lives in ``kompend.core.models.structured`` — importing
#: it from here would pull the agent stack into the models layer and re-create
#: the chunking/agents/compilation circular import.
