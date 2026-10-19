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
  are not re-summarised across runs.
- Agentic sessions are bounded per batch (``SummarizerSessionPolicy``), so a
  session that exhausts its request budget or ends without recording anything
  only loses its own batch; the remaining batches still run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import cast

from pydantic_ai.exceptions import UsageLimitExceeded

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.summarizer import SummarizerAgent
from kompend.agents.summarizer_agentic import AgenticSummarizerAgent
from kompend.agents.types import SummarizerDeps, SummaryNode
from kompend.core.cache.store import CacheStore
from kompend.core.models import ChunkNode, ChunkStrategy, ChunkTree
from kompend.core.models.structured import structured_to_text
from kompend.core.templates import get_template_loader
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
    """

    batch_size: int
    stall_retries: int


class NodeSummarizer:
    """Generate bottom-up summaries for internal nodes of a :class:`ChunkTree`.

    Parameters
    ----------
    llm_client:
        LLM client for abstractive summaries.  When ``None``, an extractive
        fallback is used.
    cache_store:
        Optional cache for summary persistence across runs.
    max_content_chars:
        Maximum characters of child content assembled for a single summary
        prompt.  Content exceeding this is truncated.
    target_chars:
        Target summary length (informational, passed to LLM via template).
    skip_single_child:
        When ``True``, internal nodes with exactly one child are skipped —
        the child's content already represents the parent fully.
    session_policy:
        Batching/retry policy for the agentic summarizer. ``None`` keeps the
        pre-batching behaviour: one session for the whole tree with no stall
        retries. The pipeline always passes the configured policy.
    """

    def __init__(
        self,
        agent: SummarizerAgent | AgenticSummarizerAgent | None = None,
        cache_store: CacheStore | None = None,
        *,
        max_content_chars: int = 8000,
        target_chars: int = 600,
        skip_single_child: bool = True,
        session_policy: SummarizerSessionPolicy | None = None,
    ) -> None:
        self._agent = agent
        self._cache = cache_store
        self._max_content_chars = max_content_chars
        self._target_chars = target_chars
        self._skip_single_child = skip_single_child
        self._session_policy = session_policy
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
            if self._skip_single_child and len(node.children_ids) <= 1:
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
        batch_size = self._session_policy.batch_size if self._session_policy else None
        stall_retries = self._session_policy.stall_retries if self._session_policy else 0
        sessions = 0
        last_session_recorded = True

        while pending:
            queue = [tree.nodes[node_id] for node_id in order if node_id in pending]
            limit = batch_size or len(queue)
            for start in range(0, len(queue), limit):
                window = queue[start : start + limit]
                payload = self._agentic_payload(tree, window)
                payload_ids = {entry["id"] for entry in payload}
                # Cache hits were applied by ``_agentic_payload`` — no session needed.
                pending.difference_update(node.id for node in window if node.id not in payload_ids)
                if not payload:
                    continue

                sessions += 1
                deps = SummarizerDeps(nodes=payload, source_id=tree.root_id, timeout=agent.timeout)
                try:
                    agent.run(deps)
                except UsageLimitExceeded as exc:
                    logger.warning(
                        "Agentic summarizer hit its request budget (%s) — resuming for the remaining node(s)",
                        exc,
                    )
                except Exception as exc:
                    logger.warning("Agentic summarizer failed: %s", exc)
                self._apply_agentic_results(window, payload, deps.results)

                recorded = {node_id for node_id in payload_ids if deps.results.get(node_id, "").strip()}
                if recorded:
                    pending.difference_update(recorded)
                    last_session_recorded = True
                    continue
                last_session_recorded = False
                for node_id in payload_ids:
                    strikes[node_id] = strikes.get(node_id, 0) + 1
                pending.difference_update(node_id for node_id in payload_ids if strikes[node_id] > stall_retries)

        missing = [node for node in eligible if not node.summary]
        if missing:
            logger.warning(
                "Agentic summarizer left %d node(s) unsummarized after %d session(s)%s",
                len(missing),
                sessions,
                " (last session recorded nothing for its payload)" if not last_session_recorded else "",
            )

        return tree

    def _agentic_payload(self, tree: ChunkTree, nodes: list[ChunkNode]) -> list[SummaryNode]:
        """Build the per-node session payload for ``nodes``.

        Cached summaries are applied to their nodes in place and excluded from
        the payload, so a resumed session never re-asks for finished work.
        """
        payload: list[SummaryNode] = []
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
            if len(content) > self._max_content_chars:
                content = content[: self._max_content_chars]
            heading_chain = " / ".join(node.heading_chain) if node.heading_chain else ""
            cached = self._cache_lookup(content, heading_chain)
            if cached is not None:
                node.summary = cached
                continue
            payload.append(SummaryNode(id=node.id, heading_chain=heading_chain, content=content))
        return payload

    def _apply_agentic_results(
        self,
        nodes: list[ChunkNode],
        payload: list[SummaryNode],
        results: dict[str, str],
    ) -> None:
        """Copy session summaries onto the batch's nodes and persist them in the cache."""
        content_by_id = {entry["id"]: entry["content"] for entry in payload}
        for node in nodes:
            summary = results.get(node.id)
            if not summary:
                continue
            node.summary = summary
            heading_chain = " / ".join(node.heading_chain) if node.heading_chain else ""
            content = content_by_id.get(node.id, "")
            if content:
                self._cache_store(content, heading_chain, summary)

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
        if len(content) > self._max_content_chars:
            content = content[: self._max_content_chars]

        heading_chain = " / ".join(node.heading_chain) if node.heading_chain else ""

        cached = self._cache_lookup(content, heading_chain)
        if cached is not None:
            return cached

        summary = self._llm_summarize(content, heading_chain) if self._agent is not None else self._extractive_summary(parts)

        self._cache_store(content, heading_chain, summary)
        return summary

    def _llm_summarize(self, content: str, heading_chain: str) -> str:
        """Call LLM via the node-summary template."""
        prompt = self._loader.render(_TEMPLATE, content=content, heading_chain=heading_chain)
        deps = PipelineAgentDeps(text=prompt, source_id=heading_chain, timeout=120)
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
        return {
            "content_hash": content_hash,
            "heading_chain": hashlib.sha256(heading_chain.encode()).hexdigest()[:16],
            "model": model,  # nocheck: summarizer cache deps, not wiki metadata
        }

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
