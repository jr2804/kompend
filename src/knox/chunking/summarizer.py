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
"""

from __future__ import annotations

import hashlib
import re

from knox.agents.deps import PipelineAgentDeps
from knox.agents.summarizer import SummarizerAgent
from knox.core.cache.store import CacheStore
from knox.core.models import (
    ChunkNode,
    ChunkStrategy,
    ChunkTree,
    StructuredData,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)
from knox.core.templates import get_template_loader
from knox.logging import get_logger

logger = get_logger(__name__)

_TEMPLATE = "chunking/node_summary.md.j2"
_CACHE_LAYER = "summary"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


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
    """

    def __init__(
        self,
        agent: SummarizerAgent | None = None,
        cache_store: CacheStore | None = None,
        *,
        max_content_chars: int = 8000,
        target_chars: int = 600,
        skip_single_child: bool = True,
    ) -> None:
        self._agent = agent
        self._cache = cache_store
        self._max_content_chars = max_content_chars
        self._target_chars = target_chars
        self._skip_single_child = skip_single_child
        self._loader = get_template_loader()

    def summarize(self, tree: ChunkTree) -> ChunkTree:
        """Summarise all eligible internal nodes in ``tree`` (in-place).

        Returns the same tree object for chaining.
        Trees with ``SINGLE_PASS`` strategy are returned unchanged.
        """
        if tree.strategy == ChunkStrategy.SINGLE_PASS:
            return tree

        for node in reversed(list(tree.walk())):
            if node.chunk is not None:
                continue
            if node.structured is not None:
                continue
            if self._skip_single_child and len(node.children_ids) <= 1:
                if len(node.children_ids) == 0:
                    continue
                child = tree.nodes.get(node.children_ids[0])
                if child is not None:
                    node.summary = child.summary or ""
                continue
            node.summary = self._summarize_node(tree, node)

        return tree

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

        heading_path = " / ".join(node.heading_path) if node.heading_path else ""

        cached = self._cache_lookup(content, heading_path)
        if cached is not None:
            return cached

        summary = self._llm_summarize(content, heading_path) if self._agent is not None else self._extractive_summary(parts)

        self._cache_store(content, heading_path, summary)
        return summary

    def _llm_summarize(self, content: str, heading_path: str) -> str:
        """Call LLM via the node-summary template."""
        prompt = self._loader.render(_TEMPLATE, content=content, heading_path=heading_path)
        deps = PipelineAgentDeps(text=prompt, source_id=heading_path, timeout=120)
        try:
            result = self._agent.run(deps)
            return result.strip()
        except Exception as exc:
            logger.warning("Summarizer failed for '%s': %s", heading_path, exc)
            return self._extractive_summary([content])

    def _cache_lookup(self, content: str, heading_path: str) -> str | None:
        if self._cache is None:
            return None
        deps = self._cache_deps(content, heading_path)
        result = self._cache.get(_CACHE_LAYER, deps)
        if isinstance(result, str):
            return result
        return None

    def _cache_store(self, content: str, heading_path: str, summary: str) -> None:
        if self._cache is None:
            return
        deps = self._cache_deps(content, heading_path)
        self._cache.set(_CACHE_LAYER, deps, summary)

    def _cache_deps(self, content: str, heading_path: str) -> dict[str, str]:
        model = ""
        if self._agent is not None:
            model = self._agent.litellm_model
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return {
            "content_hash": content_hash,
            "heading_path": hashlib.sha256(heading_path.encode()).hexdigest()[:16],
            "model": model,
        }

    @staticmethod
    def _child_content(tree: ChunkTree, child: ChunkNode) -> str:
        """Return text representation of a child node for summary input."""
        if child.summary:
            return child.summary
        if child.chunk is not None:
            return child.chunk.text
        if child.structured is not None:
            return _structured_to_text(child.structured)
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


def _structured_to_text(s: StructuredData) -> str:
    """Flatten a :class:`StructuredData` variant to a text representation."""
    if isinstance(s, StructuredTable):
        header = " | ".join(s.columns)
        rows = [" | ".join(row.get(col, "") for col in s.columns) for row in s.rows]
        caption = f" — {s.caption}" if s.caption else ""
        return f"Table{caption}\n{header}\n" + "\n".join(rows)
    if isinstance(s, StructuredEquation):
        parts = [s.latex]
        if s.variables:
            parts.append("where " + ", ".join(f"{v.name}={v.definition}" for v in s.variables if v.definition))
        if s.caption:
            parts.append(f"({s.caption})")
        return " ".join(parts)
    if isinstance(s, StructuredFigure):
        parts = []
        if s.caption:
            parts.append(s.caption)
        if s.alt_text:
            parts.append(f"[{s.alt_text}]")
        if s.image_ref:
            parts.append(f"({s.image_ref})")
        return " ".join(parts) if parts else "[figure]"
    return ""
