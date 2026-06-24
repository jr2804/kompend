from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from knox.core.models import WikiNode

logger = logging.getLogger(__name__)

# Absolute token reserves for retrieval context budgeting.  Replaces an earlier
# percentage-based safety margin that wasted ~150K tokens at 1M-context scale
# while only covering ~500-token-prompt overhead at 8K scale.  Absolute reserves
# scale correctly: prompt and response sizes are bounded, not proportional to
# the model's window.
_DEFAULT_SYSTEM_PROMPT_RESERVE: Final[int] = 4096
_DEFAULT_RESPONSE_RESERVE: Final[int] = 4096


class RetrievalMode(StrEnum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    TARGETED = "targeted"


@dataclass
class BudgetPlan:
    mode: RetrievalMode
    node_ids: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    available_tokens: int = 0
    depth: int = 0
    rationale: str = ""


class ContextBudgetManager:
    """Calculate available token budget and plan retrieval strategy.

    The budget is: ``max_input - system_prompt_reserve - response_reserve``.
    Both reserves are absolute token counts (not percentages) so retrieval
    on a 1M-token model does not surrender hundreds of thousands of tokens
    to a margin sized for an 8K model.

    The plan decides what to include at what resolution based on query type
    and candidate count.
    """

    def __init__(
        self,
        max_input_tokens: int,
        *,
        system_prompt_reserve: int | None = None,
        response_reserve: int | None = None,
    ) -> None:
        self._max_input = max_input_tokens
        self._system_reserve = system_prompt_reserve if system_prompt_reserve is not None else _DEFAULT_SYSTEM_PROMPT_RESERVE
        self._response_reserve = response_reserve if response_reserve is not None else _DEFAULT_RESPONSE_RESERVE

    @property
    def available_tokens(self) -> int:
        return max(1, self._max_input - self._system_reserve - self._response_reserve)

    def plan_zoom_in(
        self,
        candidates: list[WikiNode],
        query: str,
        *,
        top_k_summaries: int = 10,
        top_k_expand: int = 3,
    ) -> BudgetPlan:
        """Retrieve summaries first, then expand top matches to full content.

        Used when the query is broad and many candidates exist. The first pass
        retrieves section summaries (cheap), the second pass expands the most
        promising sections to full leaf content.
        """
        if not candidates:
            return BudgetPlan(mode=RetrievalMode.ZOOM_IN, available_tokens=self.available_tokens)

        summary_nodes = [n for n in candidates if n.depth > 0 and n.parent_id is not None]
        summary_nodes = summary_nodes[:top_k_summaries]

        expand_ids: list[str] = []
        for node in summary_nodes[:top_k_expand]:
            expand_ids.append(node.id)
            expand_ids.extend(node.children_ids)

        est = self._estimate_tokens_for_nodes(candidates, expand_ids)
        return BudgetPlan(
            mode=RetrievalMode.ZOOM_IN,
            node_ids=expand_ids,
            estimated_tokens=est,
            available_tokens=self.available_tokens,
            depth=1,
            rationale=f"Zoom-in: {len(summary_nodes)} summaries, expand top {top_k_expand}",
        )

    def plan_zoom_out(
        self,
        candidates: list[WikiNode],
        query: str,
        *,
        max_sections: int = 5,
    ) -> BudgetPlan:
        """Too many leaf hits — collapse to section summaries.

        Used when the query returns many leaf-level hits. Instead of flooding
        context with snippets, retrieve section-level summaries and let the
        LLM decide which sections to explore further.
        """
        if not candidates:
            return BudgetPlan(mode=RetrievalMode.ZOOM_OUT, available_tokens=self.available_tokens)

        top_level: dict[str, WikiNode] = {}
        for node in candidates:
            if node.parent_id is None:
                top_level[node.id] = node
            elif node.parent_id not in top_level:
                parent = self._find_node(candidates, node.parent_id)
                if parent is not None:
                    top_level[parent.id] = parent

        selected = list(top_level.values())[:max_sections]
        section_ids = [n.id for n in selected]

        est = self._estimate_tokens_for_nodes(candidates, section_ids)
        return BudgetPlan(
            mode=RetrievalMode.ZOOM_OUT,
            node_ids=section_ids,
            estimated_tokens=est,
            available_tokens=self.available_tokens,
            depth=0,
            rationale=f"Zoom-out: {len(selected)} section summaries from {len(candidates)} leaf hits",
        )

    def plan_targeted(
        self,
        candidates: list[WikiNode],
        query: str,
        *,
        top_k: int = 5,
    ) -> BudgetPlan:
        """Query mentions table/equation — direct structured query, no prose.

        Retrieves only nodes that contain structured data (tables, equations)
        matching the query terms, skipping prose-only nodes entirely.
        """
        if not candidates:
            return BudgetPlan(mode=RetrievalMode.TARGETED, available_tokens=self.available_tokens)

        structured_nodes: list[WikiNode] = []
        for node in candidates:
            if node.structured_sections:
                structured_nodes.append(node)

        if not structured_nodes:
            return BudgetPlan(
                mode=RetrievalMode.TARGETED,
                available_tokens=self.available_tokens,
                rationale="No structured data found in candidates",
            )

        selected = structured_nodes[:top_k]
        ids = [n.id for n in selected]
        est = self._estimate_tokens_for_nodes(candidates, ids)
        return BudgetPlan(
            mode=RetrievalMode.TARGETED,
            node_ids=ids,
            estimated_tokens=est,
            available_tokens=self.available_tokens,
            depth=0,
            rationale=f"Targeted: {len(selected)} structured nodes from {len(candidates)} candidates",
        )

    def plan(
        self,
        candidates: list[WikiNode],
        query: str,
        *,
        top_k: int = 5,
    ) -> BudgetPlan:
        """Auto-select the best retrieval mode based on candidate count and query.

        Heuristic:
        - Few candidates (<= top_k): ZOOM_IN (get full content)
        - Many candidates (> top_k * 3): ZOOM_OUT (collapse to summaries)
        - Query mentions table/equation keywords: TARGETED
        """
        query_lower = query.lower()
        structured_keywords = {"table", "equation", "formula", "figure", "chart", "graph"}
        if any(kw in query_lower for kw in structured_keywords):
            return self.plan_targeted(candidates, query, top_k=top_k)

        if len(candidates) <= top_k:
            return self.plan_zoom_in(candidates, query, top_k_summaries=top_k, top_k_expand=top_k)

        return self.plan_zoom_out(candidates, query, max_sections=top_k)

    @staticmethod
    def _estimate_tokens_for_nodes(candidates: list[WikiNode], node_ids: list[str]) -> int:
        node_map = {n.id: n for n in candidates}
        total = 0
        for nid in node_ids:
            node = node_map.get(nid)
            if node is not None:
                total += max(1, len(node.content or "") // 4)
        return total

    @staticmethod
    def _find_node(candidates: list[WikiNode], node_id: str) -> WikiNode | None:
        for n in candidates:
            if n.id == node_id:
                return n
        return None
