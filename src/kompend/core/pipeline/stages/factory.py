"""Pipeline factory - builds WikiPipeline with default components."""

from __future__ import annotations

from collections.abc import Sequence

from kompend.agents.context_chunk import ContextChunkAgent, make_context_chunk_agent
from kompend.agents.summarizer import SummarizerAgent, make_summarizer_agent
from kompend.agents.traversal import TraversalPlan, decide_traversal_plan
from kompend.chunking.contextual import ContextualEnricher
from kompend.chunking.hierarchical import HierarchicalChunker
from kompend.chunking.outline import OutlineNode
from kompend.chunking.summarizer import (
    LevelPlan,
    NodeSummarizer,
    PlanContext,
    PlanEntry,
    PlanMode,
    SummarizerContentPolicy,
    SummarizerSessionPolicy,
    TraversalPlanner,
)
from kompend.compilation.base import BaseCompiler
from kompend.compilation.graph import GraphCompiler
from kompend.compilation.two_phase import TwoPhaseCompiler, tokens_to_chars
from kompend.compilation.wicer import WiCERCompiler
from kompend.core.cache.store import (
    SUMMARIZER_PLAN_LAYER,
    CacheStore,
    summarizer_plan_deps,
    summarizer_policy_key,
    summarizer_task_hash,
)
from kompend.core.cancel import CancelState
from kompend.core.config._configs import DecisionConfig
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.pipeline.config import Compiler, PipelineConfig, Tier
from kompend.core.pipeline.stages.parse import build_resolver_chain
from kompend.core.pipeline.stages.pipeline import WikiPipeline
from kompend.core.store import vault_db_path
from kompend.core.templates import get_template_loader
from kompend.llm.context_window import discover_context_window
from kompend.llm.decision import decision_model_string
from kompend.logging import get_logger
from kompend.output.base import BaseOutputWriter
from kompend.preprocessing.parsers import build_parser
from kompend.preprocessing.splitting import SourceSplitter
from kompend.preprocessing.splitting._build import build_splitter
from kompend.resolution.base import BaseResolver
from kompend.types import JSONDict

logger = get_logger(__name__)

_TASK_TEMPLATE = "chunking/summarizer_traversal_task.md"


def _cache_payload(level: LevelPlan) -> JSONDict:
    """Plan entries as a JSON-safe dict for the ``summarizer_plan`` layer.

    Modes are stored as their plain string values — the cache never holds
    enums, so reading it needs no enum registry.
    """
    return {
        "entries": [
            {
                "node_id": entry.node_id,
                "mode": entry.mode.value,
                "focus": entry.focus,
                "reason": entry.reason,
            }
            for entry in level.entries
        ]
    }


def _level_plan_from_cache(payload: JSONDict) -> LevelPlan:
    """Rebuild a level plan from its cached JSON shape.

    Entries that lost their shape (hand-edited cache, older schema) are
    skipped — a degraded plan degrades to EXPAND in the executor, never to
    a dropped subtree.
    """
    raw_entries = payload.get("entries", [])
    entries: list[PlanEntry] = []
    for raw in raw_entries if isinstance(raw_entries, list) else []:
        if not isinstance(raw, dict):
            continue
        node_id = raw.get("node_id")
        mode = raw.get("mode")
        focus = raw.get("focus")
        reason = raw.get("reason")
        if not isinstance(node_id, str) or not isinstance(mode, str):
            continue
        entries.append(
            PlanEntry(
                node_id=node_id,
                mode=PlanMode(mode),
                focus=focus if isinstance(focus, str) else None,
                reason=reason if isinstance(reason, str) else None,
            )
        )
    return LevelPlan(entries=tuple(entries))


def _level_plan_from_traversal(plan: TraversalPlan) -> LevelPlan:
    """Convert the planner's pydantic plan to the executor's dataclass plan."""
    entries = tuple(
        PlanEntry(
            node_id=entry.node_id,
            mode=PlanMode(entry.mode.value),
            focus=entry.focus,
            reason=entry.reason,
        )
        for entry in plan.entries
    )
    return LevelPlan(entries=entries)


class _TraversalPlannerAdapter:
    """Executor-side planner backed by the decision model.

    Implements the :class:`TraversalPlanner` protocol from
    ``kompend.chunking.summarizer`` on top of :func:`decide_traversal_plan`:
    one decision-model call per level, the resulting plan cached in the
    ``summarizer_plan`` cache layer keyed on tree identity, task statement,
    model, and budgets — a recompile of an unchanged tree never re-asks the
    decision model. The adapter owns task text and budgets; the executor
    owns only what it can compute from the tree.
    """

    def __init__(
        self,
        *,
        decision: DecisionConfig,
        model: str,
        budget_nodes: int,
        budget_windows: int,
        budget_chars: int,
        cache_store: CacheStore | None,
        include_model: bool,
        task: str,
    ) -> None:
        self._decision = decision
        self._model = model
        self._budget_nodes = budget_nodes
        self._budget_windows = budget_windows
        self._budget_chars = budget_chars
        self._cache = cache_store
        self._include_model = include_model
        self._task = task

    def plan_level(self, *, outline: Sequence[OutlineNode], depth: int, context: PlanContext) -> LevelPlan | None:
        deps = summarizer_plan_deps(
            tree_hash=context.tree_hash,
            task_hash=summarizer_task_hash(self._task, ""),
            model=self._model,
            policy=summarizer_policy_key("auto", max_nodes=self._budget_nodes, max_windows=self._budget_windows),
            include_model=self._include_model,
        )
        if self._cache is not None:
            cached = self._cache.get_dict(SUMMARIZER_PLAN_LAYER, deps)
            if cached is not None:
                return _level_plan_from_cache(cached)
        plan = decide_traversal_plan(
            self._decision,
            title=context.title,
            purpose=f"Compiled wiki for {context.title}",
            task=self._task,
            outline=list(outline),
            depth=depth,
            budget_nodes=self._budget_nodes,
            budget_windows=self._budget_windows,
            budget_chars=self._budget_chars,
            model=self._model,
        )
        if plan is None:
            return None
        level = _level_plan_from_traversal(plan)
        if self._cache is not None:
            self._cache.set(SUMMARIZER_PLAN_LAYER, deps, _cache_payload(level))
        return level


def build_pipeline(
    config: PipelineConfig,
    *,
    cancel: CancelState | None = None,
) -> WikiPipeline:
    """Shared factory used by both CLI and MCP.

    Constructs a WikiPipeline with default components:
    - The parsecraft parser, wired from ``config.parser_options``
    - HierarchicalChunker
    - Compiler selected by ``config.compiler``:
      - ``"graph"``: TwoPhaseCompiler wrapping GraphCompiler (default)
      - ``"wicer"``: WiCERCompiler wrapping TwoPhaseCompiler (so the
        CEGAR loop refines real synthesized pages, not raw chunk dumps)
    - Output writer selected by ``config.compile_target``

    Parameters
    ----------
    config:
        Pipeline configuration.
    cancel:
        Cooperative cancellation token; when set, progress-emitting loops
        raise ``KompendCancelled`` at unit boundaries after a cancel was
        requested (CLI double-ESC watcher).
    """
    writer_cls = BaseOutputWriter.get(config.output.format)
    if writer_cls is None:
        available = ", ".join(BaseOutputWriter.available_formats())
        msg = f"Unknown output format {config.output.format!r}. Available: {available}"
        raise ValueError(msg)

    parser = build_parser(config.parser_options)

    # Page-range splitting is opt-in and needs no LLM to be *decided about*: a
    # disabled block yields ``None`` and the parse stage keeps its pre-split
    # behaviour exactly (one tree, no split ledger row). An enabled block with
    # no judge reachable still yields a splitter, so the decline is recorded
    # per source instead of the setting silently doing nothing.
    splitter: SourceSplitter | None = build_splitter(
        config.splitting,
        config.decision,
        base_dir=config.paths.base_dir,
    )

    # T0 (index) and T1 (parse) run without an LLM — ``kompend ingest``
    # stops here. T2 (chunk) and beyond need the LLM for context-window
    # discovery and adaptive chunk sizing.
    needs_llm = Tier.level_of(config.stop_at_tier) >= Tier.level_of(Tier.CHUNK)

    resolver: BaseResolver | None = None
    base_compiler: BaseCompiler | None = None
    summarizer_agent: SummarizerAgent | None = None
    enricher_agent: ContextChunkAgent | None = None
    chunker_context_window = None
    summary_cache: CacheStore | None = None
    planner: TraversalPlanner | None = None
    if needs_llm:
        model = config.llm.default_model
        if not model:
            msg = f"LLM model required for tier {config.stop_at_tier}"
            raise ValueError(msg)

        cw = discover_context_window(
            model,
            llm_config=config.llm,
        )
        chunker_context_window = cw
        summary_cache = CacheStore(config.paths.cache_dir)
        if config.budget.summary_traversal == "auto":
            traversal_model = config.budget.summary_traversal_model or decision_model_string(config.decision)
            if traversal_model is not None:
                planner = _TraversalPlannerAdapter(
                    decision=config.decision,
                    model=traversal_model,
                    budget_nodes=config.budget.summary_traversal_max_nodes,
                    budget_windows=config.budget.summary_traversal_max_windows,
                    # The same content budget one session payload may carry:
                    # a plan can only shape what a session could summarize.
                    budget_chars=tokens_to_chars(
                        cw.input_budget(),
                        chars_per_token=config.llm.default_chars_per_token,
                    ),
                    cache_store=summary_cache,
                    include_model=not config.cache_ignore_model,
                    task=get_template_loader().get_text(_TASK_TEMPLATE),
                )
        summarizer_agent = make_summarizer_agent(
            model_string=model,
            timeout=config.llm.timeout,
            skills=config.skills,
            context_window=cw.max_input,
            requests_per_node=config.budget.summary_requests_per_node,
            rlm=config.llm.rlm,
            llm=config.llm,
            memory_dir=config.paths.agent_memory_dir,
        )
        if config.search.contextual.enabled:
            enricher_agent = make_context_chunk_agent(
                model_string=model,
                timeout=config.llm.timeout,
                skills=config.skills,
                context_window=cw.max_input,
                rlm=config.llm.rlm,
                llm=config.llm,
            )
        resolver = build_resolver_chain(config)

        if config.compiler == Compiler.WICER:
            base_compiler = WiCERCompiler(
                config=config,
                base_compiler=TwoPhaseCompiler(
                    config=config,
                    base=GraphCompiler(
                        model=model,
                        min_content_length=config.budget.min_content_length,
                        min_concept_sources=config.budget.min_concept_sources,
                        max_concept_pages=config.budget.max_concept_pages,
                        display=config.display,
                    ),
                    context_window=cw,
                ),
                context_window=cw,
            )
        elif config.compiler == Compiler.GRAPH:
            base_compiler = TwoPhaseCompiler(
                config=config,
                base=GraphCompiler(
                    model=model,
                    min_content_length=config.budget.min_content_length,
                    min_concept_sources=config.budget.min_concept_sources,
                    max_concept_pages=config.budget.max_concept_pages,
                    display=config.display,
                ),
                context_window=cw,
            )
        else:
            msg = f"Unknown compiler: {config.compiler!r}"
            raise ValueError(msg)

    return WikiPipeline(
        config=config,
        parser=parser,
        cancel=cancel,
        splitter=splitter,
        chunker=HierarchicalChunker(
            context_window=chunker_context_window,
            chunk_size_min=config.budget.chunk_size_min,
            chunk_size_max=config.budget.chunk_size_max,
            summary_chunk_threshold=config.budget.summary_chunk_threshold,
        ),
        resolver=resolver,
        compiler=base_compiler,
        output_writer=writer_cls(
            flavor=config.output.flavor,
            frontmatter_defaults=config.output.frontmatter or {},
            rumdl_config=resolve_rumdl_config(config.paths.base_dir, config.output.rumdl_config),
            max_related_links=config.output.max_related_links,
            description_max_chars=config.display.description_max_chars,
            references_file=config.paths.references_file,
            assets_file=config.paths.assets_file,
            base_dir=config.paths.base_dir,
            output_dir_name=config.paths.output_dir_name,
        ),
        summarizer=NodeSummarizer(
            agent=summarizer_agent,
            cache_store=summary_cache,
            content_policy=SummarizerContentPolicy.from_config(config.budget),
            cache_ignore_model=config.cache_ignore_model,
            session_policy=SummarizerSessionPolicy(
                batch_size=config.budget.summary_session_batch_size,
                stall_retries=config.budget.summary_session_stall_retries,
                # The window dictates how much node content one payload may
                # carry; a bigger budget here is a request that cannot fit.
                payload_chars=tokens_to_chars(
                    cw.input_budget(),
                    chars_per_token=config.llm.default_chars_per_token,
                ),
            ),
            planner=planner,
            ledger_db_path=vault_db_path(config.paths.cache_dir),
        ),
        enricher=ContextualEnricher(agent=enricher_agent) if enricher_agent is not None else None,
    )
