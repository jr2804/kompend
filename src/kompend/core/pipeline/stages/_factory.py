"""Pipeline factory - builds WikiPipeline with default components."""

from __future__ import annotations

from pathlib import Path

from kompend.agents.context_chunk_agentic import (
    AgenticContextChunkAgent,
    make_agentic_context_chunk_agent,
)
from kompend.agents.summarizer_agentic import AgenticSummarizerAgent, make_agentic_summarizer_agent
from kompend.chunking.contextual import ContextualEnricher
from kompend.chunking.hierarchical import HierarchicalChunker
from kompend.chunking.summarizer import (
    NodeSummarizer,
    SummarizerContentPolicy,
    SummarizerSessionPolicy,
)
from kompend.compilation._two_phase_core import tokens_to_chars
from kompend.compilation.base import BaseCompiler
from kompend.compilation.graph import GraphCompiler
from kompend.compilation.two_phase import TwoPhaseCompiler
from kompend.compilation.wicer import WiCERCompiler
from kompend.core.cache.store import CacheStore
from kompend.core.cancel import CancelState
from kompend.core.config._configs import LlmMemoryStore
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.pipeline.config import Compiler, PipelineConfig, Tier
from kompend.core.pipeline.stages._parse import build_resolver_chain
from kompend.core.pipeline.stages._pipeline import WikiPipeline
from kompend.llm.context_window import discover_context_window
from kompend.logging import get_logger
from kompend.output.base import BaseOutputWriter
from kompend.preprocessing.parsers._registry import build_default_router
from kompend.resolution.base import BaseResolver

logger = get_logger(__name__)


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

    router = build_default_router(config.parser_options)

    # T0 (index) and T1 (parse) run without an LLM — ``kompend ingest``
    # stops here. T2 (chunk) and beyond need the LLM for context-window
    # discovery and adaptive chunk sizing.
    needs_llm = Tier.level_of(config.stop_at_tier) >= Tier.level_of(Tier.CHUNK)

    resolver: BaseResolver | None = None
    base_compiler: BaseCompiler | None = None
    summarizer_agent: AgenticSummarizerAgent | None = None
    enricher_agent: AgenticContextChunkAgent | None = None
    chunker_context_window = None
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
        summarizer_agent = make_agentic_summarizer_agent(
            model_string=model,
            timeout=config.llm.timeout,
            skills=config.skills,
            context_window=cw.max_input,
            requests_per_node=config.budget.summary_requests_per_node,
            rlm=config.llm.rlm,
            llm=config.llm,
            memory_dir=_vault_memory_dir(config),
        )
        if config.search.contextual.enabled:
            enricher_agent = make_agentic_context_chunk_agent(
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
        router=router,
        cancel=cancel,
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
            cache_store=CacheStore(config.paths.cache_dir),
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
        ),
        enricher=ContextualEnricher(agent=enricher_agent) if enricher_agent is not None else None,
    )


def _vault_memory_dir(config: PipelineConfig) -> Path | None:
    """The agent-memory directory, or ``None`` for the in-process notebook.

    Two switches decide: ``compile.llm.agent_memory`` turns the capability
    on and ``compile.llm.memory_store = "vault"`` moves it under
    :attr:`VaultPaths.agent_memory_dir`. Anything else keeps a fresh
    in-process store per stage run.
    """
    if not (config.llm.agent_memory and config.llm.memory_store is LlmMemoryStore.VAULT):
        return None
    return config.paths.agent_memory_dir
