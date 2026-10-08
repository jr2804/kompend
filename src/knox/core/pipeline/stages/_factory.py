"""Pipeline factory — builds WikiPipeline with default components."""

from __future__ import annotations

from knox.agents.context_chunk_agentic import (
    AgenticContextChunkAgent,
    make_agentic_context_chunk_agent,
)
from knox.agents.summarizer_agentic import AgenticSummarizerAgent, make_agentic_summarizer_agent
from knox.chunking.contextual import ContextualEnricher
from knox.chunking.hierarchical import HierarchicalChunker
from knox.chunking.summarizer import NodeSummarizer
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.compilation.two_phase import TwoPhaseCompiler
from knox.compilation.wicer import WiCERCompiler
from knox.core.cache.store import CacheStore
from knox.core.config.rumdl import resolve_rumdl_config
from knox.core.pipeline.config import Compiler, PipelineConfig, Tier
from knox.core.pipeline.stages._parse import build_resolver_chain
from knox.core.pipeline.stages._pipeline import WikiPipeline
from knox.llm.context_window import discover_context_window
from knox.logging import get_logger
from knox.output.base import BaseOutputWriter
from knox.preprocessing.parsers._registry import build_default_router
from knox.resolution.base import BaseResolver

logger = get_logger(__name__)


def build_pipeline(
    config: PipelineConfig,
) -> WikiPipeline:
    """Shared factory used by both CLI and MCP.

    Constructs a WikiPipeline with default components:
    - Parser chain selected by ``config.parser`` (comma-separated backend names;
      default ``"liteparse"``; first matching backend per file wins)
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
    """
    writer_cls = BaseOutputWriter.get(config.output.format)
    if writer_cls is None:
        available = ", ".join(BaseOutputWriter.available_formats())
        msg = f"Unknown output format {config.output.format!r}. Available: {available}"
        raise ValueError(msg)

    router = build_default_router(config.parser_options, name=config.parser)

    # T0 (index) and T1 (parse) run without an LLM — ``knox ingest``
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
            retries=config.llm.output_retries if config.llm.output_retries_enabled else 0,
            output_retries=config.llm.agentic_output_retries if config.llm.output_retries_enabled else 0,
            context_window=cw.max_input,
            rlm=config.llm.rlm,
        )
        if config.search.contextual.enabled:
            enricher_agent = make_agentic_context_chunk_agent(
                model_string=model,
                timeout=config.llm.timeout,
                skills=config.skills,
                retries=config.llm.output_retries if config.llm.output_retries_enabled else 0,
                output_retries=config.llm.agentic_output_retries if config.llm.output_retries_enabled else 0,
                context_window=cw.max_input,
                rlm=config.llm.rlm,
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
                ),
                context_window=cw,
            )
        else:
            msg = f"Unknown compiler: {config.compiler!r}"
            raise ValueError(msg)

    return WikiPipeline(
        config=config,
        router=router,
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
            references_file=config.paths.references_file,
            assets_file=config.paths.assets_file,
            base_dir=config.paths.base_dir,
            output_dir_name=config.paths.output_dir_name,
        ),
        summarizer=NodeSummarizer(
            agent=summarizer_agent,
            cache_store=CacheStore(config.paths.cache_dir),
            max_content_chars=config.budget.summary_max_content_chars,
            target_chars=config.budget.summary_target_chars,
            skip_single_child=config.budget.summary_skip_single_child,
        ),
        enricher=ContextualEnricher(agent=enricher_agent) if enricher_agent is not None else None,
    )
