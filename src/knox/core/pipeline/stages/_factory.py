"""Pipeline factory — builds WikiPipeline with default components."""

from __future__ import annotations

from knox.chunking.hierarchical import HierarchicalChunker
from knox.chunking.summarizer import NodeSummarizer
from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.compilation.two_phase import TwoPhaseCompiler
from knox.compilation.wicer import WiCERCompiler
from knox.core.cache.store import CacheStore
from knox.core.pipeline.config import TIER_LEVEL, Compiler, PipelineConfig, Tier
from knox.core.pipeline.stages._parse import build_resolver_chain
from knox.core.pipeline.stages._pipeline import WikiPipeline
from knox.llm.context_window import discover_context_window
from knox.logging import get_logger
from knox.output.base import BaseOutputWriter
from knox.preprocessing.parsers._registry import ParserBuildOptions, build_default_router
from knox.resolution.base import BaseResolver

logger = get_logger(__name__)


def build_pipeline(
    config: PipelineConfig,
) -> WikiPipeline:
    """Shared factory used by both CLI and MCP.

    Constructs a WikiPipeline with default components:
    - Parser chain selected by ``config.parser`` (comma-separated backend names;
      default ``"docling"``; first matching backend per file wins)
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

    router = build_default_router(
        ParserBuildOptions(
            images_dir=config.paths.images_dir,
            image_mode=config.image_mode,
            images_scale=config.images_scale,
            pdf_config=config.pdf,
        ),
        names=config.parser,
    )

    # T0 (index) and T1 (parse) run without an LLM — ``knox ingest``
    # stops here. T2 (chunk) and beyond need the LLM for context-window
    # discovery and adaptive chunk sizing.
    needs_llm = TIER_LEVEL[config.stop_at_tier] >= TIER_LEVEL[Tier.CHUNK]

    resolver: BaseResolver | None = None
    base_compiler: BaseCompiler | None = None
    chunker_context_window = None
    if needs_llm:
        model = config.llm.default_model
        if not model:
            msg = f"LLM model required for tier {config.stop_at_tier}"
            raise ValueError(msg)

        cw = discover_context_window(
            model,
            fallback_max_input=config.llm.fallback_max_input,
            fallback_max_output=config.llm.fallback_max_output,
            context_window_cap=config.llm.context_window_cap,
        )
        chunker_context_window = cw
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
            rumdl_config=config.output.rumdl_config,
            max_related_links=config.output.max_related_links,
        ),
        summarizer=NodeSummarizer(
            cache_store=CacheStore(config.paths.cache_dir),
            max_content_chars=config.budget.summary_max_content_chars,
            target_chars=config.budget.summary_target_chars,
            skip_single_child=config.budget.summary_skip_single_child,
        ),
    )
