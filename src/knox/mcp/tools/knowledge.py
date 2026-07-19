"""Knowledge compilation MCP tools."""

import dataclasses
from pathlib import Path
from typing import Any

from knox.agents.capture import CaptureAgent
from knox.agents.distill import DistillAgent
from knox.capture.capture import capture
from knox.capture.distill import distill
from knox.capture.learn import learn
from knox.cli._shared import resolve_project_paths
from knox.core.config import (
    DEFAULTS,
    EmbeddingConfig,
    LLMConfig,
    resolve_config,
)
from knox.core.config.keys import COMPILE_LLM_OUTPUT_RETRIES, DISPLAY_MCP_ELEMENT_CHARS, PATHS_WIKI_DIR
from knox.core.pipeline.config import Compiler, ImageMode, OutputConfig, OutputFormat, PipelineConfig
from knox.core.pipeline.orchestrator import build_pipeline
from knox.logging import get_logger
from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser

logger = get_logger(__name__)

def parse_document(file_path: str) -> dict[str, Any]:
    """Parse a document and return structured elements."""
    doc_file = Path(file_path)
    if not doc_file.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    router = ParserRouter()
    try:
        router.register(DoclingParser())
    except Exception as exc:
        logger.warning("DoclingParser not available: %s", exc, exc_info=True)

    try:
        doc = router.parse(doc_file)
        return {
            "status": "ok",
            "title": doc.title or doc_file.name,
            "page_count": doc.page_count,
            "elements": [
                {
                    "type": str(el.type),
                    "content": str(el.content)[: DEFAULTS.get(DISPLAY_MCP_ELEMENT_CHARS, 500)],
                    "page": el.page,
                    "heading_level": el.heading_level,
                }
                for el in doc.elements
            ],
            "metadata": doc.metadata,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

def compile_wiki(
    sources_dir: str,
    output: str = "./output",
    output_format: str = "markdown",
    image_mode: str = "file",
    urls: list[str] | None = None,
    compiler: str = "graph",
    vlm_model: str = "",
) -> dict[str, Any]:
    """Compile sources into a knowledge wiki.

    Parameters
    ----------
    compiler:
        Compiler backend: ``"graph"`` (default) or ``"wicer"``.
    """
    source_dir = Path(sources_dir)
    if not source_dir.exists():
        return {"status": "error", "message": f"Sources directory not found: {sources_dir}"}

    try:
        compiler_enum = Compiler(compiler)
    except ValueError:
        return {
            "status": "error",
            "message": f"Unknown compiler: {compiler!r}. Use graph or wicer.",
        }

    try:
        format_enum = OutputFormat(output_format)
    except ValueError:
        return {
            "status": "error",
            "message": f"Unknown format: {output_format}. Use markdown, json, or typst.",
        }

    try:
        image_mode_enum = ImageMode(image_mode)
    except ValueError:
        return {"status": "error", "message": f"Unknown image_mode: {image_mode!r}. Use file or base64."}

    config = PipelineConfig(
        source_dir=source_dir,
        output_dir=Path(output),
        llm=LLMConfig(default_model="", vlm_model=vlm_model),  # type: ignore[unknown-argument]  # ty:ignore[unknown-argument]
        embedding=EmbeddingConfig(model=""),  # type: ignore[unknown-argument]  # ty:ignore[unknown-argument]
        online_sources=urls or [],
        output=OutputConfig(format=format_enum),  # type: ignore[unknown-argument]  # ty:ignore[unknown-argument]
        image_mode=image_mode_enum,
        compiler=compiler_enum,
    )

    pipeline = build_pipeline(config)

    try:
        wiki = pipeline.run()
        ext = "md" if output_format == "markdown" else output_format
        output_file = Path(output) / f"wiki.{ext}"
        return {
            "status": "ok",
            "node_count": len(wiki.nodes),
            "edge_count": len(wiki.edges),
            "toc_count": len(wiki.toc),
            "output_file": str(output_file.resolve()),
            "wiki": {
                "nodes": [dataclasses.asdict(n) for n in wiki.nodes],
                "edges": list(wiki.edges),
                "toc": list(wiki.toc),
                "quality_score": wiki.quality_score,
                "iteration": wiki.iteration,
                "metadata": dict(wiki.metadata),
            },
        }
    except Exception as exc:
        logger.exception("Compilation failed")
        return {"status": "error", "message": str(exc)}

def wiki_capture(
    text: str,
    sources_dir: str = "./sources",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Extract knowledge items from text and save as source files.

    Uses LLM to extract decisions, discoveries, corrections, and references
    from conversation text. Optional tags are attached to each item as a
    baseline (the LLM may add more). Results are written to sources/_captured/.
    """
    source_dir = Path(sources_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = resolve_config()
        model = cfg.get("compile.llm.default_model", "")
        capture_agent = CaptureAgent(model, timeout=120, retries=cfg.get(COMPILE_LLM_OUTPUT_RETRIES)) if model else None
    except Exception:
        capture_agent = None

    try:
        paths = capture(text, sources_dir=source_dir, capture_agent=capture_agent, tags=tags)
        return {
            "status": "ok",
            "count": len(paths),
            "files": [str(p) for p in paths],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

def wiki_learn(
    text: str,
    sources_dir: str = "./sources",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Store a single fact directly as a source file.

    Unlike wiki_capture, this stores the text verbatim without LLM extraction.
    """
    source_dir = Path(sources_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        learned_file = learn(text, sources_dir=source_dir, tags=tags)
        return {
            "status": "ok",
            "file": str(learned_file),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

def wiki_distill(
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Synthesize captured notes into coherent wiki pages."""
    try:
        cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: wiki_dir})
        sources_dir, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

        try:
            model = cfg.get("compile.llm.default_model", "")
            distill_agent = DistillAgent(model, timeout=120, retries=cfg.get(COMPILE_LLM_OUTPUT_RETRIES)) if model else None
        except Exception:
            distill_agent = None

        paths = distill(sources_dir, distill_agent=distill_agent)
        return {
            "status": "ok",
            "count": len(paths),
            "files": [str(p) for p in paths],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

