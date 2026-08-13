"""Knowledge compilation MCP tools."""

import dataclasses
from pathlib import Path
from typing import Any

from binaryornot.check import is_binary

from knox.agents.capture import CaptureAgent
from knox.agents.distill import DistillAgent
from knox.capture.capture import capture
from knox.capture.distill import distill
from knox.capture.learn import learn
from knox.core.config import (
    DEFAULTS,
    EmbeddingConfig,
    LLMConfig,
    resolve_config,
)
from knox.core.config.keys import COMPILE_LLM_OUTPUT_RETRIES, DISPLAY_MCP_ELEMENT_CHARS
from knox.core.models import MetadataField, WikiField
from knox.core.models.attachments import AttachmentManifest
from knox.core.paths import vault_paths
from knox.core.pipeline.config import Compiler, ImageMode, OutputConfig, OutputFormat, PipelineConfig
from knox.core.pipeline.orchestrator import build_pipeline
from knox.logging import get_logger
from knox.preprocessing.parsers import ParserBuildOptions, ParserRouter, get_backend

logger = get_logger(__name__)


def parse_document(file_path: str) -> dict[str, Any]:
    """Parse a document and return structured elements."""
    doc_file = Path(file_path)
    if not doc_file.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    router = ParserRouter()
    try:
        docling_factory = get_backend("docling")
        router.register(docling_factory(ParserBuildOptions()))
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
        llm=LLMConfig(default_model="", vlm_model=vlm_model),  # type: ignore[unknown-argument]
        embedding=EmbeddingConfig(model=""),  # type: ignore[unknown-argument]
        online_sources=urls or [],
        output=OutputConfig(format=format_enum),  # type: ignore[unknown-argument]
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
            MetadataField.NODE_COUNT: len(wiki.nodes),
            MetadataField.EDGE_COUNT: len(wiki.edges),
            "toc_count": len(wiki.toc),
            "output_file": str(output_file.resolve()),
            "wiki": {
                WikiField.NODES: [dataclasses.asdict(n) for n in wiki.nodes],
                WikiField.EDGES: list(wiki.edges),
                WikiField.TOC: list(wiki.toc),
                WikiField.QUALITY_SCORE: wiki.quality_score,
                WikiField.ITERATION: wiki.iteration,
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
    wiki_dir: str | None = None,
) -> dict[str, Any]:
    """Synthesize captured notes into coherent wiki pages."""
    try:
        vp = vault_paths(wiki_dir)
        cfg = resolve_config(base_dir=vp.base_dir)
        sources_dir = vp.sources_dir

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


def wiki_list_attachments(
    source: str | None = None,
    *,
    wiki_dir: str | None = None,
) -> dict[str, Any]:
    """List registered attachments for a source (or all sources).

    Args:
        source: Optional source stem to filter by (e.g. ``"report"``).
        wiki_dir: Wiki directory override.

    Returns:
        Dict with ``attachments`` list (each: source, path, ingestible, embed).
    """
    vp = vault_paths(wiki_dir)
    manifest = AttachmentManifest(vp.attachments_manifest_path)
    if not manifest.exists():
        return {"status": "ok", "attachments": [], "count": 0}

    entries = manifest.read_all().values()
    if source:
        entries = [a for a in entries if a.source == source]

    return {
        "status": "ok",
        "count": len(entries),
        "attachments": [
            {
                "source": a.source,
                "path": a.path,
                "filename": a.filename,
                "ingestible": a.ingestible,
                "embed": a.embed,
                "exists": (vp.attachments_dir / a.path).exists(),
            }
            for a in entries
        ],
    }


def wiki_read_attachment(
    source: str,
    attachment_name: str,
    *,
    wiki_dir: str | None = None,
    max_bytes: int = 100_000,
) -> dict[str, Any]:
    """Read the text content of an attachment.

    Binary files return metadata only (no content). Text files return
    up to ``max_bytes`` of content.

    Args:
        source: Source stem the attachment belongs to.
        attachment_name: Attachment filename within the source's attachment directory.
        wiki_dir: Wiki directory override.
        max_bytes: Maximum bytes to read for text files.

    Returns:
        Dict with ``content`` (text) or ``binary: true``.
    """
    vp = vault_paths(wiki_dir)
    att_path = f"{source}/{attachment_name}"
    att_file = vp.attachments_dir / att_path

    if not att_file.exists():
        return {"status": "error", "message": f"Attachment not found: {att_path}"}

    info = {
        "status": "ok",
        "source": source,
        "path": att_path,
        "size": att_file.stat().st_size,
    }

    if is_binary(str(att_file)):
        info["binary"] = True
        return info

    content = att_file.read_text(errors="replace")[:max_bytes]
    info["content"] = content
    return info


def wiki_attachment_info(
    source: str,
    attachment_name: str,
    *,
    wiki_dir: str | None = None,
) -> dict[str, Any]:
    """Return metadata for a single attachment.

    Args:
        source: Source stem the attachment belongs to.
        attachment_name: Attachment filename within the source's attachment directory.
        wiki_dir: Wiki directory override.

    Returns:
        Dict with manifest entry and file existence/size.
    """
    vp = vault_paths(wiki_dir)
    manifest = AttachmentManifest(vp.attachments_manifest_path)
    att_path = f"{source}/{attachment_name}"

    entry = None
    if manifest.exists():
        entry = manifest.read_all().get(att_path)

    att_file = vp.attachments_dir / att_path

    return {
        "status": "ok",
        "source": source,
        "path": att_path,
        "registered": entry is not None,
        "ingestible": entry.ingestible if entry else None,
        "embed": entry.embed if entry else None,
        "exists": att_file.exists(),
        "size": att_file.stat().st_size if att_file.exists() else 0,
    }
