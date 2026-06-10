"""Fast wiki generation using a subset of sources (for verification).

Uses only the first 3 PDFs to keep runtime under ~30 minutes.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fast_wiki")

LLM_MODEL = "ollama/hf.co/ibm-granite/granite-4.1-3b-GGUF:Q8_0"
EMBED_MODEL = "ollama/granite-embedding:278m"
EMBED_DIM = 768


def main() -> None:
    out_dir = Path("example-wiki")
    for sub in ("vault", "json", "gfm", "typst", ".knox"):
        p = out_dir / sub
        if p.exists():
            shutil.rmtree(p)

    source_files = sorted(Path("tests/data/sources").glob("*.pdf"))[:3]
    logger.info("Using %d source files: %s", len(source_files), [p.name for p in source_files])

    from knox.chunking.hybrid import HybridChunker
    from knox.compilation.graph import GraphCompiler
    from knox.compilation.two_phase import TwoPhaseCompiler
    from knox.core.models import ParsedDocument, ResolvedChunk
    from knox.llm.client import LLMClient
    from knox.output.json_writer import JsonWriter
    from knox.output.markdown import MarkdownWriter
    from knox.output.typst import TypstWriter
    from knox.output.wiki_index import write_index_and_log, write_overview
    from knox.preprocessing.parsers.docling_parser import DoclingParser

    parser = DoclingParser(image_mode="file")
    parsed_docs: list[ParsedDocument] = []
    for src in source_files:
        doc = parser.parse(src)
        imgs = sum(1 for e in doc.elements if e.type.name == "IMAGE")
        logger.info("Parsed %s: %d elements (%d images)", src.name, len(doc.elements), imgs)
        parsed_docs.append(doc)

    chunker = HybridChunker(max_tokens=512)
    all_chunks = []
    for doc in parsed_docs:
        all_chunks.extend(chunker.chunk(doc))
    logger.info("Total chunks: %d", len(all_chunks))

    resolved = [ResolvedChunk(chunk=c) for c in all_chunks]

    compiler = TwoPhaseCompiler(base=GraphCompiler(), prompt_budget=150_000, model=LLM_MODEL, timeout=120)
    logger.info("Compiling with TwoPhaseCompiler...")
    wiki = compiler.compile(resolved)
    logger.info(
        "Compiled: %d nodes, %d edges, quality=%.2f",
        len(wiki.nodes), len(wiki.edges), wiki.quality_score,
    )

    md_vault = out_dir / "vault"
    MarkdownWriter(flavor="obsidian", vault_mode=True).write(wiki, md_vault)
    md_files = list(md_vault.rglob("*.md"))
    logger.info("Markdown vault: %d .md files", len(md_files))

    write_index_and_log(wiki, md_vault)
    overview_path = write_overview(wiki, md_vault, model=LLM_MODEL)
    if overview_path:
        logger.info("overview.md generated")

    JsonWriter().write(wiki, out_dir / "json")
    MarkdownWriter(flavor="gfm", vault_mode=False).write(wiki, out_dir / "gfm")
    TypstWriter().write(wiki, out_dir / "typst")

    logger.info("=" * 60)
    logger.info("Fast wiki complete at %s", out_dir.resolve())
    logger.info("  Nodes: %d | Edges: %d | Images extracted: check vault/images/", len(wiki.nodes), len(wiki.edges))


if __name__ == "__main__":
    main()
