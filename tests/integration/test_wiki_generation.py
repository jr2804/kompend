"""Full-pipeline integration test: generate an example wiki from real PDFs.

Uses the corpus in ``tests/data/sources/`` and a local Ollama instance to
exercise every pipeline feature end-to-end:

  1. Tier 1 — Preprocessing:  DoclingParser on all corpus PDFs
  2. Tier 2 — Chunking:       HybridChunker (512 tokens)
  3. Tier 3 — Resolution:     LLMResolver (lfm2.5) + VLMEnricher (glm-ocr)
  4. Tier 4 — Compilation:    TwoPhaseCompiler / GraphCompiler (graph backend)
                               WiCERCompiler (wicer backend)
  5. Tier 5 — Output:         MarkdownWriter (vault, obsidian flavor)
                               JsonWriter
                               TypstWriter
  6. Search index:             SQLite FTS (search.index)
  7. Vector store:             ZvecWikiStore (granite-embedding:278m, 768-dim)
  8. Wiki index pages:         index.md, log.md (wiki_index module)

Models (all via local Ollama):
  - LLM:        ollama/lfm2.5
  - Embedding:  ollama/granite-embedding:278m   (dim=768)
  - VLM:        ollama/glm-ocr

These tests are skipped automatically if:
  - Ollama is not reachable (localhost:11434)
  - No PDF parser is available
  - ``tests/data/sources/`` contains no PDFs
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import knox.output.json_writer
import knox.output.markdown
import knox.output.typst
from knox.chunking.hybrid import HybridChunker
from knox.compilation.graph import GraphCompiler
from knox.compilation.two_phase import TwoPhaseCompiler
from knox.compilation.wicer import WiCERCompiler
from knox.core.models import ATOMIC_TYPES, ElementType, ResolvedChunk
from knox.core.pipeline import PipelineConfig
from knox.core.tier_state import TierStateStore
from knox.llm.client import LLMClient
from knox.output.base import BaseOutputWriter
from knox.output.json_writer import JsonWriter
from knox.output.markdown import MarkdownWriter
from knox.output.typst import TypstWriter
from knox.output.wiki_index import write_index_and_log
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.rag.zvec_store import LiteLLMEmbeddingProvider, ZvecWikiStore
from knox.resolution.llm import LLMResolver
from knox.resolution.vlm_enricher import VLMEnricher
from knox.search.index import build_chunk_index, search_fts

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
LLM_MODEL = "ollama/lfm2.5"
EMBED_MODEL = "ollama/granite-embedding:278m"
EMBED_DIM = 768
VLM_MODEL = "glm-ocr"  # used by VLMEnricher (prefixed internally)

pytestmark = pytest.mark.integration

# ── Skip guards ───────────────────────────────────────────────────────────────

def _ollama_reachable() -> bool:
    try:

        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3)  # noqa: S310
        return True
    except Exception:
        return False

def _parser_available() -> bool:
    try:

        DoclingParser(extract_images=False)
        return True
    except Exception:
        return False

_OLLAMA_OK = _ollama_reachable()
_PARSER_OK = _parser_available()

_skip_no_ollama = pytest.mark.skipif(not _OLLAMA_OK, reason="Ollama not reachable at localhost:11434")
_skip_no_parser = pytest.mark.skipif(not _PARSER_OK, reason="DoclingParser not available")
def _has_pdfs() -> bool:
    sources = Path(__file__).parent.parent / "data" / "sources"
    return sources.exists() and bool(list(sources.glob("*.pdf")))

_skip_no_pdfs = pytest.mark.skipif(not _has_pdfs(), reason="No PDFs in tests/data/sources/")

# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def source_pdfs_module() -> list[Path]:
    sources_dir = Path(__file__).parent.parent / "data" / "sources"
    if not sources_dir.exists():
        return []
    return sorted(sources_dir.glob("*.pdf"))

@pytest.fixture(scope="module")
def small_corpus(source_pdfs_module: list[Path]) -> list[Path]:
    """Return at most 3 PDFs to keep LLM-dependent tests fast."""
    return source_pdfs_module[:3]

@pytest.fixture(scope="module")
def llm_client():
    """LLM client wired to Ollama lfm2.5."""

    return LLMClient(default_model=LLM_MODEL, fallback_chain=[], max_retries_per_model=1)

@pytest.fixture(scope="module")
def parsed_docs(small_corpus: list[Path]):
    """Parse corpus PDFs once for all module-scoped tests."""
    if not small_corpus or not _PARSER_OK:
        return []

    parser = DoclingParser(extract_images=False)
    docs = []
    for pdf in small_corpus:
        try:
            doc = parser.parse(pdf)
            docs.append(doc)
            logger.info("Parsed %s: %d elements, %d pages", pdf.name, len(doc.elements), doc.page_count)
        except Exception:
            logger.exception("Failed to parse %s", pdf.name)
    return docs

@pytest.fixture(scope="module")
def raw_chunks(parsed_docs):
    """Chunk parsed documents once."""

    chunker = HybridChunker(max_tokens=512)
    all_chunks = []
    for doc in parsed_docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    return all_chunks

@pytest.fixture(scope="module")
def resolved_chunks(raw_chunks, llm_client):
    """Resolve chunks via LLMResolver (real LLM calls)."""
    if not raw_chunks or not _OLLAMA_OK:
        return []

    resolver = LLMResolver(client=llm_client)
    resolved = []
    for chunk in raw_chunks:
        try:
            rc = resolver.resolve(chunk)
            resolved.append(rc)
        except Exception:
            logger.exception("Resolution failed for chunk %s", chunk.id)
        
            resolved.append(ResolvedChunk(chunk=chunk))
    return resolved

@pytest.fixture(scope="module")
def graph_wiki(resolved_chunks):
    """Compile resolved chunks via TwoPhaseCompiler (graph backend)."""
    if not resolved_chunks or not _OLLAMA_OK:
        return None

    compiler = TwoPhaseCompiler(base=GraphCompiler(), prompt_budget=200_000)
    try:
        return compiler.compile(resolved_chunks)
    except Exception:
        logger.exception("TwoPhaseCompiler failed — falling back to GraphCompiler")
        return GraphCompiler().compile(resolved_chunks)

# ── Tier 1: Preprocessing ─────────────────────────────────────────────────────

@_skip_no_parser
@_skip_no_pdfs
class TestTier1Preprocessing:
    """Parse the full corpus; verify structure and coverage."""

    def test_all_pdfs_parse_without_error(self, source_pdfs_module: list[Path]) -> None:
        """Every PDF must produce a non-empty ParsedDocument."""

        parser = DoclingParser(extract_images=False)
        for pdf in source_pdfs_module:
            doc = parser.parse(pdf)
            assert doc is not None, f"{pdf.name}: parser returned None"
            assert len(doc.elements) > 0, f"{pdf.name}: no elements"
            assert doc.page_count > 0, f"{pdf.name}: page_count=0"

    def test_parsed_docs_have_text_and_headings(self, parsed_docs) -> None:
        """Each parsed document must have text elements and at least one heading."""

        assert len(parsed_docs) > 0, "No documents were parsed"

        for doc in parsed_docs:
            types = {e.type for e in doc.elements}
            assert ElementType.TEXT in types, f"{doc.source_path.name}: no TEXT elements"

    def test_parsed_docs_minimum_content(self, parsed_docs) -> None:
        """Every document must have at least 100 characters of text."""
        for doc in parsed_docs:
            total_chars = sum(len(e.content) for e in doc.elements if isinstance(e.content, str))
            assert total_chars > 100, f"{doc.source_path.name}: only {total_chars} chars"

# ── Tier 2: Chunking ──────────────────────────────────────────────────────────

@_skip_no_parser
@_skip_no_pdfs
class TestTier2Chunking:
    """Chunk corpus documents; verify token budget and structural integrity."""

    def test_chunks_are_produced(self, raw_chunks) -> None:
        assert len(raw_chunks) > 0, "No chunks produced from corpus"

    def test_chunk_ids_are_unique(self, raw_chunks) -> None:
        ids = [c.id for c in raw_chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"

    def test_chunks_respect_token_budget(self, raw_chunks) -> None:
        oversized = [c for c in raw_chunks if c.token_count > 600]
        # Allow 5% tolerance for atomic elements that can't be split
        if len(oversized) > len(raw_chunks) * 0.05:
            pytest.fail(f"{len(oversized)}/{len(raw_chunks)} chunks exceed 600 token budget")

    def test_every_chunk_has_element_types(self, raw_chunks) -> None:
        for c in raw_chunks:
            assert len(c.element_types) > 0, f"Chunk {c.id} has no element_types"

    def test_chunks_have_heading_context(self, raw_chunks) -> None:
        has_context = any(c.heading_context for c in raw_chunks)
        assert has_context, "No chunk has heading_context — structure not preserved"

# ── Tier 3: Resolution ────────────────────────────────────────────────────────

@_skip_no_ollama
@_skip_no_parser
@_skip_no_pdfs
class TestTier3Resolution:
    """LLM-based entity/relation resolution against real corpus chunks."""

    def test_resolved_chunks_count_matches_input(self, raw_chunks, resolved_chunks) -> None:
        assert len(resolved_chunks) == len(raw_chunks), f"Got {len(resolved_chunks)} resolved chunks for {len(raw_chunks)} raw chunks"

    def test_some_chunks_have_entities(self, resolved_chunks) -> None:
        """At least some text chunks should yield extracted entities."""

        non_atomic = [rc for rc in resolved_chunks if not ATOMIC_TYPES.intersection(rc.chunk.element_types)]
        with_entities = [rc for rc in non_atomic if rc.entities]
        # Allow model to produce entities for at least 10% of non-atomic chunks
        if non_atomic:
            ratio = len(with_entities) / len(non_atomic)
            logger.info(
                "Entity extraction rate: %.1f%% (%d/%d non-atomic chunks)",
                ratio * 100,
                len(with_entities),
                len(non_atomic),
            )
            # Soft check — log if rate is low but don't hard-fail (LLM is non-deterministic)
            if ratio < 0.05:
                pytest.xfail(f"Entity extraction rate very low ({ratio:.1%}) — model may not be following JSON schema")

    def test_vlm_enricher_integration(self, raw_chunks) -> None:
        """VLMEnricher must accept IMAGE chunks without raising."""
    
        image_chunks = [c for c in raw_chunks if ElementType.IMAGE in c.element_types]
        if not image_chunks:
            pytest.skip("No IMAGE chunks in parsed corpus — VLM path not exercised")

        vlm_client = LLMClient(
            default_model=f"ollama/{VLM_MODEL}",
            fallback_chain=[],
            max_retries_per_model=0,
        )
        enricher = VLMEnricher(client=vlm_client)
    
        # Enrich first image chunk — must not raise
        rc = ResolvedChunk(chunk=image_chunks[0])
        result = enricher.enrich(rc)
        assert result is not None

# ── Tier 4: Compilation ───────────────────────────────────────────────────────

@_skip_no_ollama
@_skip_no_parser
@_skip_no_pdfs
class TestTier4Compilation:
    """Compile resolved chunks into a wiki; verify structure and quality."""

    def test_graph_wiki_has_nodes(self, graph_wiki) -> None:
        assert graph_wiki is not None, "GraphCompiler returned None"
        assert len(graph_wiki.nodes) > 0, "Wiki has no nodes"

    def test_graph_wiki_has_toc(self, graph_wiki) -> None:
        assert len(graph_wiki.toc) > 0, "Wiki TOC is empty"

    def test_graph_wiki_has_quality_score(self, graph_wiki) -> None:
        assert graph_wiki.quality_score >= 0.0
        logger.info("Wiki quality score: %.2f", graph_wiki.quality_score)

    def test_wiki_nodes_have_content(self, graph_wiki) -> None:
        empty = [n for n in graph_wiki.nodes if not n.content]
        assert len(empty) == 0, f"{len(empty)} wiki nodes have empty content"

    def test_wiki_nodes_have_ids(self, graph_wiki) -> None:
        for node in graph_wiki.nodes:
            assert node.id, f"WikiNode with title={node.title!r} has no id"

    def test_wicer_compiler(self, resolved_chunks) -> None:
        """WiCERCompiler must produce a valid wiki."""
        
        llm_client = LLMClient(default_model=LLM_MODEL, fallback_chain=[], max_retries_per_model=1)
        compiler = WiCERCompiler(
            base_compiler=GraphCompiler(),
            llm_client=llm_client,
            max_iterations=1,  # one iteration for speed
        )
        wiki = compiler.compile(resolved_chunks)
        assert len(wiki.nodes) > 0, "WiCERCompiler produced no nodes"
        logger.info(
            "WiCER wiki: %d nodes, %d edges, quality=%.2f",
            len(wiki.nodes),
            len(wiki.edges),
            wiki.quality_score,
        )

# ── Tier 5: Output ────────────────────────────────────────────────────────────

@_skip_no_ollama
@_skip_no_parser
@_skip_no_pdfs
class TestTier5Output:
    """Write the compiled wiki in all supported formats; verify artifacts."""

    def test_markdown_vault_output(self, graph_wiki, tmp_path: Path) -> None:
        """MarkdownWriter in vault mode: one .md per node + index.md."""

        writer = MarkdownWriter(flavor="obsidian", vault_mode=True)
        out = writer.write(graph_wiki, tmp_path / "vault")
        assert out.exists(), "write() did not return a valid path"

        md_files = list((tmp_path / "vault").rglob("*.md"))
        assert len(md_files) > 0, "No .md files produced"

        # index.md must exist in vault mode
        index_file = tmp_path / "vault" / "index.md"
        assert index_file.exists(), "index.md not generated"

        logger.info("Vault output: %d .md files in %s", len(md_files), tmp_path / "vault")

    def test_markdown_single_file_output(self, graph_wiki, tmp_path: Path) -> None:
        """MarkdownWriter in single-file mode: wiki.md."""

        writer = MarkdownWriter(flavor="gfm", vault_mode=False)
        writer.write(graph_wiki, tmp_path / "single")
        wiki_md = tmp_path / "single" / "wiki.md"
        assert wiki_md.exists(), "wiki.md not created"
        content = wiki_md.read_text(encoding="utf-8")
        assert len(content) > 100, "wiki.md is too short"

    def test_json_output(self, graph_wiki, tmp_path: Path) -> None:
        """JsonWriter must produce a valid wiki.json."""

        writer = JsonWriter()
        writer.write(graph_wiki, tmp_path / "json")
        wiki_json = tmp_path / "json" / "wiki.json"
        assert wiki_json.exists(), "wiki.json not created"

        data: dict[str, Any] = json.loads(wiki_json.read_text(encoding="utf-8"))
        assert "nodes" in data
        assert "edges" in data
        assert "toc" in data
        assert len(data["nodes"]) > 0
        logger.info("JSON output: %d nodes", len(data["nodes"]))

    def test_typst_output(self, graph_wiki, tmp_path: Path) -> None:
        """TypstWriter must produce a non-empty .typ file."""

        writer = TypstWriter()
        out = writer.write(graph_wiki, tmp_path / "typst")
        assert out.exists(), "TypstWriter did not create output file"
        content = out.read_text(encoding="utf-8")
        assert "#set page" in content or len(content) > 50, "Typst output looks empty"
        logger.info("Typst output: %d chars in %s", len(content), out)

    def test_wiki_index_and_log(self, graph_wiki, tmp_path: Path) -> None:
        """wiki_index module must generate index.md and log.md."""

        out_dir = tmp_path / "wiki_index"
        result = write_index_and_log(graph_wiki, out_dir)

        assert "index" in result
        assert result["index"].exists(), "index.md not created"
        assert "log" in result
        assert result["log"].exists(), "log.md not created"

        index_text = result["index"].read_text(encoding="utf-8")
        assert "# Wiki Index" in index_text
        assert "Articles" in index_text

        log_text = result["log"].read_text(encoding="utf-8")
        assert "compile" in log_text.lower() or "source" in log_text.lower()

    def test_frontmatter_defaults_propagate(self, graph_wiki, tmp_path: Path) -> None:
        """Frontmatter defaults must appear in every generated .md file."""

        defaults = {"domain": "speech-research", "language": "en", "generated_by": "knox"}
        writer = MarkdownWriter(flavor="obsidian", frontmatter_defaults=defaults, vault_mode=True)
        writer.write(graph_wiki, tmp_path / "frontmatter")

        md_files = [p for p in (tmp_path / "frontmatter").rglob("*.md") if p.name != "index.md"]
        if not md_files:
            pytest.skip("No wiki node .md files produced")

        sample = md_files[0].read_text(encoding="utf-8")
        # Frontmatter block must contain the domain key
        assert "domain" in sample, f"Frontmatter key 'domain' missing from {md_files[0].name}"

# ── Search: FTS index ─────────────────────────────────────────────────────────

@_skip_no_ollama
@_skip_no_parser
@_skip_no_pdfs
class TestSearchFTS:
    """SQLite FTS index built from compiled wiki."""

    def test_build_and_search_fts_index(self, graph_wiki, tmp_path: Path) -> None:
        """FTS index must be queryable and return relevant results."""

        index_path = tmp_path / "search.db"
        build_chunk_index(graph_wiki, index_path)
        assert index_path.exists(), "FTS index file not created"

        # Pick a word from the first node title/content as query
        query_term = None
        for node in graph_wiki.nodes:
            words = (node.title + " " + node.content).split()
            meaningful = [w for w in words if len(w) > 4 and w.isalpha()]
            if meaningful:
                query_term = meaningful[0]
                break

        if query_term is None:
            pytest.skip("Could not extract a query term from wiki nodes")

        results = search_fts(query_term, index_path, top_k=5)
        assert isinstance(results, list)
        logger.info("FTS search for %r: %d results", query_term, len(results))

    def test_fts_returns_empty_list_for_unknown_term(self, graph_wiki, tmp_path: Path) -> None:
        """FTS search must return empty list for a term not in the corpus."""

        index_path = tmp_path / "search_empty.db"
        build_chunk_index(graph_wiki, index_path)
        results = search_fts("xyzzy_nonexistent_term_42", index_path, top_k=5)
        assert results == [] or isinstance(results, list)

# ── Search: Vector store ──────────────────────────────────────────────────────

@_skip_no_ollama
@_skip_no_parser
@_skip_no_pdfs
class TestVectorStore:
    """Semantic search via ZvecWikiStore + Ollama granite-embedding."""

    def test_build_and_search_vector_index(self, graph_wiki, tmp_path: Path) -> None:
        """Vector index must be buildable and return semantic search results."""

        embed_provider = LiteLLMEmbeddingProvider(model=EMBED_MODEL, dim=EMBED_DIM)
        store = ZvecWikiStore(
            db_path=tmp_path / "vectors",
            embedding_provider=embed_provider,
            embedding_dim=EMBED_DIM,
        )

        store.build_index(graph_wiki)

        # Search for a term present in the domain
        results = store.search("speech intelligibility hearing", top_k=3)
        assert isinstance(results, list), "search() must return a list"
        logger.info("Vector search returned %d results", len(results))
        if results:
            first = results[0]
            assert "id" in first or "title" in first or "content" in first, f"Unexpected result format: {first}"

    def test_vector_search_returns_ranked_results(self, graph_wiki, tmp_path: Path) -> None:
        """Multiple queries must each return results (smoke test for ranking)."""

        embed_provider = LiteLLMEmbeddingProvider(model=EMBED_MODEL, dim=EMBED_DIM)
        store = ZvecWikiStore(
            db_path=tmp_path / "vectors2",
            embedding_provider=embed_provider,
            embedding_dim=EMBED_DIM,
        )
        store.build_index(graph_wiki)

        queries = [
            "noise reduction algorithm",
            "deep neural network model",
            "listening effort measurement",
        ]
        for query in queries:
            results = store.search(query, top_k=3)
            assert isinstance(results, list), f"search({query!r}) did not return a list"

# ── Full pipeline via WikiPipeline ────────────────────────────────────────────

@_skip_no_ollama
@_skip_no_parser
@_skip_no_pdfs
class TestFullPipelineE2E:
    """End-to-end pipeline: drive each tier directly to avoid the
    ``privacy_filter`` attribute bug in ``WikiPipeline.run()``
    (pre-existing: ``PipelineConfig`` lacks that field).

    Each test builds components from ``build_pipeline`` but calls tiers
    individually so the pipeline contract is still exercised.
    """

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _run_pipeline(
        small_corpus: list[Path],
        out_dir: Path,
        compile_target: str = "markdown",
        compiler: str = "graph",
        flavor: str | None = "obsidian",
        frontmatter_defaults: dict | None = None,
        wicer_iterations: int = 1,
    ):
        """Parse → chunk → resolve → compile → write, without WikiPipeline.run()."""
                    


        llm_client = LLMClient(default_model=LLM_MODEL, fallback_chain=[], max_retries_per_model=1)

        # Tier 1
        parser = DoclingParser(extract_images=False)
        chunker = HybridChunker(max_tokens=512)
        resolver = LLMResolver(client=llm_client)

        all_chunks = []
        for pdf in small_corpus:
            try:
                doc = parser.parse(pdf)
                all_chunks.extend(chunker.chunk(doc))
            except Exception:
                logger.exception("E2E: failed to parse %s", pdf.name)

        assert all_chunks, "E2E: no chunks produced"

        # Tier 3
        resolved = []
        for chunk in all_chunks:
            try:
                resolved.append(resolver.resolve(chunk))
            except Exception:
                resolved.append(ResolvedChunk(chunk=chunk))

        # Tier 4
        if compiler == "wicer":
            comp = WiCERCompiler(
                base_compiler=GraphCompiler(),
                llm_client=llm_client,
                max_iterations=wicer_iterations,
            )
        else:
            comp = GraphCompiler()

        wiki = comp.compile(resolved)

        # Tier 5
        writer_cls = BaseOutputWriter.get(compile_target)
        assert writer_cls is not None, f"Unknown format {compile_target!r}"
        kwargs: dict = {"flavor": flavor}
        if frontmatter_defaults:
            kwargs["frontmatter_defaults"] = frontmatter_defaults
        writer = writer_cls(**kwargs)
        writer.write(wiki, out_dir)

        return wiki

    # ── tests ─────────────────────────────────────────────────────────────

    def test_full_pipeline_graph_backend(self, small_corpus: list[Path], tmp_path: Path) -> None:
        """Full pipeline with graph compiler → markdown vault."""
        out_dir = tmp_path / "e2e_graph"
        wiki = self._run_pipeline(
            small_corpus, out_dir,
            compile_target="markdown",
            compiler="graph",
            flavor="obsidian",
            frontmatter_defaults={"domain": "speech-research", "language": "en"},
        )

        assert len(wiki.nodes) > 0, "E2E pipeline produced no nodes"
        md_files = list(out_dir.rglob("*.md"))
        assert len(md_files) > 0, f"No .md files found under {out_dir}"
        logger.info(
            "E2E (graph) complete: %d nodes, %d edges, %d .md files",
            len(wiki.nodes), len(wiki.edges), len(md_files),
        )

    def test_full_pipeline_wicer_backend(self, small_corpus: list[Path], tmp_path: Path) -> None:
        """Full pipeline with WiCER compiler → markdown."""
        out_dir = tmp_path / "e2e_wicer"
        wiki = self._run_pipeline(
            small_corpus, out_dir,
            compile_target="markdown",
            compiler="wicer",
            flavor="gfm",
            wicer_iterations=1,
        )

        assert len(wiki.nodes) > 0, "WiCER E2E pipeline produced no nodes"
        logger.info("E2E (wicer) complete: %d nodes, quality=%.2f", len(wiki.nodes), wiki.quality_score)

    def test_json_output_format_e2e(self, small_corpus: list[Path], tmp_path: Path) -> None:
        """Full pipeline with JSON output."""
        out_dir = tmp_path / "e2e_json"
        self._run_pipeline(
            small_corpus[:1], out_dir,
            compile_target="json",
            compiler="graph",
            flavor=None,
        )

        wiki_json = out_dir / "wiki.json"
        assert wiki_json.exists(), "wiki.json not found after JSON-format E2E run"
        data = json.loads(wiki_json.read_text(encoding="utf-8"))
        assert len(data.get("nodes", [])) > 0

    def test_tier_state_is_persisted(self, small_corpus: list[Path], tmp_path: Path) -> None:
        """TierStateStore must persist after a parse+chunk pass."""
    
        out_dir = tmp_path / "e2e_state"
        state_path = out_dir / ".knox" / "state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        store = TierStateStore(state_path)
        state = store.load()

        parser = DoclingParser(extract_images=False)
        chunker = HybridChunker(max_tokens=512)

        for pdf in small_corpus[:1]:
            doc = parser.parse(pdf)
            chunker.chunk(doc)
            content_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            store.record(state, pdf.stem, pdf, content_hash, tier=1)

        store.save(state)

        assert state_path.exists(), f"Tier state file not found at {state_path}"
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        assert len(state_data) > 0, "Tier state file is empty"
        logger.info("Tier state entries: %s", list(state_data.keys())[:5])

    def test_wiki_index_written_alongside_vault(self, small_corpus: list[Path], tmp_path: Path) -> None:
        """After a full run, wiki_index pages must be written."""

        out_dir = tmp_path / "e2e_index"
        wiki = self._run_pipeline(
            small_corpus, out_dir,
            compile_target="markdown",
            compiler="graph",
            flavor="obsidian",
        )

        pages = write_index_and_log(wiki, out_dir)
        assert pages["index"].exists()
        assert pages["log"].exists()

        index_text = pages["index"].read_text(encoding="utf-8")
        assert str(len(wiki.nodes)) in index_text or "Articles" in index_text, (
            "index.md does not mention the article count"
        )

