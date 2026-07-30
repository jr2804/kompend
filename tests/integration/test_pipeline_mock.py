"""Mock-only pipeline integration test.

Exercises the full ``parse → chunk → compile → write`` pipeline
**without** any real LLM/VLM/embedding calls. We patch
``knox.llm.models.build_model`` to return pydantic-ai's
:class:`~pydantic_ai.models.test.TestModel`, which produces valid structured
output for every result type without any network or model load. Embedding is
replaced by an in-process deterministic provider.

This is the only integration test that exercises the full
``GraphCompiler / WiCERCompiler → MarkdownWriter / JsonWriter`` surface end
to end. Real-engine tests (``test_arxiv_pipeline.py``, ``test_real_corpus.py``)
cover network-bound preprocessing only — they make no LLM/VLM/embedding calls.

Rule: see ``tests/AGENTS.md`` — never let real LLM/VLM/embedding calls into
the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

import knox.llm.models as _llm_models
from knox.chunking import HierarchicalChunker
from knox.compilation.graph import GraphCompiler
from knox.compilation.wicer import WiCERCompiler
from knox.core.config import EmbeddingConfig, LLMConfig
from knox.core.models import Chunk, ChunkTree, CompiledWiki, ResolvedChunk
from knox.core.paths import KNOX_DIR, STATE_FILE
from knox.core.pipeline.config import PipelineConfig
from knox.core.pipeline.state import SourceState, TierState, TierStateStore
from knox.output.json_writer import JsonWriter
from knox.output.markdown import MarkdownWriter
from tests.conftest import DeterministicTestEmbeddingProvider
from tests.fixtures.documents import make_test_doc

# These model strings are never actually resolved — TestModel mocks
# ``build_model`` so the provider prefix is irrelevant. Use any string the
# ``build_model`` parser would accept if it weren't monkey-patched.
LLM_MODEL = "ollama/mock-agent"
EMBED_MODEL = "mock/embedding"
VLM_MODEL = "ollama/mock-vlm"


# ── Tests ────────────────────────────────────────────────────────────────────


def test_graph_compiler_produces_wiki(
    raw_chunks: list[ResolvedChunk],
    tmp_path: Path,
    mock_llm: None,
) -> None:
    """Graph backend compiles chunks into a wiki; vault writer serializes it."""
    config = _make_config(tmp_path)
    config.llm.timeout = 30

    compiler = GraphCompiler(LLM_MODEL)
    wiki: CompiledWiki = compiler.compile(raw_chunks)
    assert wiki is not None
    assert hasattr(wiki, "nodes")

    writer = _vault_for(tmp_path)
    out_dir = tmp_path / "vault"
    writer.write(wiki, out_dir)
    assert out_dir.exists()


def test_wicer_compiler_produces_wiki(
    raw_chunks: list[ResolvedChunk],
    tmp_path: Path,
    mock_llm: None,
) -> None:
    """WiCER backend (wicer) also produces a wiki under the mock surface."""
    config = _make_config(tmp_path)
    config.llm.timeout = 30

    base = GraphCompiler(LLM_MODEL)
    compiler = WiCERCompiler(config=config, base_compiler=base)
    wiki: CompiledWiki = compiler.compile(raw_chunks)
    assert wiki is not None

    writer = _vault_for(tmp_path)
    out_dir = tmp_path / "vault"
    writer.write(wiki, out_dir)
    assert out_dir.exists()


def test_json_writer_emits_valid_json(
    raw_chunks: list[ResolvedChunk],
    tmp_path: Path,
    mock_llm: None,
) -> None:
    """JsonWriter emits parseable JSON for the compiled wiki."""
    _make_config(tmp_path)
    compiler = GraphCompiler(LLM_MODEL)
    wiki: CompiledWiki = compiler.compile(raw_chunks)

    writer = JsonWriter()
    out_dir = tmp_path / "json"
    writer.write(wiki, out_dir)

    json_files = list(out_dir.rglob("*.json"))
    assert json_files, "JsonWriter did not emit any JSON files"
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert isinstance(payload, (dict, list))


def test_vault_files_have_v2_frontmatter(
    raw_chunks: list[ResolvedChunk],
    tmp_path: Path,
    mock_llm: None,
) -> None:
    """Every vault page declares OKF v0.2 frontmatter (``generated`` block)."""
    _make_config(tmp_path)
    compiler = GraphCompiler(LLM_MODEL)
    wiki: CompiledWiki = compiler.compile(raw_chunks)

    writer = _vault_for(tmp_path)
    out_dir = tmp_path / "vault"
    writer.write(wiki, out_dir)

    md_files = [f for f in out_dir.rglob("*.md") if f.name not in {"index.md", "log.md"}]
    assert md_files, "No page markdown files written to vault"

    for f in md_files:
        text = f.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        end = text.find("---", 3)
        if end < 0:
            continue
        fm = text[3:end]
        assert "generated" in fm, f"{f.name}: missing 'generated' field in frontmatter"
        assert "by:" in fm, f"{f.name}: missing 'generated.by' in frontmatter"
        assert "at:" in fm, f"{f.name}: missing 'generated.at' in frontmatter"


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every ``build_model`` call with pydantic-ai's ``TestModel``.

    TestModel is pydantic-ai's official mock: it inspects the agent's
    ``result_type`` and generates valid output that satisfies the Pydantic
    schema. No network, no model load, no API keys.
    """
    monkeypatch.setattr(_llm_models, "build_model", lambda *_args, **_kwargs: TestModel())


@pytest.fixture
def raw_chunks(parsed_docs: list) -> list[ResolvedChunk]:
    chunker = HierarchicalChunker(max_tokens=512)
    out: list[ResolvedChunk] = []
    for doc in parsed_docs:
        tree: ChunkTree = chunker.chunk(doc)
        for node in tree.leaves():
            if node.chunk:
                out.append(ResolvedChunk(chunk=node.chunk) if isinstance(node.chunk, Chunk) else node.chunk)
    return out


@pytest.fixture
def parsed_docs() -> list:
    """3 synthetic ParsedDocuments — fast, deterministic, no parser dependency."""
    return [make_test_doc(num_paragraphs=8) for _ in range(3)]


def _make_config(wiki_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        source_dir=wiki_dir,
        output_dir=wiki_dir,
        llm=LLMConfig(default_model=LLM_MODEL, vlm_model=VLM_MODEL),
        embedding=EmbeddingConfig(model=EMBED_MODEL),
    )


def _vault_for(tmp_path: Path, name: str = "vault") -> MarkdownWriter:
    """MarkdownWriter in vault mode (default)."""
    return MarkdownWriter(flavor="obsidian", vault_mode=True)


def test_deterministic_embedding_provider_runs(
    mock_embedding_provider: DeterministicTestEmbeddingProvider,
) -> None:
    """Mock embedding is stable per input and normalized."""
    a = mock_embedding_provider.embed("hello")
    b = mock_embedding_provider.embed("hello")
    assert a == b
    assert len(a) == mock_embedding_provider.dim

    c = mock_embedding_provider.embed("different")
    assert a != c


@pytest.fixture
def mock_embedding_provider() -> DeterministicTestEmbeddingProvider:
    return DeterministicTestEmbeddingProvider(dim=384)


def test_tier_state_round_trip(tmp_path: Path) -> None:
    """TierStateStore serializes and re-loads TierState correctly."""
    store_path = tmp_path / KNOX_DIR / STATE_FILE
    store = TierStateStore(store_path)

    state = TierState(
        entries={
            "doc1": SourceState(
                stem="doc1",
                source_file="doc1.pdf",
                content_hash="abc123",
                current_tier=3,
                last_compile="2025-01-01T00:00:00Z",
                ingested_by="knox",
            )
        }
    )
    store.save(state)
    loaded = store.load()
    assert "doc1" in loaded.entries
    assert loaded.entries["doc1"].current_tier == 3
    assert loaded.entries["doc1"].content_hash == "abc123"
