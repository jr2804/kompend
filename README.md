# KNOX

**Knowledge Organizer for eXtended documents**

A compiled LLM-wiki system for scientific and technical documents. Parse, chunk, resolve, and compile PDFs, DOCX files, spreadsheets, and web sources into an interlinked, queryable knowledge base — with structure-preserving parsing, hybrid chunking, multi-turn resolution, and WiCER iterative compilation.

Inspired by [Karpathy's compiled LLM-wiki concept](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Quick Start

```bash
# Install
uv pip install "knox[docling]"

# Compile a sources directory into a wiki
knox compile ./sources --output ./wiki

# Parse a single document
knox parse ./paper.pdf --output ./parsed/

# See all options
knox --help
```

## What Makes KNOX Different

| Problem | Existing Tools | KNOX |
|---|---|---|
| Tables/figures flattened to text | All (pdfplumber → markdown → wiki) | Structure-preserving parser (Docling JSON sidecar) |
| No structure-aware chunking | Most use char-count splitting | Hybrid: custom atomic splitter + Chonkie for text |
| One-shot compilation | "Compile once and query" | WiCER loop: compile → evaluate → refine |
| Multi-turn resolution missing | No tool supports it | Tier 3: LLM + MCP tools per document |
| Model/context-agnostic | Hard-coded OpenAI/Anthropic | litellm + auto-detect context window + budget |
| GUI required for basic ops | nashsu/llm_wiki GUI-only | Headless CLI + MCP + Python SDK |
| Images orphaned | Flattened to filenames | VLM-captioned, indexed as first-class entities |

## Architecture

5-tier pipeline — each tier independently replaceable:

```
Raw Sources → TIER 1: Pre-Processing → TIER 2: Chunking → TIER 3: Resolution → TIER 4: Compilation → TIER 5: Output
```

Read the full architecture in [`docs/architecture.md`](./docs/architecture.md).

## Documentation

| Document | What's Covered |
|---|---|
| [`docs/overview.md`](./docs/overview.md) | Project summary, design philosophy, core principles |
| [`docs/architecture.md`](./docs/architecture.md) | 5-tier pipeline, data models, integration points |
| [`docs/ecosurvey.md`](./docs/ecosurvey.md) | Survey of existing tools, parsers, chunkers, MCP servers |
| [`docs/roadmap.md`](./docs/roadmap.md) | Milestones, priorities, open questions, tentative dates |
| [`docs/decisions.md`](./docs/decisions.md) | Decision log — what was chosen and why |

## Development

```bash
cd /opt/workspace/work/knox
uv sync                    # install deps
uv run pytest tests/       # run tests
mise test                  # run all quality checks
```

## Project Status

| Milestone | Status |
|---|---|
| M0: Scaffolding | ✅ Complete |
| M1: Docling Parser (Tier 1) | ✅ Complete |
| M2: Hybrid Chunking (Tier 2) | 🎯 Current |
| M3: LLM Resolution (Tier 3) | ⏳ Pending |
| M4: WiCER Compilation (Tier 4) | ⏳ Pending |
| M5: Output Writers (Tier 5) | ⏳ Pending |
| M6: MCP Server | ⏳ Pending |

See [`docs/roadmap.md`](./docs/roadmap.md) for full details.

## License

AGPL-3.0. See [LICENSE](./LICENSE).

## Contact

Jan Reimes — jan@reimes.de

## Acknowledgments

- [Karpathy's compiled LLM-wiki Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — the core concept
- [WiCER paper](https://arxiv.org/pdf/2605.07068v1) (Huerta, Zinnia Tech) — iterative compilation loop
- [Docling](https://github.com/DS4SD/docling) (IBM) — structure-preserving parsing
- [Chonkie](https://github.com/chonkie-inc/chonkie) — multi-strategy chunking
- [litellm](https://github.com/BerriAI/litellm) — provider-agnostic LLM calls
