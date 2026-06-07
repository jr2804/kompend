# KNOX — Knowledge Organizer for eXtended documents

**tl;dr:** A compiled LLM-wiki system for scientific/technical documents with structure-preserving parsing, hybrid chunking, multi-turn resolution, and WiCER iterative compilation. CLI-first, model-agnostic, agent-native.

## What This Is

KNOX turns raw source documents (PDFs, DOCX, XLSX, PPTX, web articles) into an interlinked, queryable knowledge wiki — compiled once, queried fast. It is inspired by [Karpathy's compiled LLM-wiki concept](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) but built for the pain points existing tools ignore.

## Why Existing Frameworks Fail

| Problem | Existing Tools | KNOX Fix |
|---|---|---|
| Tables/figures flattened to text | All (pdfplumber → markdown → wiki) | Structure-preserving parser (Docling JSON sidecar) |
| No structure-aware chunking | Most use char-count splitting | Hybrid: custom atomic splitter + Chonkie for text |
| One-shot compilation | "Compile once and query" | WiCER loop: compile → evaluate → refine |
| Multi-turn resolution missing | No tool supports it | Tier 3: LLM + MCP tools per document |
| Model/context-agnostic | Hard-coded OpenAI/Anthropic | litellm + auto-detect context window + budget |
| GUI required for basic ops | nashsu/llm_wiki GUI-only | Headless CLI + MCP + Python SDK |
| Images orphaned | Flattened to filenames | VLM-captioned, indexed as first-class entities |

## Project Status

| | Status |
|---|---|
| **Name** | KNOX (slug: `knox`) |
| **Repository** | `codeberg.org/jr2804/knox` (private until first release) |
| **License** | AGPL-3.0 |
| **Language** | Python 3.12+ |
| **Package Manager** | uv |
| **CI/CD** | None (manual) |
| **Template** | [copier-uv-plus](https://codeberg.org/jr2804/copier-uv-plus) v1.3.0 |
| **Current** | Architecture scaffold complete. Milestone 1 (Docling parser) done. Milestone 2 pending. |

## Where to Read More

| Document | Purpose |
|---|---|
| [`architecture.md`](./architecture.md) | 5-tier architecture, data models, core principles |
| [`ecosurvey.md`](./ecosurvey.md) | Survey of existing frameworks + gaps |
| [`roadmap.md`](./roadmap.md) | Milestones, priorities, open questions |
| [`decisions.md`](./decisions.md) | Decision log (what was chosen and why) |

## Quick Start

```bash
cd /opt/workspace/work/knox
uv sync                    # install deps
uv run knox --help         # see CLI commands
uv run pytest tests/       # run tests
```

## Core Principles

1. **Parser-agnostic** — Use the best pre-processor per format. Do not bake a weak parser into the compiler.
2. **Structure-preserving** — Tables, figures, equations are first-class entities with metadata, not flattened text.
3. **Model-aware** — Detect context window, chunk intelligently, use cheap models for bulk work, expensive models only for synthesis.
4. **Agent-native** — MCP server + CLI + Python SDK. No GUI dependency. Headless cron-friendly.
5. **Resolution-ready** — Hooks for MCP/CLI tools during ingestion. One-shot is the default; multi-turn is opt-in per document.
6. **Multimodal** — Vision-capable LLMs analyze extracted figures during compilation. Figures become searchable concepts.
7. **Iterative compilation (WiCER)** — Compile, evaluate with probe Q&A, diagnose failures, refine with pinned facts. Not one-shot.
8. **Hybrid chunking** — Custom structure-aware splitting for atomic elements + Chonkie for text-heavy sections.

## Contact

Author: Jan Reimes  
Email: <jan@reimes.de>
