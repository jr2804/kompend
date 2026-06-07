# Decision Log

All architectural, tooling, and design decisions for KNOX. Each entry includes the decision, the context, and the rationale.

## Naming

### Decision: Project name is "KNOX"

**Date:** Jun 5, 2026  
**Context:** Need a pronounceable, memorable name for the compiled LLM-wiki system. Must avoid collision with existing tools ("llm-wiki", "sage-wiki", "llmwiki" all taken).  
**Options considered:**

- KNOX (Knowledge Organizer for eXtended documents) — vault-like, strong
- KITE (Knowledge Ingestion & Transformation Engine) — light, fast
- WIKEN (Wiki Knowledge ENgine) — clear intent
- KARBY (KARpathy Better knowledge sYstem) — playful nod
- WISE (Wiki-Ingestion System for Enhanced documents) — professional

**Decision:** KNOX. Short, pronounceable, implies a secure/organized knowledge vault.
**Consequences:** Slug is `knox`, repo at `codeberg.org/jr2804/knox`.

## Stack

### Decision: Python backend with uv

**Date:** Jun 5, 2026  
**Context:** Need fast iteration, rich ML/NLP ecosystem, easy agent integration. Go (sage-wiki) is fast but lacks Docling/pandas/litellm. Rust is too slow for prototyping.  
**Decision:** Python 3.12+, uv for packaging, mise for task running.
**Consequences:** Cannot produce a single static binary. Distribution via PyPI + `uv pip install`.

### Decision: Use copier-uv-plus template

**Date:** Jun 6, 2026  
**Context:** Need project scaffolding with modern Python tooling (ruff, ty, mise tasks, AGENTS.md, skills).  
**Decision:** `codeberg.org/jr2804/copier-uv-plus` v1.3.0.
**Consequences:** Template provides `.config/mise/`, `ruff.toml`, `ty.toml`, `.pre-commit-config.yaml`. Must be properly instantiated via copier (not manual). Initially botched — template had Jinja ext dependencies; fixed by installing `copier-template-extensions`.

### Decision: AGPL-3.0 license

**Date:** Jun 5, 2026  
**Context:** Open-source from day 1, but private until first release. Want copyleft to protect the commons.  
**Decision:** AGPL-3.0. Strong copyleft, viral, protects against proprietary forks.
**Consequences:** Some enterprises may avoid. Acceptable for research/academic tool.

## Architecture

### Decision: 5-tier pipeline with decoupled parsing

**Date:** Jun 5, 2026  
**Context:** Existing tools (sage-wiki, llmwiki) bake weak parsers into the compiler. This causes the "flattening problem".  
**Decision:** Decouple parsing (Tier 1) from compilation (Tier 4). Parsers produce `ParsedDocument` with structured JSON sidecar. Compilers consume structured data, not flat markdown.
**Consequences:** More code, more interfaces. But each tier independently replaceable and testable.

### Decision: Hybrid chunking (custom + Chonkie)

**Date:** Jun 6, 2026  
**Context:** Chonkie operates on text/markdown strings. KNOX's parser produces typed `DocumentElement` objects. Feeding typed objects into Chonkie means serialization → lossy.  
**Decision:** Custom structure-aware chunker for atomic elements (tables/figures/equations never split) + Chonkie for text-heavy sections.
**Consequences:** Two code paths. Must maintain consistency between custom chunker and Chonkie token counting.

### Decision: WiCER iterative compilation loop

**Date:** Jun 6, 2026  
**Context:** WiCER paper (arXiv:2605.07068v1) shows blind compilation over-compresses 2–3× beyond targets. Quality loss is catastrophic, not gradual.  
**Decision:** Implement CEGAR-style loop: compile → evaluate probe Q&A → diagnose failures → refine with pinned facts → repeat.
**Consequences:** Compilation is 2–3× slower than one-shot. But quality recovered to ~80% of full-context baseline. Acceptable tradeoff.

## Parsing

### Decision: Docling as primary PDF parser

**Date:** Jun 5, 2026  
**Context:** Benchmark winner is opendataloader-pdf (0.907), but Docling (0.882) has better ecosystem (MCP server, IBM backing, Python-native).  
**Decision:** Docling as primary. opendataloader-pdf as alternative via `BaseParser` swap.
**Consequences:** Slightly lower benchmark scores. But easier integration, broader format support, active development.

### Decision: Format-native parsing, not PDF-first

**Date:** Jun 6, 2026  
**Context:** "Convert anything to PDF first" simplifies the pipeline but loses native structure from DOCX/XLSX.  
**Decision:** Route by format: PDF→Docling, DOCX→Docling/pandoc, XLSX→pandas, PPTX→pandoc. Keep native structure where possible.
**Consequences:** More parser modules to maintain. But tables from XLSX retain formulas; DOCX styles map to headings.

### Decision: Image extraction mode configurable

**Date:** Jun 5, 2026  
**Context:** Extracted images can be embedded as base64 (self-contained) or saved as files (smaller JSON, referenced by path).  
**Decision:** Configurable `image_mode`: `"base64"` (embedded, portable) or `"file"` (external, smaller). Default `"file"`.
**Consequences:** Base64 produces large JSON sidecars. File mode requires path management.

## Chunking

### Decision: Structure-aware chunking: atomic elements never split

**Date:** Jun 6, 2026  
**Context:** Splitting a table or figure across chunks destroys queryability. "What was the performance requirement in Section 3.2?" requires the full table.  
**Decision:** Tables, figures, equations, code blocks are atomic — never split. Only text sections get chunked.
**Consequences:** Some chunks may exceed token budget if a table is very large. Use `TableChunker` for row-wise splitting in that case.

### Decision: Model token limit auto-detection via litellm

**Date:** Jun 6, 2026  
**Context:** Different models have different context windows (4K, 8K, 32K, 128K, 1M). Manual configuration is error-prone.  
**Decision:** Use litellm's model metadata to auto-detect context window. Reserve 20% for system prompts.
**Consequences:** Requires litellm to have metadata for the model. Edge cases: unknown modelsfall back to 32K default.

## Resolution

### Decision: One-shot default, multi-turn opt-in

**Date:** Jun 5, 2026  
**Context:** Multi-turn resolution (abbreviation expansion, cross-reference validation) is powerful but slow (seconds per document).  
**Decision:** Default: one-shot ingestion (fast). Per-document opt-in: `--resolve` flag enables multi-turn.
**Consequences:** Most documents compile fast. Complex papers get extra attention on demand.

### Decision: VLM for figure captioning, not native parser

**Date:** Jun 6, 2026  
**Context:** Docling has vision fallback for figures, but caption quality is poor. Standalone VLM (GPT-4V, Claude 3, Gemini) produces better captions.  
**Decision:** Use parser for figure extraction (PNG + bbox + surrounding text). Send to VLM for captioning during resolution tier.
**Consequences:** Requires VLM API or local model. Adds latency. But captions are queryable.

## Compilation

### Decision: litellm for provider abstraction

**Date:** Jun 5, 2026  
**Context:** Need to support Ollama (local), OpenRouter, DeepSeek, Anthropic, OpenAI without hard-coding each.  
**Decision:** litellm. Single API, all providers.
**Consequences:** Adds dependency. But eliminates provider-specific code.

### Decision: Local-first model strategy

**Date:** Jun 6, 2026  
**Context:** Running everything via API is expensive and slow. Local models on BunStation (Ollama) are fast and free.  
**Decision:** Default to local Ollama models for pre-processing (chunking heuristics) and WiCER evaluation (LLM-as-judge). Fallback to remote for compilation synthesis if quality insufficient.
**Consequences:** Requires Ollama running. But dramatically reduces cost.

## Output

### Decision: No custom GUI. Use Open-WebUI

**Date:** Jun 6, 2026  
**Context:** Building a web UI is time-consuming and not core value. Existing tools (Open-WebUI) already solve this.  
**Decision:** KNOX exposes CLI + MCP + Python SDK. Open-WebUI connects via API/MCP.
**Consequences:** Users who want a web UI need separate Open-WebUI install. But zero UI dev effort.

### Decision: Markdown + JSON dual output

**Date:** Jun 6, 2026  
**Context:** Markdown for humans (Obsidian), JSON for machines (graph queries, MCP tools).  
**Decision:** Always produce both. Markdown is primary human-readable. JSON is primary machine-readable.
**Consequences:** Double disk usage. Acceptable.

## Integration

### Decision: MCP server = thin CLI wrapper

**Date:** Jun 5, 2026  
**Context:** Some tools duplicate business logic between CLI and MCP. This causes drift.  
**Decision:** MCP server calls CLI commands. Zero logic in MCP layer.
**Consequences:** MCP is slower (fork/exec overhead). But no drift.

### Decision: Hermes skill wraps CLI, not MCP

**Date:** Jun 5, 2026  
**Context:** Hermes can call either MCP tools or CLI commands. Which is better?  
**Decision:** Hermes skill calls CLI (`knox compile`, `knox parse`). Simpler, no MCP server required.
**Consequences:** Hermes must have knox installed. But no separate MCP process to manage.

## Testing

### Decision: Use listening-effort corpus for local dev

**Date:** Jun 5, 2026  
**Context:** Need real-world scientific papers for testing. ETSI standards, AES papers as benchmark.  
**Decision:** Use `Worky/research/listening-effort/` corpus. Never commit to upstream repo.
**Consequences:** Local testing only. CI cannot replicate without corpus.

## References

- Worky's research: `Worky/research/llm-wiki/` (Obsidian vault)
- WiCER paper: <https://arxiv.org/pdf/2605.07068v1>
- Karpathy's Gist: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
