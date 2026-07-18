# KNOX

<p align="center">
  <img src="./assets/readme/hero.svg" width="100%"
       alt="KNOX — Knowledge Organizer for eXtended documents: compile scattered documents into an interlinked, queryable wiki with any LLM">
</p>

KNOX refuses to pick a side in the LLM-wiki space. Existing tools either
demand a frontier-model subscription **and** lock you into their stack, or
insist everything run on an underpowered local model while lecturing you
about *privacy*. KNOX is provider-agnostic, adapts to whatever LLM you have
— from a local 8B model to a frontier API — and uses structure-preserving
parsing so technical content (tables, equations, figures) survives
extraction intact. [Read the full motivation →](docs/about/motivation.md)

> **OKF v0.1 compliant** — produces [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) bundles by default.
> **ACP agent** — implements the [Agent Client Protocol](https://agentclientprotocol.com) for credential-free LLM injection via `knox agent`.
> **PydanticAI-native** — every LLM call routes through typed agents with failover chains; [models.dev](https://models.dev) catalog resolves any provider.

## How it works

<p align="center">
  <img src="./assets/readme/pipeline.svg" width="100%"
       alt="KNOX 5-tier pipeline: Preprocessing → Chunking → Resolution → Compilation → Output">
</p>

KNOX runs a 5-tier pipeline — each tier independently replaceable, with a
clear `Base` interface:

| Tier | What it does | Needs LLM? |
|------|-------------|:----------:|
| **T1 Preprocessing** | Parse PDF, DOCX, PPTX, EPUB, XLSX, HTML, URLs into a uniform `ParsedDocument` preserving structure (headings, tables, figures, equations, code). | ❌ |
| **T2 Chunking** | Split into a `ChunkTree` sized to the model's context window (auto-detected). Atomic elements never split; text runs via recursive paragraph→line→sentence→token boundaries. | ✅ |
| **T3 Resolution** | Extract entities, resolve cross-references, acronym/definition consistency. Optional VLM enricher for figure captioning. | ✅ |
| **T4 Compilation** | Build an interlinked `CompiledWiki` graph. Two backends: `TwoPhaseCompiler` (default, extract→generate) and `WiCERCompiler` (iterative compile→evaluate→refine). | ✅ |
| **T5 Output** | Write OKF bundles (default), Markdown, JSON, or Typst. Optionally index for semantic search and synthesize an overview. | ❌ |

`knox ingest` stops at T1 — **no LLM, no cost**. `knox compile` runs T2–T5.

## Quick start

```bash
uv tool install knox

# 1. Initialize a project
knox init ./my-wiki && cd ./my-wiki

# 2. Ingest sources (T1 only — no LLM needed)
knox ingest                              # everything in sources/
knox ingest paper.pdf notes.docx *.xlsx  # explicit files
knox ingest --url https://arxiv.org/pdf/2605.07068v1

# 3. Compile the wiki (T2–T5 — needs an LLM)
knox compile --set compile.llm.default_model=ollama/qwen3

# 4. Query it
knox query "How does attention scale with sequence length?"
```

> **Debugging tip:** `knox parse paper.pdf --elements` inspects one file's
> extracted structure before ingesting the full corpus. If parsing looks
> wrong, fix it first — the pipeline won't recover from bad extraction.

## Commands

| Command | Purpose |
|---|---|
| `knox init [DIR]` | Scaffold a wiki project (`--bare`, `--force`, `--git`) |
| `knox ingest FILES...` | Parse sources into `.knox/cache/` (T0–1, no LLM) |
| `knox compile FILES...` | Full pipeline → interlinked wiki (T2–5, needs LLM) |
| `knox query "..."` | Ask a natural-language question against the wiki |
| `knox organize` | LLM-categorize `sources/` into a hierarchy and move files |
| `knox parse FILES...` | Debug: parse documents and print structure (`--elements`) |
| `knox doctor` | Validate dirs, parsers, LLM/VLM connectivity |
| `knox status` | Show source/article/word counts and last compile time |
| `knox config get/set/list/path` | Inspect and edit configuration (5-layer precedence) |
| `knox lint run` | Lint wiki: 8 structural + 4 LLM rules (`--fix`, `--llm`) |
| `knox review list/approve/reject` | Manage the review queue for compiled pages |
| `knox capture "..."` / `learn` / `distill` | Capture notes → synthesize into wiki pages |
| `knox write "TITLE"` | One-shot wiki page from a title + optional sources |
| `knox trace [SLUG]` | Trace a page to its source files |
| `knox memory add/search/synthesize` | Manage query memory and context synthesis |
| `knox serve` | Start the web UI (`--host`, `--port`, `--open`) |
| `knox mcp` | Start the MCP server (stdio, for AI clients) |
| `knox agent` | Start the ACP agent (editor integration) |

Full option reference: `knox <command> --help`

## Key subsystems

| Subsystem | Description |
|---|---|
| **RAG / Search** | `HybridRetriever` fuses FTS + vector + token-overlap + raw sources via reciprocal rank fusion. Embeddings via `litellm:` prefix or sentence-transformers. |
| **Memory lifecycle** | Confidence decay, tier promotion (working → episodic → semantic → procedural), supersession detection. |
| **Review queue** | Route compiled pages through approve/reject/lock before publishing. |
| **MCP server** | Stdio server exposing `parse_document`, `compile_wiki`, `wiki_capture`, `wiki_learn` to Claude Code, Cursor, etc. |
| **ACP agent** | Agent Client Protocol server: editors inject LLM credentials via `providers/set` — no API keys in your config. |
| **Configuration** | 5-layer precedence: CLI > env (`KNOX_*`) > project (`.knox/config.toml`, walks up) > global (`~/.config/knox/`) > defaults. |

## LLM configuration

KNOX uses `provider/model` strings resolved by `build_model()`:

1. **models.dev catalog** — custom providers with known API URLs (`claudinio/`, `opencode/`, `deepseek/`) construct OpenAI-compatible models.
2. **pydantic-ai native** — standard providers (`ollama/`, `openai/`, `anthropic/`, `google/`) via `infer_model()`.

API keys come from the env var each provider advertises. No `api_key` in config:

```toml
[compile.llm]
default_model = "ollama/qwen3"
fallback_chain = ["openai/gpt-4o"]   # tried when the primary fails
reasoning_effort = "none"             # tame over-thinking local models
```

See the [LLM configuration guide](./docs/guides/llm-config.md) for the full reference.

## Documentation

> **Full docs at [knox.anomalyco.dev](https://knox.anomalyco.dev)** — architecture deep-dives, tier references, WiCER loop, CLI guides, integrations.
>
> See [`docs/index.md`](./docs/index.md) for the complete documentation index.

## Development

```bash
git clone <repo-url> && cd knox
uv sync                    # install deps
uv run knox --help         # verify CLI
uv run pytest tests/       # run tests
mise test                  # run all quality checks
```

## License

MIT. See [LICENSE](./LICENSE).

## Acknowledgments

- [Karpathy's compiled LLM-wiki Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — the core concept
- [Rohit G00's LLM Wiki v2 Gist](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2) — memory lifecycle, knowledge graph, hybrid search
- [WiCER paper](https://arxiv.org/pdf/2605.07068v1) (Huerta, Zinnia Tech) — iterative compilation loop
- [Docling](https://github.com/DS4SD/docling) (IBM) — structure-preserving parsing
- [Chonkie](https://github.com/chonkie-inc/chonkie) — multi-strategy chunking
- [pydantic-ai](https://ai.pydantic.dev) — typed agent framework
- [models.dev](https://models.dev) — provider/model catalog
