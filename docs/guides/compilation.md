# Compilation Guide

KNOX compiles documents into a knowledge wiki through a five-tier pipeline.
This guide explains incremental compilation, compiler backends, prompt
budgeting, and per-source state.

## Pipeline tiers

| Tier | Name | What runs | Cost | Default |
|------|------|-----------|------|---------|
| 0 | Index | Parse + persist state | Free | ✅ |
| 1 | Chunk | Index + chunk documents | Free | ✅ |
| 2 | Resolve | Chunks → entities / references | Free | ✅ |
| 3 | Compile | Resolve → wiki pages | ~$0.05/doc | ✅ |
| 4 | Output | Write wiki to disk | Free | ✅ |

## Incremental compilation

Use `--start-at-tier` to skip tiers whose cached output is still valid. The pipeline validates that previous tiers ran before continuing.

```bash
knox compile *.pdf                                # full pipeline (T0→T4)
knox compile *.pdf --start-at-tier resolve        # skip re-parse+chunk, T2→T4
knox compile *.pdf --start-at-tier output         # re-render existing wiki, T4 only
knox compile *.pdf --force                        # invalidate all caches, full re-run
```

| Flag | Effect |
|------|--------|
| `--start-at-tier index` (default) | Run T0→T4 |
| `--start-at-tier chunk` | Skip T0 (uses cached parse output) |
| `--start-at-tier resolve` | Skip T0+T1 (uses cached chunks) |
| `--start-at-tier compile` | Skip T0–T2 (uses cached resolved graph) |
| `--start-at-tier output` | Re-render from cached CompiledWiki only |
| `--force` | Ignore all caches; full re-run |

### Ingest (T0+T1 only)

`knox ingest` runs parse + chunk and stops. Use it to pre-stage large corpora cheaply before an LLM compile:

```bash
knox ingest *.pdf              # stage + parse + chunk only
knox compile --start-at-tier resolve   # later: T2→T4 using cached chunks
```

The pipeline loads its tier checkpoint from `.knox/compile_state.json`
inside the output directory. On restart, unchanged sources are skipped.

## Two-phase compiler (`--compiler two-phase`)

The **non-default** `TwoPhaseCompiler` runs a concept-extraction pipeline:

1. **Extraction pass** — one LLM call per source, returns structured JSON
   (`{summary, entities, concepts, tags}`).
2. **Generation pass** — concepts from all sources are merged by name and
   one page is generated per unique concept. Cross-source edges are added
   for shared entities.

```bash
knox compile *.pdf --compiler two-phase
```

To use the **default** graph compiler explicitly:

```bash
knox compile *.pdf --compiler graph
```

## Vision enrichment (VLM)

When a vision-capable model is configured, KNOX can caption extracted images and figures during Tier 2 resolution:

```bash
knox compile *.pdf --vlm-model openai/gpt-4o
# or via environment variable
export KNOX_VLM_MODEL=ollama/llava
knox compile *.pdf
```

The VLM receives the extracted image plus surrounding context and produces a caption that is carried into the compiled wiki.

## Typed knowledge-graph relations

By default, KNOX links pages only by shared entity mentions (`entity_reference`). With relation extraction enabled, the resolver asks the LLM to identify typed relations between entities (e.g., `uses`, `depends_on`, `contradicts`, `caused`, `fixed`, `supersedes`) and attaches a confidence score:

```bash
knox compile *.pdf --resolve-relations --relation-confidence-threshold 0.7
# or via config
export KNOX_RESOLVE_RELATIONS=true
```

Typed relations are emitted as graph edges and rendered in the `Related` section of each wiki page (`[[Page]] (uses) — confidence 0.85`).

## Hybrid search

The `knox query` command fuses four result channels via reciprocal rank fusion (RRF, k=60):

```mermaid
graph LR
    Q[Query] --> FTS[FTS5 index]
    Q --> VEC[zvec vector store]
    Q --> TOK[Token overlap]
    Q --> RAW[Raw sources]
    FTS --> RRF[RRF fusion]
    VEC --> RRF
    TOK --> RRF
    RAW --> RRF
    RRF --> R[Ranked results]
```

The vector store is built at compile time with `--index --index-store` and re-opened at query time using the persisted embedding model. If the vector store is absent, the retriever falls back to token overlap only.

## Page kinds

The default `graph` compiler infers a built-in page kind for each wiki page:

| Kind | Heuristic |
|------|-----------|
| `overview` | Root or "Overview" page |
| `comparison` | Comparison/contrast-style heading context |
| `entity` | Page title matches a resolved entity name |
| `concept` | Default for everything else |

The kind is written to the page frontmatter (`kind: entity`) and included in the compiled wiki metadata distribution.

## Memory lifecycle

KNOX tracks every wiki page through a four-tier memory lifecycle with confidence scoring, supersession detection, and retention decay:

```mermaid
graph LR
    W[WORKING] -->|2 reviews| E[EPISODIC]
    E -->|5 reviews| S[SEMANTIC]
    S -->|10 reviews| P[PROCEDURAL]
    W -.->|stale| X[Decayed]
    E -.->|stale| X
    S -.->|stale| X
```

| Concept | Description |
|---------|-------------|
| **Confidence** | Starts at 0.5, asymptotically approaches 1.0 with each compile cycle |
| **Retention** | `confidence * exp(-days_since_review / half_life)` — pages decay if not recompiled |
| **Supersession** | If a new page has the same title but a different node ID, the old page is marked `superseded_by` |
| **Tier promotion** | WORKING→EPISODIC (2 reviews), EPISODIC→SEMANTIC (5), SEMANTIC→PROCEDURAL (10) |

```bash
knox compile *.pdf --memory              # enable (default)
knox compile *.pdf --no-memory           # disable
knox compile *.pdf --memory-half-life-days 180
knox compile *.pdf --memory-decay-threshold 0.2
```

Memory state is persisted at `base_dir/.knox/memory.json`. Stale and superseded pages surface as lint failures in the `stale-page` rule.

## Event hooks

KNOX fires lifecycle events that user callbacks can observe. Hooks are registered in `knox.toml` under `[hooks]` using a `module:attribute` syntax:

```toml
[hooks]
on_query = ["my_plugin:log_query"]
post_compile = ["my_plugin:reindex", "my_plugin:notify"]
```

| Event | When | Context keys |
|-------|------|-------------|
| `PRE_COMPILE` | Start of `pipeline.run()` | `source_dir`, `output_dir`, `mode` |
| `POST_COMPILE` | After compile + memory update | `wiki`, `output_dir`, `node_count` |
| `ON_CAPTURE` | After `capture()` / `learn()` writes files | `paths`, `kind` |
| `ON_QUERY` | After search results in `query` command | `query`, `results` |
| `ON_LINT` | After lint report is generated | `report` |

```mermaid
sequenceDiagram
    participant U as User
    participant CLI
    participant Pipeline
    participant H as Hooks

    U->>CLI: knox compile
    CLI->>H: load_hooks_from_config()
    CLI->>Pipeline: run()
    Pipeline->>H: fire(PRE_COMPILE)
    Pipeline->>Pipeline: parse → chunk → resolve → compile
    Pipeline->>H: fire(POST_COMPILE)
    Pipeline-->>CLI: CompiledWiki
```

Hook callbacks receive a `HookContext` and exceptions are isolated — a failing hook never breaks the pipeline.

## Prompt budget

Each concept page in phase 2 sees at most `--prompt-budget` characters
of source content (default 200 000). When a concept draws from many
sources, the content is **fair-share truncated** — each source gets
an equal slice, not first-N-chunks. Excess truncation is logged.

```bash
knox compile *.pdf --prompt-budget 100000
```

## Per-source state

KNOX persists a tier checkpoint for every source file:

```json
{
  "entries": {
    "paper": {
      "stem": "paper",
      "source_path": "sources/paper.pdf",
      "content_hash": "a1b2c3...",
      "current_tier": 3,
      "last_compile": "2026-06-08T12:34:56+00:00",
      "search_hits": 0
    }
  }
}
```

The file lives at `<output>/.knox/compile_state.json`. Delete it (or
pass `--force`) to force a full recompile.

## Tier listing

```bash
knox compile *.pdf --list-tiers
```

Prints a JSON list of every tier with the count of sources that have
reached it. Useful in CI for "how complete is this wiki?" checks.

## FAQ

**Q: How do I re-compile a single source?**
A: `knox compile sources/foo.pdf --force`

**Q: Why is my source at tier 0 after a failed compile?**
A: The tier is updated only after the chunk succeeds. Re-run `knox compile`.

**Q: How does merge work on re-ingest?**
A: In two-phase mode, entities/concepts from the new source are merged
with existing pages — old content is preserved, new provenance is added.

**Q: Can I run just the LLM stages?**
A: Yes: `knox compile *.pdf --start-at-tier resolve` skips re-parsing and chunking.

**Q: How do I get a JSON status report?**
A: `knox compile *.pdf --list-tiers` outputs JSON.
