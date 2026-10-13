# Compilation Guide

kompend compiles documents into a knowledge wiki through a five-tier pipeline.
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
kompend compile *.pdf                                # full pipeline (T0→T4)
kompend compile *.pdf --start-at-tier resolve        # skip re-parse+chunk, T2→T4
kompend compile *.pdf --start-at-tier output         # re-render existing wiki, T4 only
kompend compile *.pdf --force                        # invalidate all caches, full re-run
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

`kompend ingest` downloads URLs and parses every source into structured
JSON — no chunking, no LLM. Use it to pre-stage large corpora cheaply
before an LLM compile:

```bash
kompend ingest *.pdf              # download + parse only (no LLM)
kompend compile --start-at-tier chunk   # later: T2→T5 using parsed sources
```

Parsed documents are cached in `.kompend/kompend.db` using the same key format
as the pipeline, so compile picks them up transparently.

## WiCER compiler (config: `compile.compiler = wicer`)

The **non-default** `TwoPhaseCompiler` runs a concept-extraction pipeline:

1. **Extraction pass** — one LLM call per source, returns structured JSON
   (`{summary, entities, concepts, tags}`).
2. **Generation pass** — concepts from all sources are merged by name and
   one page is generated per unique concept. Cross-source edges are added
   for shared entities.

```bash
kompend compile *.pdf --set compile.compiler=wicer
```

To use the **default** graph compiler explicitly:

```bash
kompend compile *.pdf --set compile.compiler=graph
```

## Vision enrichment (VLM)

When a vision-capable model is configured, kompend can caption extracted images and figures during Tier 2 resolution:

```bash
kompend compile *.pdf --vlm-model openai/gpt-4o
# or via environment variable
export KOMP_VLM_MODEL=ollama/llava
kompend compile *.pdf
```

The VLM receives the extracted image plus surrounding context and produces a caption that is carried into the compiled wiki.

## Typed knowledge-graph relations

By default, kompend links pages only by shared entity mentions (`entity_reference`). With relation extraction enabled, the resolver asks the LLM to identify typed relations between entities (e.g., `uses`, `depends_on`, `contradicts`, `caused`, `fixed`, `supersedes`) and attaches a confidence score:

```bash
kompend compile *.pdf --resolve-relations --relation-confidence-threshold 0.7
# or via config
export KOMP_RESOLVE_RELATIONS=true
```

Typed relations are emitted as graph edges and rendered in the `Related` section of each wiki page (`[[Page]] (uses) — confidence 0.85`).

## Hybrid search

The `kompend query` command fuses four result channels via reciprocal rank fusion (RRF, k=60):

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

The vector store is built at compile time by default (indexing is on; pass `--no-index` to skip) and re-opened at query time using the persisted embedding model. If the vector store is absent, the retriever falls back to token overlap only.

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

kompend tracks every wiki page through a four-tier memory lifecycle with confidence scoring, supersession detection, and retention decay:

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
kompend compile *.pdf --memory              # enable (default)
kompend compile *.pdf --no-memory           # disable
kompend compile *.pdf --set compile.memory_half_life_days=180
kompend compile *.pdf --set compile.memory_decay_threshold=0.2
```

Memory state is persisted in the `memory_states` table of `base_dir/.kompend/kompend.db`. Stale and superseded pages surface as lint failures in the `stale-page` rule.

## Event hooks

kompend fires lifecycle events that user callbacks can observe. Hooks are registered in `kompend.toml` under `[hooks]` using a `module:attribute` syntax:

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

    U->>CLI: kompend compile
    CLI->>H: load_hooks_from_config()
    CLI->>Pipeline: run()
    Pipeline->>H: fire(PRE_COMPILE)
    Pipeline->>Pipeline: parse → chunk → resolve → compile
    Pipeline->>H: fire(POST_COMPILE)
    Pipeline-->>CLI: CompiledWiki
```

Hook callbacks receive a `HookContext` and exceptions are isolated — a failing hook never breaks the pipeline.

## Prompt budget

Each concept page in phase 2 sees at most `compile.budget.gen_output_max`
characters of source content. When a concept draws from many sources, the
content is **fair-share truncated** — each source gets an equal slice, not
first-N-chunks. Excess truncation is logged. Tune it in config:

```bash
kompend config set compile.budget.gen_output_max 100000
kompend compile *.pdf --set compile.budget.gen_output_max=100000
```

## Per-source state

kompend persists a tier checkpoint for every source file in the `tier_state`
table of `.kompend/kompend.db` (stem, source file, content hash, current tier,
last compile, search hits). Pass `--force` to force a full recompile.

## Tier listing

```bash
kompend compile *.pdf --list-tiers
```

Prints a JSON list of every tier with the count of sources that have
reached it. Useful in CI for "how complete is this wiki?" checks.

## FAQ

**Q: How do I re-compile a single source?**
A: `kompend compile sources/foo.pdf --force`

**Q: Why is my source at tier 0 after a failed compile?**
A: The tier is updated only after the chunk succeeds. Re-run `kompend compile`.

**Q: How does merge work on re-ingest?**
A: In two-phase mode, entities/concepts from the new source are merged
with existing pages — old content is preserved, new provenance is added.

**Q: Can I run just the LLM stages?**
A: Yes: `kompend compile *.pdf --start-at-tier resolve` skips re-parsing and chunking.

**Q: How do I get a JSON status report?**
A: `kompend compile *.pdf --list-tiers` outputs JSON.

## Q: Ingest hangs or crashes on large scanned PDFs

Memory pressure on huge scanned PDFs comes from the OCR backend parsecraft
escalates to: its ONNX/VLM models allocate gigabytes during batch inference.
Symptoms include an unresponsive process, `std::bad_alloc`, or extremely slow
ingest.

Kompend hands one document at a time to parsecraft and exposes no per-backend
batch knobs, so the levers are:

1. **Keep ingest sequential** — `kompend ingest --workers 1` (the default) keeps a
   single document in flight so peak memory stays bounded per file.
2. **Isolate the document** — `kompend ingest big-scan.pdf` parses only that file,
   so a failure does not take the rest of the corpus with it.
3. **Trim the candidate set** — uninstalling the unused `parsecraft[ocr]` extra
   removes that backend and its model weights from the routing candidates.

Routing and the OCR memory budget live in parsecraft. If one document still
exhausts memory, the fix belongs there rather than in kompend configuration.
