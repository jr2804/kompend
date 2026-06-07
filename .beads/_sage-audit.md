# sage-wiki Feature Audit for KNOX

Scanned: `#configuration`, `#commands`, and all tier-2 sections of the sage-wiki README.
Filtered through issues you filed (`jr2804` — items #83, #84, #85, #88) to identify broken approaches to **avoid**.

---

## TL;DR — Most Promising Ideas (by adoption confidence)

| Adopt | Idea | Why |
|-------|------|-----|
| ✅ **Now** | `doctor` command — validate config + connectivity | Low effort, high debugging value |
| ✅ **Now** | `status` command — wiki stats, health, tier distribution | Good UX feedback |
| ✅ **Soon** | `write` command — one-shot summary/article from sources | Natural addition to compile |
| ✅ **Soon** | `ingest` command — add source + queue compile | Better UX than `parse` |
| ✅ **Soon** | `provenance` command — source↔article mapping | Critical for trust |
| ✅ **Soon** | `query` command — Q&A against compiled wiki | Natural companion to compile |
| ⚠️ **Design** | Tiered compilation (index-only → full compile) | Maps perfectly to KNOX's 5 tiers |
| ⚠️ **Design** | Output trust / grounding verification | Sophisticates WiCER loop |
| ⚠️ **Design** | Contribution packs (domain ontology + prompts) | Big effort but high value |
| ⚠️ **Design** | External parser subprocess protocol | Parallells BaseParser but for any language |
| ⚠️ **Design** | Custom frontmatter fields from LLM | Clean extension to CompiledWiki model |
| 🔶 **Later** | Graph-enhanced search context (4-signal) | Post-MVP |
| 🔶 **Later** | Hub federation (multi-wiki search) | Post-MVP |
| 🔶 **Later** | Agent skill file generation | Post-MVP |
| 🔶 **Later** | Chunk-level FTS5 + vector indexing | We already chunk; adding search is next |
| ❌ **Skip** | Subscription auth | KNOX is library-first, not SaaS |
| ❌ **Skip** | Web UI / TUI | Not in scope |
| ❌ **Skip** | Batch API for 50% cost reduction | Tied to specific Go provider impl |

---

## 1. Commands to Steal

sage-wiki has **44 commands**; KNOX has 2 (`parse`, `compile`). Several would slot in naturally:

### ✅ Immediate additions (low effort, high value)

**`doctor`** — Validate config and connectivity.
```bash
knox doctor            # Check: config validity, sources dir, providers, deps
```
- Tries a `litellm.completion` ping, checks parser availability, verifies source paths
- Pairs perfectly with the LLM failover work (Q12) — actually *test* the chain
- Sage impl is thin; ours would be a health check against PipelineConfig

**`status`** — Wiki stats and health.
```bash
knox status            # Sources: 23 files (12 PDF, 8 DOCX, 3 MD) | Compiled: 145 articles | Last compile: 2h ago
```
- Reads from compiled wiki metadata (`.knox/` state dir)
- Reports tier distribution if tiered compilation lands
- Sage's version has a nice tabular TUI display; ours can be plaintext

**`provenance`** — Show source↔article traceability.
```bash
knox provenance attention.pdf   # → self-attention, multi-head-attention, scaled-dot-product
knox provenance self-attention  # → sources: attention.pdf, transformer-paper.pdf
```
- Essential for trust: which sources fed which articles
- KNOX's GraphCompiler already tracks this — just expose it
- Sage's version is command-line; could also be an MCP tool

**`query`** — Q&A against compiled wiki.
```bash
knox query "what is flash attention?" --top-k 5
```
- Would use the zvec index (when implemented) + deterministic fallback
- Sage does LLM query expansion + re-ranking + graph context — we can start simpler

**`write`** — Write a summary or article from one or more sources (bypasses full compile).
```bash
knox write summary attention.pdf       # → one-shot summary
knox write article attention.pdf       # → full concept article (like a mini-compile)
```
- Sage distinguishes `summarize` vs `article` vs `paper` prompt templates
- Maps to KNOX's T3 (resolution) + T4 (compilation) but for a single doc
- Useful for quick LLM outputs without running the full pipeline

**`ingest` / `add-source`** — Add a source and optionally queue a compile.
```bash
knox ingest https://arxiv.org/pdf/1234.56789
knox add-source notes/meeting-notes.md
```
- `ingest` = download + register → compile
- `add-source` = register existing local file → compile
- Nicer UX than `knox compile some-file.pdf` for interactive use

**`list`** — List wiki entities, concepts, or sources.
```bash
knox list sources          # files in sources dir
knox list concepts         # entities in compiled ontology
knox list articles         # compiled wiki pages
```

### 🔶 Worth designing, not immediate

**`learn` / `capture`** — Inject knowledge from conversation or text.
```bash
knox learn "The attention mechanism uses QKV projections"
```
- Sage uses these via MCP during AI sessions
- Stores a snippet as a source file → queues compile
- Would make KNOX useful as an "agent memory layer"

**`diff`** — Show pending source changes.
```bash
knox diff                  # new/modified sources since last compile
```
- Based on SHA-256 manifest (sage calls it a "compile manifest")
- Helps with incremental compilation decisions

**`lint`** — Run quality passes on prompts or compiled articles.
```bash
knox lint                  # check prompts for completeness, check compiled articles for issues
```
- Sage has configurable lint passes; we could start with a prompt syntax + completeness check

**`pack`** commands — `install`, `apply`, `remove`, `list`, `search`, `update`, `info`, `create`, `validate`, `conflicts`.
- Too early — defer to after core pipeline is solid

**`verify` / `outputs`** — Output trust system CLI.
- Defer to when we implement grounding verification in WiCER

---

## 2. Configuration Ideas

sage-wiki uses a single `config.yaml`. KNOX currently uses only CLI flags + env vars. A persisted config file is the natural next step.

### ✅ Steal: Full config file with env-var override

```yaml
# .knox/config.yaml (auto-created by `knox init`)
version: 1
project: my-knowledge-base

sources:
  - path: sources
    type: auto
    recursive: true

output: wiki

llm:
  resolve:   ollama/llama3.2
  vlm:       ollama/phi-3-vision
  wicer/probe:  anthropic/claude-haiku-4-5-20251001
  wicer/judge:  anthropic/claude-sonnet-4-20250514
  wicer/refine: anthropic/claude-sonnet-4-20250514

embed:
  model: text-embedding-3-small
  # provider auto-detected from api provider
```

**Why better than flags-only:**
- Repeatable builds
- Shareable between team members
- Supports complex nested config (per-stage llm profile, tier defaults)
- CLI flags override file values (standard pattern)

**Design rule:** A config file should be *optional*. Everything in it must have a flag/env-var equivalent. No `--config-file` flag? Then file at `.knox/config.yaml` is auto-discovered.

### ✅ Steal: `ignore:` patterns

```yaml
# Folders to never read or send to APIs
ignore:
  - drafts/
  - personal/
  - "*.tmp"
```
- Currently KNOX scans `sources/` dir; no way to exclude subpaths
- Valuable for Obsidian vault overlay mode
- Simple: `resolve_files()` already walks dirs — add an ignore filter

### ✅ Steal: Custom frontmatter fields

Sage allows declaring extra LLM-assessed fields in article frontmatter via `compiler.article_fields`. KNOX's `CompiledWiki` model already has per-article metadata — this is a clean extension:

```yaml
compiler:
  article_fields:
    - language
    - domain
```

Then prompts ask for these fields and the LLM response is parsed into YAML frontmatter. Ground-truth fields (concept, aliases, sources, timestamp) come from the pipeline, not the LLM.

### ⚠️ Steal with care: Tiered compilation

Sage's most architecturally significant feature. Maps directly to KNOX's 5-tier pipeline:

| Tier | KNOX equivalent | What happens | Cost |
|------|----------------|-------------|------|
| 0 — Index | T1 (parse) + T5 (output raw) | Parse + store, no LLM | Free |
| 1 — Index + embed | T1 + T2 (chunk) + embed | Parse + chunk + vectorize | ~$0.00002 |
| 2 — Code parse | T1 + T3 (resolution, deterministic) | Structure summary, no LLM | Free |
| 3 — Full compile | T1 + T2 + T3 + T4 + T5 | Full pipeline | ~$0.05-0.15 |

**What sage does right:**
- `default_tier` configurable, per-extension overrides (`tier_defaults`)
- Auto-promote sources to tier 3 based on search frequency (3+ hits)
- Auto-demote stale articles (90d without queries)
- Compile-on-demand: search identifies uncompiled sources, MCP tool compiles just that cluster

**What sage does wrong (your issue #88):**
- Upsert resets all pass flags on restart → progress lost
- Our design: persistent checkpoint state per source (which tiers completed, which WiCER iterations done)

**For KNOX:** This is a major feature, but it aligns perfectly with our architecture. The 5 tiers already exist — we just need to make them individually skippable and add a tracking DB.

### ⚠️ Steal with care: Output trust / grounding verification

Sage quarantines query outputs as "pending" until verified:

1. **Consensus** — same question, same answer from different chunks → confirmation accumulates
2. **Grounding** — `sage-wiki verify` runs LLM entailment checking claims against source passages
3. **Promotion** — when thresholds met, output enters the searchable corpus

**For KNOX:** This maps elegantly onto the WiCER compilation loop (T4). WiCER already has probe → answer → evaluate → refine. The "evaluate" step *is* grounding verification. We could add a post-compile trust gate:

- Compiled articles start as "draft" (present but flagged)
- `knox verify` runs grounding checks using WiCER's LLM-as-judge probes
- Articles promote to "verified" when grounding threshold met
- Consensus from multiple independent compilations (or multiple models via Q12's failover chain) boosts confidence

**What to avoid (from sage issues):**
- Never silently drop the `reasoning` field from provider responses (your #85)

### 🔶 Consider later: External parser subprocess protocol

Sage allows any script as a parser via stdin/stdout with timeout + env stripping:

```yaml
parsers:
  - extensions: [".rtf"]
    command: python3
    args: ["rtf_parser.py"]
    timeout: 30s
```

KNOX already has `BaseParser` and a `ParserRouter`. An `ExternalParser(BaseParser)` that wraps subprocess calls would be a natural addition. The protocol is trivial:
- stdin: file bytes
- stdout: JSON with the same schema as `ParsedDocument`

Security requires double opt-in (`parsers.external + parsers.trust_external`).

### 🔶 Consider later: Contribution packs

Bundles of ontology types + prompts + sample sources for a domain. 8 bundled packs ship with sage. KNOX could ship 3-4 initially:
- `academic-research` — 8 built-in relations + WiCER prompts for research papers
- `software-engineering` — architecture decisions, ADRs
- `personal-knowledge` — note-taking defaults

This is a lot of design work. Defer until core pipeline is solid.

---

## 3. Anti-Patterns to Avoid (from your filed issues)

| Issue | What broke | How to avoid |
|-------|-----------|-------------|
| **#83** | Ollama provider claims BatchProvider support → 404 on `/v1/files` | Only advertise batch/parallel where the provider actually supports it. LiteLLM has `supports_batch` metadata — use it. |
| **#84** | `storage.Open` doesn't create parent dir; `init --config` overwrites existing config | Create parent dirs on open. Use atomic writes for config files. Never overwrite existing config without `--force` or interactive confirmation. |
| **#85** | OpenAI `ParseResponse` ignores `reasoning` field → empty summaries with reasoning models | When using reasoning models (`o3-mini`, `deepseek-r1`), the response has a `reasoning` field before `content`. Our litellm wrapper must handle: `content = response.choices[0].message.content or response.choices[0].message.reasoning` or similar. This affects Q12's client design. |
| **#88** | Upsert resets all pass flags on restart → all progress lost from previous run | Incremental state must persist across restarts. Track per-source completion in a local DB (SQLite). Never "start over" by default — only on explicit `--force` or when source hash changes. |

The deeper pattern from #88 is: **incremental compilation must be idempotent and checkpointed.** The compile manifest tracks which sources were processed through which pipeline tiers and WiCER iterations. On restart, only incomplete/pending sources are re-run. This is exactly what KNOX's `input_files` list + SHA-256 cache (issue `knox-fiu`, Q11) should feed into.

---

## 4. Search & Retrieval Ideas

Not immediately actionable but worth noting for post-MVP:

- **Chunk-level indexing** — Each chunk gets its own FTS5 + vector entry, not just the full article. KNOX's `HybridChunker` already produces chunks; indexing them individually is the next step.
- **LLM query expansion** — Generate keyword rewrites, semantic rewrites, and a hypothetical answer to improve BM25 + vector retrieval.
- **Graph-enhanced context** — Sage's 4-signal graph scorer (direct relation ×3.0, shared source ×4.0, common neighbor ×1.5, type affinity ×1.0) for Q&A context assembly. KNOX's GraphCompiler builds the relation graph — same data, just query it at search time.
- **LLM re-ranking with position-aware blending** — Score top-N candidates with the LLM, but protect high-confidence BM25 results (ranks 1-3 get weight ×0.75 from retrieval, ranks 11+ get ×0.60 from reranker).
- **Cross-lingual vector search** — Brute-force cosine across all chunk vectors + BM25 via RRF fusion. Zero lexical overlap? No problem if semantic vectors match.

---

## Summary: Proposed New Issues

Building on Q12 (failover) and Q13 (per-task profiles), I'd suggest creating issues for:

| Priority | Feature | Depends on | Rationale |
|----------|---------|-----------|-----------|
| P2 | **Q14: `doctor` + `status` commands** | — | Quick wins, high debugging value |
| P2 | **Q15: Persisted config file `.knox/config.yaml`** | Q13 | Unlocks all complex config; flags always override |
| P2 | **Q16: `provenance` command** | — | Surface existing GraphCompiler tracking |
| P2 | **Q17: `query` command against compiled wiki** | Q11 (zvec) | First "read" use of the compiled wiki |
| P2 | **Q18: Custom frontmatter fields + `write` command** | — | Extend CompiledWiki model for LLM-assessed fields |
| P2 | **Q19: `ignore` patterns for sources dir** | — | Needed for vault-overlay mode |
| P2 | **Q20: Tiered compilation (skippable tiers)** | Q11 (cache) | Major scaling feature, maps to our 5 tiers |
| P3 | **Q21: Output trust / grounding gates** | Q20, Q12 | Sophisticates WiCER, prevents hallucination propagation |
| P3 | **Q22: `learn`/`capture` — knowledge injection** | MCP server | Agent memory layer |
| P3 | **Q23: External parser subprocess protocol** | — | Extends `BaseParser` for any-language parser |
| P4 | **Q24: Contribution packs** | Q15 | Domain-specific ontology + prompt bundles |
| P4 | **Q25: Search quality improvements** | Q17 | Chunk indexing, query expansion, graph context |

Anti-patterns to enforce during implementation:
- Always create parent dirs (anti-#84)
- Never overwrite config without `--force` (anti-#84)
- Handle `reasoning` field in LLM provider responses (anti-#85)
- Persistent per-source completion state, never full reset (anti-#88)
- Only use batch/parallel when provider metadata confirms support (anti-#83)
