# Credits & Acknowledgements

KNOX builds on ideas from the broader LLM Wiki ecosystem. The following
open-source projects were investigated for patterns, approaches, and design
decisions. Many thanks to their creators and contributors.

## Investigated Projects

| Project | Repository | License | Key Ideas |
|---------|-----------|---------|-----------|
| **OpenKB** by VectifyAI | [VectifyAI/OpenKB](https://github.com/VectifyAI/OpenKB) | Apache 2.0 | Skill Factory, PageIndex vectorless retrieval, AGENTS.md as runtime config |
| **llmwiki** by lucasastorian | [lucasastorian/llmwiki](https://github.com/lucasastorian/llmwiki) | Apache 2.0 | MCP-per-workspace, `guide` tool, filesystem as source of truth |
| **llm-wiki-compiler** by atomicstrata | [atomicstrata/llm-wiki-compiler](https://github.com/atomicstrata/llm-wiki-compiler) | MIT | Two-phase compile, eval harness, claim-level provenance, review queue, schema layer |
| **llm-wiki** by Pratiyush | [Pratiyush/llm-wiki](https://github.com/Pratiyush/llm-wiki) | MIT | Session transcript ingestion, 16 lint rules, 5-state lifecycle, 8+ agent adapters |
| **llm-wiki** by NiharShrotri | [NiharShrotri/llm-wiki](https://github.com/NiharShrotri/llm-wiki) | MIT | Web UI, 3-way query scope, 3-pass ingest, auto-fix lint |
| **qmd** by tobi | [tobi/qmd](https://github.com/tobi/qmd) | MIT | On-device hybrid search engine (BM25+vector+rerank), SDK, hierarchical context metadata |
| **notebooklm-llm-wiki-flow** by reallygood83 | [reallygood83/notebooklm-llm-wiki-flow](https://github.com/reallygood83/notebooklm-llm-wiki-flow) | MIT | NotebookLM→LLM Wiki pipeline, YAML workflows, typed error handling, staged writes |
| **llm-wiki-pm** by anh-chu | [anh-chu/llm-wiki-pm](https://github.com/anh-chu/llm-wiki-pm) | MIT | PM-focused skill, role packs, worker subagents, supersession, crystallize pattern |
| **WeKnora** by Tencent | [Tencent/WeKnora](https://github.com/Tencent/WeKnora) | MIT | Enterprise wiki mode, ReACT agent orchestration, 20+ LLM providers, RBAC, multi-source sync |
| **sage-wiki** by xoai | [xoai/sage-wiki](https://github.com/xoai/sage-wiki) | MIT | Configuration design, per-task model selection, output trust, contribution packs |
| **Karpathy's LLM Wiki** | [gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) | — | Foundational concept: LLM-maintained, self-compiling wiki from raw sources |

## Per-Project Deep Dives

### atomicstrata/llm-wiki-compiler

| Idea | Description | KNOX Status |
|------|------------|-------------|
| **Two-phase compile** | Extract all entities/concepts first, then generate pages. Enables cross-source merging and order-independence. | **Planned (Q20/N3).** Core architecture decision. |
| **Eval harness** | Production-grade health score: citation coverage, freshness, link integrity, style guide conformance. CI thresholds. | **Planned (N2).** Quality measurement from the start. |
| **Claim-level provenance** | Line-range citations `^[source.md:42-58]` embedded in every claim. Lint validates citations. | **Planned (Q16/N4).** Trust foundation. |
| **Review queue** | `compile --review` stages changes for human approval. Approve/reject/lock per page. | **Planned (N8).** Human-in-the-loop. |
| **Schema layer** | Page kinds (concept, entity, comparison, overview) with per-kind compile rules and templates. | **Planned (N9).** Structured compilation. |
| **Per-concept prompt budget** | Fair-share truncation at 200K chars. Prevents one topic from starving others in context window. | **Planned (N10).** Scale safeguard. |
| **Query write-back (`--save`)** | Persist query answers as new wiki pages. Knowledge compounds with every question. | **Planned (Q17 enhancement).** Same pattern as NiharShrotri's `--save-as`. |

### NiharShrotri/llm-wiki

| Idea | Description | KNOX Status |
|------|------------|-------------|
| **Web UI** | Full dashboard with drag-and-drop upload, live progress, D3 force-directed graph. Survives tab close and server restart. | **Planned (N5).** Significant but deferrable. |
| **3-way query scope** | `Wiki` (thematic from compiled pages), `Raw` (exact lookups in original docs), `Hybrid` (both combined via RRF). | **Planned (N6).** Essential for query flexibility. |
| **Intent classification** | Casual messages ("hi", "thanks") skip retrieval entirely, saving ~30s per interaction. | **Planned (N7).** Optimization for interactive mode. |
| **Write-back (`--save-as`)** | Persist query answers as new `synthesis/` pages. | **Planned (Q17 enhancement).** Same pattern as atomicstrata's `--save`. |
| **Auto-fix on lint** | `wiki lint --fix` resolves most stylistic issues automatically. | **Planned (N1 enhancement).** Natural extension once lint exists. |
| **3-pass ingest pipeline** | 1) Extraction (thinking mode, JSON structured output) → 2) Page drafting (streaming) → 3) Source summary (provenance tracking). | **Planned (Q20 refinement).** Structured extraction before generation. |
| **Smart merging** | Re-ingesting a related source updates existing pages instead of overwriting. Preserves provenance. | **Planned (Q20 enhancement).** Builds on two-phase compile. |

### tobi/qmd

| Idea | Description | KNOX Status |
|------|------------|-------------|
| **MCP server with typed tools** | Exposes `query`, `get`, `multi_get`, `status` as MCP tools with structured input/output schemas. | **Reference for knox-4jc.** Clean tool design pattern. |
| **Hierarchical context metadata** | Collections/documents carry context tags returned alongside results. LLMs make better selection decisions. | **Planned (Q17 enhancement).** qmd's key differentiator. |
| **Typed sub-queries** | `lex` (BM25), `vec` (vector), `hyde` (hypothetical answer) as separate query types, combined via RRF + reranking. | **Planned (Q25).** Cleaner than a single hybrid weight. |
| **Progress callbacks** | `onProgress` for indexing and embedding operations. | **Planned (Q25 enhancement).** UX improvement for long ops. |
| **Document body with line ranges** | `getDocumentBody(path, {fromLine, maxLines})` — retrieve specific lines. | **Related to N4.** Line-level retrieval for provenance. |
| **All-local via node-llama-cpp** | BM25 + vector + reranker all run locally. No cloud. | **KNOX already local-first.** Validates the approach. |

### Tencent/WeKnora

| Idea | Description | KNOX Status |
|------|------------|-------------|
| **Wiki Mode** | Agent-driven auto-generation of structured, interlinked markdown wiki from raw documents. Same core concept as KNOX. | **Already implemented.** Validates the approach. |
| **20+ LLM providers** | Including Ollama, plus per-model thinking-mode config. | **Planned (Q12+Q13).** Reference for provider matrix design. |
| **Multiple retrieval strategies** | BM25 sparse / Dense / GraphRAG / parent-child / multi-dimensional indexing — selectable per KB. | **Planned (Q25).** Configurable retrieval is more flexible than single hybrid. |
| **Parent-child chunking** | Hierarchical chunking (parent chunks with child passages). Improves retrieval context. | **Planned (Q25 enhancement).** Known RAG improvement. |
| **CLI `--json` switch** | Machine-readable output for all commands. `gh`-style `<noun> <verb>` convention. | **Adopted for all commands.** Cross-project pattern. |
| **Built-in MCP service** | Exposes knowledge retrieval tools via stdio/SSE/HTTP. | **Already planned (knox-4jc).** |
| **Langfuse observability** | Full pipeline tracing: ReAct loops, token tracking, tool calls. | **Under consideration.** Valuable for production use. |

### Pratiyush/llm-wiki

| Idea | Description | KNOX Status |
|------|------------|-------------|
| **Session transcript ingestion** | Structured ingestion of Claude Code .jsonl, Codex CLI, Cursor session transcripts. Full adapter pattern with format detection. | **Not yet planned.** Richer than text snippets; captures full agent conversations. |
| **16 lint rules** | 8 structural (broken links, missing metadata, orphan pages) + 8 LLM-powered (factual consistency, clarity, coherence). | **Planned (N1).** Reference for rule design. |
| **5-state lifecycle** | draft → reviewed → verified → stale → archived. 90-day auto-stale timer. Ebbinghaus-inspired confidence decay. | **Planned (Q21 enhancement).** Improves trust model. |
| **8+ agent adapters** | Claude Code, Cursor, Windsurf, Codex CLI, Continue.dev, GitHub Copilot, Cline, Aider. | **Reference for MCP design (knox-4jc).** Broad adapter strategy. |

### notebooklm-llm-wiki-flow (reallygood83)

| Idea | Description | KNOX Status |
|------|------------|-------------|
| **NotebookLM integration** | Bridges NotebookLM output to LLM Wiki pipeline. Feed URLs/PDFs → NotebookLM → LLM Wiki → Obsidian. | **Under consideration.** Specific workflow bridge. |
| **YAML workflow execution** | `run-from-yaml WORKFLOW.yaml` for repeatable, composable research workflows. | **Under consideration.** Useful for batch/scripted compilation. |
| **Typed error handling** | Protocol-based boundaries, typed runner errors, typed flow phases. No bare `except` anywhere. | **Planned (Q12 enhancement).** TransientError / PermanentError / ConfigurationError hierarchy. |
| **Staged writes + manifest.json** | Per-run manifest for audit trail. Writes go through stages (staging → commit). | **Related to N8 (review queue).** Prevents partial corruption. |
| **Template-based rendering (Jinja2)** | Jinja2 templates for entity pages. Separates content from presentation. | **Planned (Q24 enhancement).** Makes customization accessible. |
| **Config precedence chain** | Environment variables > project.yaml > .env > defaults. Clear, documented priority. | **Adopted for Q15.** Same chain design. |

### llm-wiki-pm (anh-chu)

| Idea | Description | KNOX Status |
|------|------------|-------------|
| **Role packs** | PM, researcher, executive, founder personas that tune recall sensitivity, output format, and templates. | **Planned (Q24 enhancement).** Adds user-persona dimension to contribution packs. |
| **Supersession** | When a page is superseded, inbound wikilinks auto-redirect to the new page. | **Planned (Q21 enhancement).** Trust + link management. |
| **Crystallize pattern** | Meeting transcripts become structured decision digests (decisions, rationale, action items). | **Planned (Q22 enhancement).** Specific extraction template for knowledge capture. |
| **Privacy/redaction** | Pre-ingest filter strips sensitive content. `private:` frontmatter flag excludes pages from search. | **Planned (Q21 enhancement).** Enterprise readiness. |
| **Sub-skills** | Optional installable sub-skills (brief, PRD, research, CRM) that extend the base skill. | **Planned (Q24 enhancement).** Composable pack distribution. |
| **SCHEMA.md** | Governs wiki structure: domain statement, tag taxonomy, page thresholds. Read by agent at startup. | **Related to N9 (schema).** Markdown-based schema definition. |

## Common Architectural Patterns

Across all 11 investigated projects, the following patterns emerge:

1. **Local-first with optional cloud** — Every project runs locally. Cloud is additive, never required.
2. **Filesystem as source of truth** — SQLite/JSON indices are derived, always rebuildable from sources.
3. **Markdown + `[[wikilinks]]`** — Universal interchange format. Obsidian-compatible.
4. **LLM-managed wiki** — LLM writes pages; user curates sources and asks questions.
5. **Incremental compilation** — Hash-based change detection; only changed work runs through LLM.
6. **Hybrid search (BM25 + vector)** — No project relies on vector search alone. Always paired with keyword.
7. **Three-layer architecture** — `raw/` (immutable sources) → LLM processing → `wiki/` (generated pages).
8. **MCP server** — 7/11 projects expose MCP tools for agent integration.
9. **Obsidian compatibility** — 6/11 projects are designed to work as Obsidian vaults.
10. **No vector DB dependency** — 5/11 use embedded vectors (SQLite, JSON files, or local GGUF models), not external vector databases.

## Design Decisions Informed by This Research

### Adopted (in planned features)

| Decision | Source | KNOX Issue |
|----------|--------|-----------|
| Per-task LLM models (not one model for all stages) | sage-wiki, WeKnora | Q13 |
| Config file with env-var/flags override chain | sage-wiki, notebooklm-flow | Q15 |
| `ignore:` patterns for source scanning | sage-wiki, every project | Q19 |
| Two-phase compile (extract → generate) | atomicstrata | Q20 |
| Claim-level provenance: `^[source.md:42-58]` | atomicstrata | Q16 |
| Review queue: `compile --review` → approve/reject | atomicstrata | N8 |
| Eval harness with health score + regression tracking | atomicstrata | N2 |
| 3-way query scope (Wiki / Raw / Hybrid) | NiharShrotri | Q17 enhancement |
| Query write-back (`--save-as`) | atomicstrata, NiharShrotri | Q17 enhancement |
| Typed sub-queries (lex/vec/hyde) + RRF | qmd | Q25 |
| External parser via subprocess protocol | sage-wiki | Q23 |
| External parser via HTTP URL (CONVERTER_URL) | lucasastorian | Q23 enhancement |
| Contribution packs | sage-wiki, OpenKB, anh-chu | Q24 |
| Role packs (persona-tuned output) | anh-chu | Q24 enhancement |
| Template-based rendering (Jinja2) | notebooklm-flow | Q24 enhancement |
| Crystallize pattern (transcripts → decisions) | anh-chu | Q22 enhancement |
| Privacy/redaction: `private:` frontmatter | anh-chu | Q21 enhancement |
| `--json` flag for all CLI commands | WeKnora | All commands |
| Web UI with dashboard + graph + drag-and-drop | NiharShrotri, WeKnora | N5 |

### Under Consideration (not yet committed)

| Idea | Source | Notes |
|------|--------|-------|
| Skill Factory (distill wiki → agent skill) | OpenKB | Complements Q24 packs; different purpose (output-side vs input-side) |
| NotebookLM bridge pipeline | notebooklm-flow | Specific integration; low priority |
| YAML workflow engine | notebooklm-flow | Useful for scripted batch compilation |
| Supersession with auto-redirect | anh-chu | Depends on link management |
| Worker subagents | anh-chu | KNOX isn't agent-based, but pattern is interesting |
| Multi-source auto-sync (Notion, Feishu, etc.) | WeKnora | Enterprise feature; low priority |
| Parent-child chunking | WeKnora | Improves retrieval context; medium priority |
| Multi-LLM observability (Langfuse-style) | WeKnora | Pipeline tracing; medium priority |
| Session transcript ingestion | Pratiyush | Richer than text snippets; captures full agent conversations |
| 5-state lifecycle (draft→reviewed→verified→stale→archived) | Pratiyush | Improves trust model beyond binary trusted/untrusted |

### Actively Avoided

| Anti-pattern | Source | Why |
|-------------|--------|-----|
| Fake batch provider support | sage-wiki (#83) | Only advertise batch when provider supports it |
| Missing parent directory creation | sage-wiki (#84) | Always create parent dirs on open |
| Ignoring `reasoning` field in provider responses | sage-wiki (#85) | Handle reasoning in all provider parsers |
| Full reset on restart (no incremental state) | sage-wiki (#88) | Persistent per-source completion in SQLite |
| Reliance on external vector DB | 4/9 projects | Embedded vectors are simpler and reliable |
| SaaS-first architecture | WeKnora | KNOX is local-first, library-first |

---

*Analysis conducted 2026-06-07 across 11 projects. Full multi-repo audit available in beads issue history.*
