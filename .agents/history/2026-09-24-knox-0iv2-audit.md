# knox-0iv2 epic audit — agentic multi-turn workflows

Created: 2026-09-24 · Completed: 2026-09-24 · Status: **resolved** — epic closed,
all verdicts acted on. Retained as the record of how a tracker/reality contradiction
was resolved from code rather than from tracker state.

Read-only code audit. Ground truth from code, not the tracker. bd shows 13 open
children; this resolves the "user believes done, bd says open" contradiction.

Method: `grep -rn "tools=" src/knox/agents/*.py`; per-class dynamic-access sweep
(`grep -rn "<Class>" src/` incl. `.j2`/`.toml`, checked for getattr/dispatch);
`git log -S "<Class>"` to prove a construction site ever existed; read each
agent's factory + its single non-test caller chain.

## Summary

| # | Child | Verdict | Notes |
|---|-------|---------|-------|
| .1 | Agentic RemediationAgent | **COMPLETE** (closed) | tools+harness, reachable |
| .2 | Agentic WikiIndexAgent (overview) | **COMPLETE** (closed) | tools+harness, reachable |
| .15 | AgenticHarness helper | **COMPLETE** (closed) | return type now `Sequence[AbstractCapability]` (knox-3 fix) |
| .3 | Agentic VerificationAgent | **COMPLETE** (bd OPEN) | close |
| .4 | Agentic WriteAgent | **COMPLETE** (bd OPEN) | close; tool-set diverges from plan |
| .5 | Agentic ContextChunkAgent | **HOLLOW** | class+tools+prompt, no factory, test-only caller |
| .6 | Agentic SummarizerAgent | **HOLLOW** | class+tools+prompt, `NodeSummarizer` built without agent |
| .7 | Agentic ExtractionAgent (consolidate w/ reduce) | **PARTIAL / re-scoped** | dedicated class gone; AGENTIC strategy uses `ExtractionAgent`+memory+tools |
| .8 | Agentic CategorizeAgent | **NOT IMPLEMENTED** (deleted) | no class remains |
| .9 | Agentic CaptureAgent + DistillAgent | **PARTIAL** | distill DONE+fixed; capture half never existed |
| .10 | Agentic QueryAgent | **HOLLOW/GHOST** | class+tools+prompt, no factory, `query_agent=` never passed |
| .11 | ReferenceExtraction + Resolver | **COMPLETE, not agentic** | both live; no tools (single-shot) |
| .12 | Remove legacy GenerationAgent | **NOT DONE (narrowed)** | legacy class + dead `generate_all_pages()` still present |
| .13 | Agentic LintRuleAgent | **NOT IMPLEMENTED** | no agentic variant, no lint tools |
| .14 | Document single-shot-by-design | **UNVERIFIABLE from code** | docs task |

Net: **5 genuinely done** (`.1 .2 .3 .4 .15`), **2 hollow/ghost** (`.5 .6 .10`),
**3 not implemented or dropped** (`.8 .13`, capture half of `.9`), **3 re-scoped
by a different design** (`.7 .11`, and `.12` narrowed to dead-code removal),
**1 docs** (`.14`).

## Per-child evidence

### knox-0iv2.1 — Agentic RemediationAgent — COMPLETE

- Impl: `agents/remediate.py:37` `AgenticRemediationAgent`; `compilation/remediate_tools.py:26` `build_remediation_tools()` → `fetch_gaps`, `save_page`, `mark_unfixable`.
- Wired: `agents/remediate.py:50` `make_agentic_remediation_agent` passes `tools=build_remediation_tools()` + `extra_capabilities=harness.build_capabilities()` (`:78`/`:79`).
- Reachable: `cli/cmds/quality/lint.py:515,575`; `mcp/tools/quality.py:236`.
- Prompt↔tools: `templates/lint/remediate_agentic_system.md.j2` uses `fetch_gaps/save_page/mark_unfixable` — exact match.
- Rec: keep closed.

### knox-0iv2.2 — Agentic WikiIndexAgent (overview) — COMPLETE

- Impl: `agents/wiki_index.py:38` `make_agentic_overview_agent`; `overview_tools.py:23` → `list_pages`, `read_page`, `save_overview`.
- Wired: `wiki_index.py:63` passes `tools=build_overview_tools()`; harness caps also present.
- Reachable: `cli/_post_compile.py:321` inside `run_overview_synthesis` (gated on `--synthesize`).
- Prompt↔tools: `templates/output/synthesize_agentic_system.md.j2` — exact match.
- Rec: keep closed.

### knox-0iv2.15 — AgenticHarness — COMPLETE

- `agents/agentic_harness.py:31`; `build_capabilities()` now annotated `-> Sequence[AbstractCapability]` (was `list[AbstractCapability]` forcing non-functional `type: ignore` at call sites). No `type: ignore` remain in `agents/`. FIXED by knox-3.
- Rec: keep closed.

### knox-0iv2.3 — Agentic VerificationAgent — IMPLEMENTED, bd OPEN → CLOSE

- Impl: `agents/verification.py:25` `AgenticVerificationAgent`; `verification_tools.py:26` → `get_claim`, `search_sources`, `read_page`, `mark_verified`, `mark_contradicted`, `mark_unverifiable`.
- Wired: `verification.py:41` factory passes `tools=build_verification_tools()` + harness caps (`:53`/`:58`/`:66`).
- Reachable: `cli/_post_compile.py:92` → `verify_claims` → `knox.verification.verify_wiki` (`:167`). Only production call site; no per-claim default path remains (`verify_wiki` callers = this one).
- Prompt↔tools: `templates/lint/verify_agentic_system.md.j2` — exact match.
- **Why bd says OPEN:** implemented (2026-08-28 family) but never closed; the flow it replaced is gone. Rec: close.

### knox-0iv2.4 — Agentic WriteAgent — IMPLEMENTED, bd OPEN → CLOSE

- Impl: `agents/write.py` `AgenticWriteAgent`; `write_tools.py:23` `build_write_tools()` → `read_source`, `list_sources`, `read_related`, `save_page`.
- Wired: `write.py:46` factory → `tools=build_write_tools()` (`:66`) + harness caps.
- Reachable: `cli/cmds/knowledge/capture.py:357` (`write` command, registered `cli/app.py:53`); MCP write tool mirrors it.
- Prompt↔tools: `templates/capture/write_agentic_system.md.j2` matches the *implemented* tools.
- **Divergence:** bd plan specified `get_source`/`get_existing_page`/`save_page`; shipped tools differ (no existing-page read). Functionally complete for its own prompt. Rec: close, note the plan divergence.

### knox-0iv2.5 — Agentic ContextChunkAgent — HOLLOW

- Impl: `agents/context_chunk_agentic.py:11`; `context_chunk_tools.py:21` → `next_chunk`, `situate_chunk`; `templates/chunking/contextual_agentic_system.md.j2` matches.
- **No factory** (`git log -S` shows the shim dropped in `d443c583`, 2026-09-11; commit `059eefa6` 0iv2.5 added the class).
- Reachability: the only consumer is `chunking/contextual.py` `ContextualEnricher` (`:45/:74/:96`), which is referenced **only** in `tests/unit/test_contextual_chunker.py` — never constructed in `src/`. Agentic branch is dead.
- Rec: re-scope — either wire into `chunking/hierarchical.py` (as the child intended) or delete the class+tools+prompt as dead.

### knox-0iv2.6 — Agentic SummarizerAgent — HOLLOW

- Impl: `agents/summarizer_agentic.py`; `summarizer_tools.py:21`. Prompt `chunking/summarizer_agentic_system.md.j2`.
- **No factory** (shim dropped `d443c583`; class added `5ffeb874`).
- Reachability: `chunking/summarizer.py:91` gates on `isinstance(self._agent, AgenticSummarizerAgent)`, but production builds `NodeSummarizer(...)` at `core/pipeline/stages/_factory.py:133` **without `agent=`** → `self._agent is None` → agentic branch never taken.
- Prompt↔tools: tools exist (`get_chunk`-style) but are never loaded.
- Rec: re-scope/delete. Same class as .5: a complete agent with no construction site.

### knox-0iv2.7 — Agentic ExtractionAgent (consolidate with reduce) — PARTIAL / re-scoped

- No `AgenticExtractionAgent` class remains (`extraction_agentic.py`/`extraction_tools.py` deleted). Live "agentic" path is `compilation/flows/agentic.py:extract_agentic`, dispatched from `flows/extraction.py:133` on `ExtractionStrategy.AGENTIC`.
- That path uses the **ordinary** `ExtractionAgent` (`flows/agentic.py:98`) with a rolling memory digest and optional asset tools (`build_asset_tools` → `get_table`/`get_figure_description`, added only when the asset store is non-empty).
- The nested per-concept loop the child describes is the `INCREMENTAL` path, still present.
- Reachable via config `compile.llm.extraction_strategy=agentic`.
- Rec: **re-scope** — the child's goal (fewer calls, cross-chunk context) is largely met by the memory-digest design, not by a dedicated tool agent. Rewrite the issue around "does AGENTIC replace INCREMENTAL", or close.

### knox-0iv2.8 — Agentic CategorizeAgent — NOT IMPLEMENTED (dropped)

- No `AgenticCategorizeAgent`; `categorize_agentic.py` + `categorize_tools.py` deleted. Live `agents/categorize.py:12 CategorizeAgent` is single-shot.
- Rec: close as won't-do, or reopen with a concrete reason.

### knox-0iv2.9 — Agentic CaptureAgent + DistillAgent — PARTIAL

- **Distill half: DONE.** `agents/distill_agentic.py:20`, factory `:28` (`tools=build_distill_tools()` `:53` + harness caps). `distill_tools.py:22` → `get_group`, `save_page`, matching `templates/capture/distill_agentic_system.md.j2`. Reachable: `cli/cmds/knowledge/capture.py:206`, `mcp/tools/knowledge.py:214`. (Was HOLLOW — no tools/caps — fixed by knox-3 this session.)
- **Capture half: NOT IMPLEMENTED.** No `AgenticCaptureAgent` exists; `agents/capture.py:12 CaptureAgent` is single-shot, used by `capture/capture.py:79 _extract_with_agent`. The planned agentic capture was never built (the `capture_agentic.py` shim was dropped in `d443c583`).
- Rec: split — close the distill child; drop or re-scope the capture child.

### knox-0iv2.10 — Agentic QueryAgent — HOLLOW/GHOST

- Impl: `agents/query_agentic.py:16`; `query_tools.py:22` → `search_wiki`, `read_page`, `add_finding`; `templates/search/query_agentic_system.md.j2` matches.
- **No factory and no wiring:** `synthesize_results(..., query_agent=QueryAgent|AgenticQueryAgent|None)` (`search/retriever.py:303`) — grep shows **no caller ever passes `query_agent=`** (`cli/_query_helpers.py:269`, `search/pipeline.py:153`, `search/retriever.py:112` all omit it). `build_query_tools()` has **zero callers**. Commit `6c1988c0` wired the *param* but never a construction site.
- Rec: re-scope/delete — same hollow pattern.

### knox-0iv2.11 — ReferenceExtractionAgent + ReferenceResolverAgent — COMPLETE, but not agentic

- Both implemented and reachable: `flows/references.py:84` builds `ReferenceExtractionAgent`; `cli/cmds/knowledge/references.py:179` builds `ReferenceResolverAgent`. Prompts exist.
- **No tools** for either → they are single-shot, so the child's "multi-turn agent" plan is unrealised; the extraction/resolver *work* is done.
- Rec: close the extraction/resolver scope; open a new issue only if agentic behaviour is still wanted.

### knox-0iv2.12 — Remove legacy GenerationAgent — NOT DONE (narrowed)

- `agents/generation.py:45 GenerationAgent` still exists; `compilation/flows/generation.py:453` (in `generate_all_pages`, the legacy loop) still constructs it.
- BUT `generate_all_pages` is **dead**: its only reference anywhere is docstrings; the live entry `run_generation` → `run_generation_with_checkpoint` (`:204`) → `generate_all_pages_agentic` (`:385`) → `make_agentic_generation_agent` (`:693`). Dynamic-access sweep found no getattr/dispatch.
- So the issue is valid but **narrowed to dead-code removal**: delete `GenerationAgent` + `generate_all_pages()` + `build_gen_tools()`. Rec: keep open, re-scope to cleanup.

### knox-0iv2.13 — Agentic LintRuleAgent — NOT IMPLEMENTED

- No agentic lint variant; no `lint_tools.py`. Only `agents/lint.py:17 LintRuleAgent` (single-shot) and the separate `AgenticRemediationAgent` (0iv2.1). `templates/lint/` has `remediate_agentic_*` and `verify_agentic_*` but no lint-rule agentic prompt.
- Rec: keep open (if still wanted) or drop.

### knox-0iv2.14 — Document single-shot-by-design — UNVERIFIABLE from code

- Docs exist (`docs/architecture/decisions.md`, `docs/concepts/basic-principles.md`, etc.) but no doc enumerates which agents intentionally stay single-shot. This is a human/docs-completeness call, not resolvable by code audit.
- Rec: keep open; needs a docs review, not a code change.

## What to do with the epic

- **Close now:** `.3`, `.4` (implemented+wired+reachable; bd lag). Optionally `.1`, `.2`, `.15` stay closed.
- **Re-scope:** `.5`, `.6`, `.10` → "wire or delete"; `.7` → "AGENTIC vs INCREMENTAL"; `.11` → extraction/resolver done, agentic optional; `.12` → dead-code removal (`GenerationAgent` + `generate_all_pages`).
- **Close as won't-do / split:** `.8`; `.9` (close distill, drop capture); `.13`.
- **Human/docs:** `.14`.

**Premise correction:** the epic title ("Migrate *all* LLM-based actions to agentic
multi-turn workflows") is only partly realised. 5 agents are genuinely agentic
and live (.1–.4, distill); 3 were built but never wired (.5, .6, .10 — the
`d443c583` "drop unused factory shims" is where the wiring was severed and never
restored); 3 were deliberately dropped/deleted (.8, .13, capture); and 2 areas
were satisfied by a *different* design rather than an agentic agent (.7 memory
digest, .11 single-shot). The tracker's "everything implemented" belief holds
for the agents that were actually wired (`.3`, `.4`); the hollow ones would have
looked "done" only because the class+tools+prompt files exist.
