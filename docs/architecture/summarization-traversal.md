# Summarization Traversal

Big trees are mostly scaffolding: headings, boilerplate, and appendices between
the branches the task actually needs. Traversal planning puts a cheap
decision-model pass in front of the summarizer so worker calls are spent on
planned branches only. It is off by default; the off path is byte-identical.

## The three passes

1. **Outline** — `kompend.chunking.outline.build_outline` walks the
   `ChunkTree` bottom-up and emits one `OutlineNode` per node: id, heading
   chain, depth, child count, payload and subtree character counts, and a
   short preview. No LLM is involved; sizes come from the shared
   `conservative_tokens` counter. See [Chunk Tree](chunk-tree.md).
2. **Per-level plan** — starting at the root, the planner
   (`kompend.agents.traversal.decide_traversal_plan`) decides each level's
   children top-down: `EXPAND` (work plus descend), `SUMMARIZE` (summarize
   this branch with a focus line, do not descend), or `SKIP` with a fixed
   reason (boilerplate, appendix, repeated, other). One decision-model call
   per expanded level; a level wider than 64 children is asked in windows.
   A child missing from its level's plan counts as `EXPAND`.
3. **Executor** — `NodeSummarizer` with `traversal=AUTO` and a planner
   attached summarizes only the planned set bottom-up
   (`_summarize_planned`). SKIP nodes leave the worklist; once their nearest
   ancestor is summarized they inherit its summary, and a skipped node whose
   whole ancestor chain stayed summary-less is summarized normally. The root
   is unconditional work.

## Cost model

The reference measurement is `sources/26253-j30.docx`: 26,886 tree nodes,
942 of them summary-eligible, 28.8 M characters of payload.

- **Off path:** one worker summary per eligible node — ~942 worker calls for
  the reference document, each carrying its payload windows.
- **Traversal path:** one planner call per expanded level (plus one per extra
  64-child window), then worker calls for the planned nodes only. A level
  that resolves to SKIP everywhere below it costs one planner call and zero
  worker calls.
- **Planner + worker vs navigating session:** the agentic summarizer visits
  every eligible node through batched sessions; traversal adds a small number
  of decision-model calls to prune whole subtrees before the sessions run.
  The win scales with the share of skippable structure — boilerplate-heavy
  documents gain the most; a tree where everything is task-relevant pays the
  planner calls and keeps the worker calls.

The planner's own budgets (`budget_nodes`, `budget_windows`) shrink
window-to-window, so a merged plan can never exceed the level budget; an
EXPAND past the remaining budget is sent back for another verdict. Plan
reuse is a cache hit on the `summarizer_plan` layer (keyed by tree, task,
model, and policy — see [the cache architecture](cache.md) for where the
layers live).

## Configuration

Five keys under `[compile.budget]` (`KOMP_BUDGET_SUMMARY_TRAVERSAL*` in the
environment), documented in
[Model Context Adaptation](../guides/model-context.md#traversal-planning):
the `summary_traversal` switch (`off` by default), node/window ceilings, the
`min_nodes` gate below which planning is pure overhead, and the optional
`summary_traversal_model` override that falls back to the decision model.

## Invariants

- **Never crop.** Overlong content is windowed into further calls; windows
  that do not fit stay pending for the next session. A budget splits a
  request, it never drops a window.
- **No silent skip.** A missing plan entry means `EXPAND`; every SKIP carries
  its fixed reason in the debug log; holes left by summary-less ancestors
  are summarized normally.
- **Task-conditioned cache key.** The plan key hashes the tree identity plus
  the task statement and frontmatter — a plan from another task never
  matches. The off-path summary deps are pinned byte-for-byte so enabling
  traversal cannot invalidate existing vault caches.
- **Fail open to the full tree.** A declined level, a failed plan, or a
  disabled decision model logs a warning and summarizes the whole tree.
- **Off switch.** `summary_traversal = "off"` (the default) keeps today's
  path exactly; observability is one ledger row per tree (`summarize`,
  `ok`/`partial`, `reason_code=traversal`, `planned/skipped/expanded/windows`
  counters) with the full plan JSON in the cache layer, never in the row.
