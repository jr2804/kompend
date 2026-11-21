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
- **Planner + worker vs navigating session:** measured (bead `knox-u84v.6`,
  harness in `tools/spike_traversal/`). On a synthetic 1,191-node corpus
  with the same model, task statement and a 60-node budget, a navigating
  session (`list_children` / `browse` / `summarize_node` / `skip`) spent
  **107 requests** against the planner + worker's **1,001** — 89 % fewer —
  and the margin survives a change in how the shared budget is enforced
  (76 against 169 in the cross-check configuration). The summaries were a
  wash: both root summaries named all 12 chapters, the session made no
  wrong skip where the planner made 3, and the planner answered 5 of the 5
  planted questions where the session answered 4. The navigating session's
  **peak context is 2.4× larger** (87 k against 36 k tokens) because it
  threads its whole history, and that thread is what would break on a
  document of the reference document's size. See
  [the spike's record](../../tools/spike_traversal/README.md).

The planner's own budgets (`budget_nodes`, `budget_windows`) shrink
window-to-window, so a merged plan can never exceed the level budget; an
EXPAND past the remaining budget is sent back for another verdict. Each
window is judged by a fresh agent carrying that window's verdict model —
reusing one agent would answer every window against the first window's
schema (bead `knox-t5ol`), because the agent surface snapshots the output
schema at build. The planner is consulted only about children that can be
summarized: leaves inline into their parent, single-child wrappers are
descended through without a verdict (bead `knox-hh2u`). Plan
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
- **Fail open to the full tree.** A declined level, a failed plan, an empty
  verdict, or a disabled decision model logs a warning and summarizes the
  whole tree. A merely sloppy verdict no longer triggers this: since bead
  `knox-3zjw` the merge tolerates it the way the executor does — an
  unmentioned child is EXPAND, an out-of-level answer is ignored, an
  unknown value takes the safe default, each noted in the plan notes — so
  one bad answer degrades one child instead of failing the level.
- **Off switch.** `summary_traversal = "off"` (the default) keeps today's
  path exactly; observability is one ledger row per tree (`summarize`,
  `ok`/`partial`, `reason_code=traversal`, `planned/skipped/expanded/windows`
  counters) with the full plan JSON in the cache layer, never in the row.
