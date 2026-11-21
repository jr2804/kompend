# Traversal spike — planner + worker vs navigating session

The spike behind bead `knox-u84v.6`, child 6 of the `knox-u84v` epic
("Decision-model traversal for chunk-tree summarization").

## The question

Is the shipped **planner + worker** traversal (one decision call per level,
then the existing summarizer over the planned set) worse than a **navigating
summarizer session** that decides node by node with `list_children` /
`browse` / `summarize_node` / `skip` tools?

The epic's gate: keep planner + worker unless navigation wins by **more than
25 % on total calls**, or produces **clearly better summaries** on the
spot-checks.

## What runs

| | variant A | variant B |
|---|---|---|
| shape | the shipped path: `NodeSummarizer` with `TraversalMode.AUTO` + a traversal planner | one agentic session with four navigation tools |
| decision calls | one per expanded level (`decide_traversal_plan`'s window loop) | the session's own navigation turns |
| content | the executor assembles each node's payload | the session fetches it with `browse` |

Both arms run the **same** model, the **same** task statement
(`common.task_statement()` — the shipped traversal task text, not a
paraphrase), the **same** corpus (`tree.build_synthetic_doc()`, rebuilt and
compared field by field before the runs) and the **same** budget: at most
**60 summarized nodes** and **200 content windows**.

## The corpus

The pyvas vault's `sources/26253-j30.docx` (26,886 nodes, 942 eligible) is
not reachable from the machine this spike ran on — the vault lives outside
this repository and the document is not on the host. The bead allows "a
synthetic 1,000-node subtree cut", so the corpus is synthetic, of the same
shape and with the same properties the traversal has to cope with:

- 1,191 nodes, 366 of them internal nodes with more than one child;
- 12 chapters → 6 sections → 3–5 subsections → 2–4 leaf chunks each;
- dense technical prose in the branches that deserve a summary;
- 12 boilerplate branches (legal notice, copyright, table of contents,
  revision history, safety notice, glossary) — a skip is correct there;
- the same safety notice repeated across chapters — repeated material;
- 5 facts a reader would ask for, planted in named chapters, plus 12
  chapter headings the root summary may name.

**What the synthetic corpus does not carry**: the pyvas document's real
structure, its real boilerplate share, and a local model's behaviour. The
numbers below are the shape's numbers on this corpus; they are not a
prediction for `26253-j30.docx`.

## The model

`minimax/MiniMax-M3.1-Flash` over the Anthropic-compatible endpoint the
session's own credentials point at (`common.configure_provider()` reads
`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` from the environment; nothing
is written to disk). One model for both arms, and for the judge.

## The measurement

`common.CountingModel` wraps the model and records every request: the
provider's own input and output token counts, the **peak** input tokens of a
single request (the provider counts the whole conversation on every request,
so the maximum is the peak context the arm carried), and the request count
split into *decision* and *summarize* by phase (arm A) or by the tool the
response called (arm B).

A **window** is one request's worth of content delivered, in both arms:
arm A counts the payload entries it hands a session, arm B counts the
`browse` replies that carried text. A summary that sends no content costs no
window.

The three spot-checks (`checks.py`):

1. **Root coverage** — how many of the 12 chapter headings the root summary
   names (case-insensitive substring).
2. **Skip quality** — of the skipped branches: how many are genuinely
   boilerplate, how many carry a planted fact (a *wrong* skip — it loses an
   answer), how many are neither.
3. **Answer coverage** — an LLM judge (same model, its own counter, so the
   arms' numbers stay clean) answers for each planted fact: does the arm's
   summary set actually answer the question?

## Deviations from the shipped path (and why)

Three, all in arm A, all recorded in the run's notes:

1. **Tolerant verdict merge** (`_merge_verdict`). The shipped
   `plan_from_verdict` refuses a verdict that does not cover its level
   exactly, and the executor answers a missing plan by summarizing the whole
   tree. The spike merges the way the *executor* merges: a child the plan
   does not mention is `EXPAND`, an answer outside the level is ignored.
   Filed as `knox-3zjw`.
2. **A fresh decision agent per level.** `KompendAgentBase` snapshots
   `result_type` when it builds the pydantic-ai agent and caches that agent,
   so an agent reused across levels keeps the first level's schema. Filed as
   `knox-t5ol`; the spike builds the agent per level, as the shipped
   adapter does.
3. **A total node budget of 60 across the tree.** The shipped
   `summary_traversal_max_nodes` bounds one *level*, not the tree, so arm A
   could plan far more work than arm B is allowed. The spike counts planned
   work (EXPAND + SUMMARIZE) across levels and turns the excess into
   recorded skips with the reason `"budget"`. Filed as `knox-s2l1`.

Two measurement fixtures, also noted in the run: the decision agent's
per-call timeout is 120 s (the shipped `DecisionConfig.timeout` of 10 s is
sized for a local model; against a remote endpoint a 12-field verdict
regularly needs longer, and a timeout makes the planner decline the level),
and the counted model is exposed through `.delegate` so kompend's output
ceiling detection still sees the Anthropic protocol behind the wrapper.

**One asymmetry that leans towards arm B**: arm B's session prompt carries
navigation strategy (work chapter by chapter, skip a boilerplate branch on
its heading, page through `browse`). Arm A's planner prompt is the shipped
one — a per-child question with the skip vocabulary, and no strategy. Both
arms are tuned to their shape, but B's prompt is more prescriptive. A
verdict that favours navigation must carry that caveat.

## The result

Two configurations were run, because the first one measured the spike's own
budget device instead of the traversal shape.

**Configuration 2 (the headline run).** The budget's excess is *dropped from
the plan*: a child the plan does not mention is undecided, and the executor
inlines an undecided child's subtree into its parent's payload. No content is
lost.

| measure | A — planner + worker | B — navigating session |
|---|---|---|
| total calls | **1,001** | **107** |
| decision calls | 341 | 76 |
| summarize calls | 660 | 31 |
| content windows | 338 | 124 |
| prompt tokens | 7,340,831 | 4,796,994 |
| completion tokens | 280,686 | 37,794 |
| peak prompt tokens | 36,509 | 87,348 |
| wall time (s) | 3,495 | 485 |
| nodes summarized | 60 own (366 carry a summary, 306 inherited) | 60 |
| skips | 77 (72 "repeated material", 5 boilerplate) | 13 (12 boilerplate, 1 repeated) |
| root coverage | 12/12 | 12/12 |
| wrong skips | 3 | 0 |
| facts answered | 5/5 | 4/5 |

**Configuration 1 (cross-check).** The budget's excess became recorded
`"budget"` skips — which *drops* the branch's content. Arm A's numbers are
much smaller because it summarised much less: 169 calls against B's 76,
490 k prompt tokens against B's 2.82 M, 61 windows against B's 61. But A
answered **0 of 5** planted questions, because 199 of its 204 skips were the
budget's, and a skip's content is gone. That configuration measures the
allocation rule of the cap, not the shape; it is kept here as the
cross-check, not as the result.

### Why arm A's calls explode in configuration 2

Two structural facts, both measured, both filed:

- **The planner is consulted about nodes that can never be summarized.** A's
  planner asked about **1,092 children** over 337 levels to plan 60
  summaries; 726 of those children (67 %) are leaf chunks, which
  `_eligible_nodes` never summarizes — the verdict about them decides
  nothing (`knox-hh2u`).
- **A parent's summary carries its subtree's content.** The 955 branches the
  node budget left unplanned had their content inlined into the 60 planned
  payloads, so arm A delivered 338 windows through the shipped summarizer
  session — which is serial, one node per round trip — and spent 660
  requests on 60 summaries.

### The confound, stated plainly

In configuration 1 the two arms' *planning* costs were nearly identical — 59
decision calls for the planner, 61 navigation turns for the session — and the
entire call gap came from how each arm turns a decision into summaries: the
shipped summarizer session is serial, the navigating session issues parallel
`summarize_node` calls (15 requests for 4 summaries in the smoke run, 31 for
60 in the full run). So part of "navigation uses fewer calls" is a property
of the summarization protocol, not of the traversal shape.

### The verdict

The bead's gate: keep planner + worker unless navigation wins by more than
25 % on total calls **or** produces clearly better summaries on the
spot-checks.

- **Calls: navigation wins** — 107 against 1,001 (89 % fewer) in
  configuration 2, and 76 against 169 (55 % fewer) in configuration 1. The
  margin survives the configuration change, so it is not an artifact of the
  budget device.
- **Quality: a wash** — the root summary names all 12 chapters in both arms;
  B makes no wrong skip where A makes 3, but A answers 5 of the 5 planted
  questions where B answers 4. B is not "clearly better" on the
  spot-checks.

**Per the gate, navigation wins.** What the record does *not* support is
changing the shipped path on this measurement: the win is in requests, not
in tokens (B's peak context is 2.4× A's — 87 k against 36 k — because the
session threads its whole history, and that thread is what would break on
the 26,886-node reference document), the corpus was synthetic, the model
was a remote one rather than the pyvas local model, and arm B's prompt
carries navigation strategy that arm A's shipped planner prompt does not.

## What the spike found in the shipped planner

Three defects, filed as beads, all of which the shipped path hits:

| bead | what |
|---|---|
| `knox-s2l1` | `summary_traversal_max_nodes` bounds one *level*, not the tree — the total work is bounded only by the tree and the model's verdicts |
| `knox-3zjw` | one sloppy verdict (a field short or a field too many) declines the level, and the executor answers a missing plan by summarizing the **whole tree** |
| `knox-t5ol` | `KompendAgentBase` snapshots `result_type` when it builds the pydantic-ai agent and caches that agent, so a level wider than one 64-child verdict window is answered against the first window's schema |
| `knox-hh2u` | the planner is consulted about leaf chunks that can never be summarized — 67 % of its consultations in this run |

## Running it

```sh
# the endpoint credentials come from the environment
mise exec -- uv run python tools/spike_traversal/run_spike.py --variant both --out tools/spike_traversal/result.json
```

`--variant a|b|both`. The runner writes the measurements, the summaries, the
skips and the three spot-checks per arm into `result.json`, and prints the
comparison table with the gate arithmetic.

The committed `result.json` is the recorded run's payload with the per-arm
summary dumps reduced to the root summary and the summaries on the planted
facts' paths (~700 kB of generated prose dropped). Every measurement, every
skip and every check is verbatim; the full dump comes back by re-running the
harness.

## Files

| File | What it is |
|---|---|
| `common.py` | the shared contract: model, task statement, budget, the counting model, the measurement |
| `tree.py` | the synthetic corpus (seeded, reproducible) |
| `variant_a.py` | arm A — the shipped planner + worker path |
| `variant_b.py` | arm B — the navigating session |
| `checks.py` | the three spot-checks |
| `run_spike.py` | the runner and the gate arithmetic |
| `prompts/` | arm B's prompts and the judge's prompt (text lives here, not in Python) |
