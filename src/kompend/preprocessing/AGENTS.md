# DOX — kompend.preprocessing

<!-- on entry: loaded when agent enters src/kompend/preprocessing -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

What happens to a source *before* Tier 2 sees it: acquisition (downloaders),
duplicate detection (dedup), figure storage (images), page-range splitting of
multi-document packets (splitting), and the host-memory guard that serialises
heavy parses. Parsing itself lives in the child package `parsers/`.

## Ownership

- `src/kompend/preprocessing/` — downloaders, dedup, images, splitting, host memory
- `src/kompend/preprocessing/parsers/` — the parsecraft adapter and its routing bridge (own child doc)

## Modules

| Module | Contents |
|--------|----------|
| `splitting/` | Page-range splitting of a parsed packet into independent child sources: `_plan` (plan vocabulary, `SplitJudge` seam, `validate_plan` partition invariant and segment floor, `SplitReason`), `_elements` (`SourceSplitter`, `SplitDecision`, `partition_document`, `SPLIT_FIELDS`), `_judge` (`DecisionModelSplitJudge`, verdict models, judging windows, prompt rendering), `_rules` (category rules), `_provenance` (who owns a child's attachments), `_build` (`build_splitter` — the one config→judge site) |
| `host_memory.py` | `HeavyParseGuard`, `ensure_engine_headroom`, `available_commit_bytes`, `is_heavy_source` — the memory half of "one heavy parse at a time" |
| `images.py` | `ImageStore`, `store_asset`, rasterising/vector conversion helpers — figure bytes land here, the mapping lives in `parsers/parsecraft/assets.py` |
| `dedup/` | `DedupOrchestrator`, `DedupStore`, the MinHash/embedding matchers |
| `downloaders/` | `BaseDownloader` and the HTTP/crawler implementations behind online sources |

## Local Contracts

### Splitting is a partition, and it is fail-closed

A split plan must cover every page of the packet exactly once, in order
(`validate_plan`: no gap, no overlap, no short tail). A violation raises
`SplitPlanError` — nothing is truncated or dropped to make a plan fit. An
unpaged source, a page above `splitting.max_page_chars`, a judge error or an
unusable plan leaves the source whole and records why; the parse stage turns
that record into a ledger row (`DEGRADED`/`JUDGE_DEGRADED` for a real failure,
`SKIPPED` for an ordinary decline).

A cut must also leave a document on both sides: `splitting.min_segment_pages`
(`DEFAULT_MIN_SEGMENT_PAGES` = 3) bounds every segment, so a fragment is never
written as a source. The floor is enforced twice — `plan_from_answers` refuses
an individual cut that would leave a short side (the rest of the plan is still
the judge's), and `validate_plan` raises `SegmentTooShortError` for any plan
that arrives fragmenting, which `SourceSplitter.decide` records as
`SplitReason.SEGMENT_TOO_SHORT`. The floor bounds a *cut*: a one-segment plan is
the whole document and is not held to it.

There is no deterministic fallback. A page count cannot tell an amendment from
chapter, so a heuristic that guessed would fragment real documents — with no
judge reachable, the source is kept whole (`.agents/CODE.md`, LLM I/O policy).

### The judge answers one bool per candidate cut point

`boundary_verdict_model(page_count)` is the public, stable shape of the verdict:
one boolean field per transition between consecutive pages, the first page never
a candidate. It is not a list of segment models — the decision grammar refuses
`list[BaseModel]` — and it is not a list of strings, which collapses N questions
into N/4. `plan_from_verdict`/`plan_from_answers` turn answers into segments; a
missing answer is a continuation, never a boundary. The judge sees every page in
full: a page too large to fit declines the split instead of being cropped.

The decision service refuses more than `MAX_QUESTIONS_PER_REQUEST` = 64
questions in one request, so a longer packet is judged in consecutive windows
(`question_windows`): a window asks about a contiguous run of transitions and
sees exactly the pages those transitions sit between. Answers are keyed by the
*global* transition index (`window_verdict_model`), so windows compose; the
prompt's `whole` flag tells the model its first and last page are not boundaries;
and a window that fails raises `SplitWindowFailed` naming the window and its
page range, so the ledger row carries a structural failure rather than a bare
model error. Windows partition the *questions*, never a page.

### Rules are a resource, not a literal

The categories the judge may assign live in `templates/splitting/rules.yaml`,
resolved through the layered template loader so a project or user override wins
the way a prompt override does. `splitting.rules_file` points at a vault file
with the same shape. Parsing is strict — unknown key, duplicate id, non-slug id,
blank description is an error (`SplitRulesError`), because a half-understood
rules file asks the judge an ambiguous question and the answer decides whether a
real document is cut.

### `build_splitter` is the only config→judge site

`_build.build_splitter` resolves the model (`splitting.judge_model`, then
`decision.model` when `[decision]` is enabled), loads the rules and constructs
the `SourceSplitter`; `None` when `splitting.enabled` is false, so Tier 1 stays
byte-identical. Nothing downstream knows which judge decided. With splitting on
but no model resolved, the splitter carries `NoDecisionModel` rather than being
dropped: the run then records "asked, and there was nothing to ask" per source,
which is how an operator learns the switch is on and the judge is missing.

### A child is a virtual source; the parent keeps its attachments

A split child's stem carries its page range (`packet#p005-009`), so its cache
entry, state row and provenance are its own while it shares the parent's content
hash. Children own *copies* of the parent's elements (`partition_document`
rehomes copies, `docs/architecture/decisions.md`); the parent document is never
mutated. Because a child has no manifest entry, attachments stay with the parent:
`attachment_owner_stem` answers with `MetadataField.SPLIT_PARENT_STEM`, and an
extraction path asks it instead of deriving a stem from the child's own name.

### The split vocabulary is `MetadataField`; the package keeps only the order

The six `_split_*` keys cross file boundaries — `partition_document` writes them
onto a child's `ParsedDocument.metadata`, the parse stage stamps them onto every
leaf chunk, and figure attribution reads them back — so they live in
`kompend.core.models.enums.MetadataField`. `SPLIT_FIELDS` is the package-level
tuple of exactly those six members in stamping order: iterate it instead of
re-listing keys, and never retype a `_split_*` literal.

## Work Guidance

New splitting behaviour goes into `splitting/` beside the contract it serves, and
a new *resource* (rules, prompt) into `templates/` — never into Python string
literals (`src/kompend/AGENTS.md`, text-in-code ban). A capability that only
parsecraft can host belongs in `parsers/` instead; see that child doc.

## Verification

- `uv run pytest tests/unit/test_source_splitting.py -q` — partition invariant, judge seam, fail-closed declines, provenance, config wiring
- `uv run ruff check src/kompend/preprocessing tests/unit/test_source_splitting.py`
- `uv run ty check src/kompend/preprocessing`

## Child DOX Index

| Path | Scope |
| ---- | ----- |
| `parsers/` | The one ingestion parser (parsecraft) and its routing bridge |
