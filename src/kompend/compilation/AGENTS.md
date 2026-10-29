# DOX — kompend.compilation

<!-- on entry: loaded when agent enters src/kompend/compilation -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Tier 4 compilation — build an interlinked knowledge graph (`CompiledWiki`)
from resolved chunks. Two-phase extraction-then-generation with cross-source
merging and an optional iterative WiCER refinement loop.

## Ownership

- `src/kompend/compilation/` - compilers, graph construction, schema, transcript recovery

## Architecture

| Module                  | Contents                                                                 |
| ----------------------- | ------------------------------------------------------------------------ |
| `base.py`               | `BaseCompiler` ABC — compiler interface with progress hook               |
| `two_phase.py`          | `TwoPhaseCompiler` — extract-first then generate with cross-source merge |
| `_two_phase_core.py`    | Core extract/generate loops extracted from `two_phase.py`                |
| `wicer.py`              | `WiCERCompiler` — iterative correctness/entity/consistency refinement    |
| `graph.py`              | `GraphCompiler` — dependency-graph-aware compilation                     |
| `schema.py`             | Wiki graph schema: predicates, entity types, relation types              |
| `step_recovery.py`      | Transcript-key derivation + scan/extract for step-store resume           |
| `flows/`                | Flow-based compilation building blocks                                   |
| `flows/assembly.py`     | Cross-source assembly flow                                               |
| `flows/extraction.py`   | Extraction flow                                                          |
| `flows/agentic.py`      | AGENTIC extraction flow — per-section `ExtractionMemory` digest sized by the shared planner (`_digest_budget_tokens`: window − answer reserve − safety − system − section); over budget the digest is model-consolidated (`MemoryConsolidationAgent`, rolling delta, never cropped); provider context-overflow 400s lower the window to the provider's limit and split the section into windows that fit (`_replan_section`), and the section/consolidation agents are rebuilt against the clamped window (`_section_agents` — they turn it into compaction thresholds and the output ceiling at construction); only an overflow that cannot be split skips the section — a rejected payload is never resent unchanged |
| `flows/generation.py`   | Generation flow; `build_known_titles` (canonical titles for wikilinks) is the only reserved payload that grows with the corpus, so `_known_titles_windows` splits its `_KNOWN_TITLES_MAX_FRACTION` window share into an inline prompt head and a `list_known_titles()` page — windowed, never truncated; `_agentic_budget` declares the request's reserved payloads to the shared planner; `prepare_source_segments` raises when even the minimum segment does not fit the plan (planning-time failure, before any LLM call); `learn_wikilink_aliases` (post-generation alias map persisted as `MetadataField.WIKILINK_ALIASES`) |
| `flows/references.py`   | Reference-extraction flow (pre-extraction pass; retries = `llm.output_retries`; splits oversized sections into windows bounded by the planner's input room and the window's answer reserve (`ContextWindow.answer_reserve`); every window is processed) |
| `reference_excision.py` | Reference-section detection + excision (regex-based)                     |
| `write_tools.py`        | Tools of the agentic write agent — `read_source(stem, offset)` serves a source in windows of `budget.capture_source_char_max` characters and names the next offset, so a long source is paged through, never shortened |

## Rules

- Import from the package root: `from kompend.compilation import GraphCompiler`,
  `from kompend.compilation.wicer import WiCERCompiler`.
- `two_phase.py` is the primary entry point; `_two_phase_core.py` holds
  extracted core logic to keep the main module focused.
- `WiCERCompiler` wraps a `TwoPhaseCompiler` and adds iterative probing.
- No prompt text belongs here - templates live in `src/kompend/templates/`.
- **LLM-product cache keys carry the model** (extraction deps, generation
  deps, summary deps): a model switch invalidates them by
  design. `compile.cache_ignore_model` (`--ignore-model`) drops the model
  from all three keys so a run can be finished on another model - the
  per-page `MetadataField.MODEL` tag keeps the result attributable.
- **Segment transcripts live in the step store, not in a checkpoint file**
  (`step_recovery.py` + `flows/generation.py`): the harness
  `StepPersistence` capability persists every run into
  `SqliteStepStore(paths.step_db_file)`; the conversation id is the
  generation deps key and each pass gets a deterministic
  `...-segNNN-aNNpNN` run id. A complete snapshot whose final result
  validates as `GenerationOutput` skips the segment; an interrupted one
  resumes with its history; `--no-cache` ignores both. The store is a
  separate SQLite file because `kompend.db` is recreated clean-cut on
  schema bumps.
- **No content is excluded by size or by a threshold** (see the budget rule in
  `.agents/CODE.md`). `min_content_length` and `max_concept_pages` are warning
  thresholds: short pages and every concept-page candidate are compiled. Only
  content without any body text is dropped. `wicer_max_iterations = 0` means
  "until the loop converges"; `wicer_max_seconds` is a wall-clock stop that
  keeps the compiled wiki.
- **Every request is sized by the shared planner**
  (`kompend.llm.request_budget.RequestBudget`), never by stage-local
  arithmetic. A stage declares the window and its named reserved payloads
  (`{"system": …, "known_titles": …}`) and reads the content budget from the
  plan; the answer reserve and the safety margin are paid by default. A
  request that cannot fit raises `LLMOutputError` naming the components —
  content is never cropped to fit. `_two_phase_core.py` re-exports
  `context_safety_tokens` / `tokens_to_chars` / `chars_to_tokens` from the
  planner; do not define a second safety margin or conversion here.
- **A provider that rejects an oversized request lowers the window**
  (`TwoPhaseCompiler._clamp_window` → `EffectiveWindow.clamp`), and the stage
  re-plans against it — the source is re-batched, the segments are
  re-computed, an agentic section is split into windows, and the content is
  never cropped. The provider's own number
  (`kompend.llm.overflow.stated_context_size`) is the only thing that may
  lower the window; a limit kompend cannot read leaves the failure loud.
  Lowering is one-way for the run, so generation re-plans against every
  rejection that names a *lower* limit (`_MAX_CONTEXT_CLAMPS` bounds how
  many) and a repeated limit raises: nothing is left to re-plan. The
  extraction re-plan keeps the batches already extracted and works on the
  source's *texts* for the rest — a text that exceeds the new budget is split
  into windows, so a joined batch never has to be re-sent whole and no text
  is extracted twice.
- **The progress bar reports the units the flow works in** — the sections of
  a document (agentic extraction), its batches (the other strategies), the
  concepts of a generation segment. `ProgressFn` (`_two_phase_core.py`) is
  `(units done, units total, activity)`; the flows pass it down so a long
  document moves the bar instead of freezing it. The bar counts characters,
  so a document's share is paid out as its units complete
  (`_source_progress` in `flows/extraction.py`). Extraction also counts the
  units of every source up front — the chunk tier built every tree before the
  loop — so the line carries the run-wide counter next to the document's
  (`(12/234 · 2450/9800 sections)`); the position in the whole run stays
  readable in the unit the flow works in. The *activity* names the running
  call (e.g. `consolidating context`). `rate_suffix` is the single source of
  truth for the throughput suffix. A re-plan changes the unit count a source
  works in: the bar's weights and the corpus-wide counter follow the new
  count (`_source_progress`) instead of reporting a total the bar cannot
  reach.
- **A bar line carries counters, not labels** — `extraction_detail`
  (`flows/extraction.py`) and `generation_detail` (`flows/generation.py`)
  build them. A counter without its denominator ("segment 2" — out of how
  many?) is noise, and so is a second unit beside the main one ("4/12
  concepts covered" next to a page counter). A sub-counter in another unit
  names that unit ("12/234 sections"); a sub-counter in the main counter's
  own unit needs no word ("(4/12)"). A note ("recovered") names a state
  and stands on its own.

### Page kinds (category taxonomy)

Every generated page carries a `kind` (OKF `type`): the **mandatory taxonomy**
(`concept`, `entity`, `summary`, `comparison` - `MANDATORY_KINDS` in
`core/keys.py`) is always allowed and cannot be deactivated; extra
kinds come from `compile.extraction.categories`. The LLM assigns `kind` per
page in `generation_prompt.md.j2`; `_allowed_kind` in `flows/generation.py`
validates it at node creation — anything outside the allowed set falls back
to `concept` (permissive-consumption model, never a hard failure).
Exception: extracted entities become mandatory pages — `ExtractionSet.entity_pages()`
merges them with a pre-assigned `kind: entity`, and `_resolve_page_kind` in `flows/generation.py`
overrides the LLM's label for those entries. Entity specs colliding with an existing
concept key are dropped to avoid duplicate pages.
