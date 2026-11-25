# Output-side failures of the local model are not sized (2026-11-25)

**Bead:** `knox-dnw3.1` (related to the epic `knox-dnw3`).

## Context

Pyvas run `594a0c4d` (2026-11-22) ended `status=error duration=18929.55s` with
no new pages. Three agents died the same way on the local model
`ollama/hf.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF:UD-Q4_K_XL`: a long
structured answer that failed validation 2 to 4 times. The run log
(`D:\SVN\...run-594a0c4d.log`) is **not reachable from this machine** — it was
never read here. Every number below says where it came from: either a line
quoted in the bead (marked **quoted**), or the code path that produces it
(marked **code**). No count was invented.

## Evidence — what each quoted line is, in code terms

| Agent | Quoted log line | Retry budget (code) | Input pressure (quoted) |
|---|---|---|---|
| compaction summarizer | `Summarizer failed (model-error:UnexpectedModelBehavior: Tool 'final_result' exceeded m…` | tool-call budget (`tool_retries`, default 3 — `LLMConfig.tool_retries`, code) | n/a |
| WiCER judge | `WicerEvaluateAgent failed on 'wicer-eval': Exceeded maximum output retries (2)` | `output_retries`, default 2 (code) | 34 % (22017/64000 tokens) |
| generation segment | `AgenticGenerationAgent failed on 'seg_0': Exceeded maximum output retries (4)` | `agentic_output_retries`, default 4 (code) | 68 % |

The `(2)` and `(4)` in the quoted lines match the configured output-retry
budgets exactly — these are output-validation exhaustions, not timeouts. The
input pressures (34 %, 68 %) are quoted; in both cases the window had room.
The `final_result` line is truncated mid-word in the bead quote, so its
exact source is unconfirmed (see Q1). The `VLM returned empty caption`
lines are quoted as "many" — no count is supportable without the log.

## Q1 — What exactly fails validation?

**Answerable only in part: the current log lines discard the evidence.**
Every quoted line reports the outer exception only. The validator errors
that would distinguish the cases live in the `__cause__` chain
(`classify_agent_failure` in `src/kompend/agents/failure.py` walks it), and
no stage logs them. From the code, the candidate failures are:

- **Truncated tool arguments.** Covered by salvage (see Q2) when the JSON is
  completable; not covered when the provider cut the stream mid-string and
  the completion invents content that validates but is wrong.
- **Wrong field names.** `salvage_json` schema mode raises `ValueError` on
  these and falls back to schema-less repair, so a renamed key burns the
  whole output budget (`src/kompend/llm/format.py:194`).
- **Length.** kompend sends no `max_tokens` (CODE.md ban), so a length
  failure is a provider-side cut (`finish_reason: length`, Ollama
  `num_predict`) arriving as truncation — indistinguishable in the log
  from a model that stopped early.
- **`Tool 'final_result' exceeded m…`.** The quote is cut off; the shape
  (`Tool '<name>' exceeded …`) matches a *tool-call* budget, not an
  output-validation budget — a different failure class from the other two
  lines. Pinning it down needs the untruncated log line; it is recorded
  here as unconfirmed, not as a fourth output-retry exhaustion.

Proposal: log the classified failure kind with every exhausted budget —
`classify_agent_failure` already distinguishes `EMPTY_ANSWER` /
`SCHEMA_MISMATCH` / `TRANSIENT` / `CONTEXT_OVERFLOW` / `REQUEST_BUDGET`.
After that, the log answers Q1 directly instead of this finding guessing.
**Implemented by knox-dnw3.9** for the WiCER judge (per-batch kinds in the
eval warning); the agent-boundary equivalent (log the cause chain where
the retry loop gives up) is a follow-up in `src/kompend/agents/` — outside
this track's file scope.

## Q2 — Does the text-to-tool-call salvage cover these cases?

Mostly yes for the judge's shape; no for renamed keys and provider cuts.
`src/kompend/llm/postprocess.py` runs, in order: `SurrogateScrubber` (lone
surrogates from small Ollama models), `ToolCallRepair` (schema-guided
`salvage_json` on tool args: completes truncated JSON, fills missing
fields, filters garbled items, coerces `""` to `[]`), `ThinkingPartRepair`
(raw-text JSON converted to a `ToolCallPart`, forcing the tool validation
path that always works). Covered: truncation that completes, bad escapes,
control chars, double-encoded JSON, text-instead-of-tool-call. Not covered:
wrong field names (schema repair raises, validation then fails on every
retry), and a provider-side cut whose completion validates with invented
content (silent wrongness, not a retry). Empty answers bypass salvage
entirely — `_fail_on_empty_answer` (`kompend_model.py`) resamples once
before the output budget is spent.

Proposal: no new salvage stage. The gap is key renames and silent
completions, and neither is fixable by parsing harder — the fix is fewer
verdicts per answer (Q3) so the model stops producing them. No code change
proposed.

## Q3 — Can the request be sized from the expected answer length?

Yes, and the pattern already exists. Extraction sizes the *prompt* from the
window (`RequestBudget`) and the *answer* from measurement:
`_expected_answer_tokens` is the tracker's `average_completion_tokens` else
a conservative seed, and `time_bounded_chars` (`request_budget.py`) turns
that plus the throughput fit into a prompt-char bound — the bound that was
missing when 49k-token batches timed out at 120 s. The judge is the same
problem one stage over: its answer grows with the probe count (one verdict
per probe), but `_evaluate` sends **all** probes in every segment request.

Proposal: batch the judge's probes so prompt + expected answer fits the
ceiling with slack (`fits_within`), with per-verdict tokens measured from
the run's own eval answers (completion delta per judged probe) else the
probes' own priced size; no fit yet judges everything in one request (the
first calls are how the fit is learned). One retry at half the batch
before a batch counts as unevaluated. **Implemented by knox-dnw3.9** for
the judge. Probe generation and agentic generation still size only their
prompts — same treatment there is a follow-up (the `seg_0` failure is the
same class through a bigger answer).

## Q4 — Should a repeatedly failing model be demoted for structured tasks?

Yes, for consecutive structured-output exhaustions within the run — but it
is a breaker, not a size, so it belongs with the per-model breaker issue
(`knox-dnw3.6`), not with WiCER.

Proposal: count consecutive output-retry exhaustions per model (the
`record_unit_failure` / `UnitFailure` path already classifies them); after
a run of them, route structured-output agents (probe, judge, diagnose,
agentic generation) to the fallback chain first for the rest of the run,
resetting on the first success. Demotion is run-scoped and consecutive —
a single failure is noise, and a recovered model must not stay demoted.
Follow-up: not implemented by knox-dnw3.9. It touches `src/kompend/agents/`
and `src/kompend/llm/`, both outside this track's file scope.

## What knox-dnw3.9 implements, and what stays open

Implemented (knox-dnw3.9): the distinct evaluated/unavailable outcome with
no diagnose and no recompile on unavailable (the Q1 observability half —
unevaluated stops sharing a value with wrong); judge probe-batching from
the measured answer throughput with one retry at half size (Q3 for the
judge); the `evaluation unavailable` summary; per-batch failure-kind
counts in the eval warning so the next such log answers Q1 directly.

Open follow-ups: agent-boundary cause-chain logging (Q1, agents track);
probe/generation answer-sizing (Q3, other tracks own those files); the
`final_result` tool-budget line needs its untruncated log text before
anyone sizes it; per-model structured-output demotion (Q4, breaker issue
knox-dnw3.6).
