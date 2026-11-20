---
---

# Self-Calibration

**Rule.** Compilation and linting must self-calibrate. Every context-window-,
throughput-, or corpus-dependent quantity is derived from measurement — the
discovered window, the model's measured throughput and token density, the
corpus size — and adapts when a request fails. No absolute number is an
operational default. Config values are ratios and bounds that shape the
adaptation (floors, caps, utilization fractions, sample windows), never the
outcome.

**Why.** A run must scale from a few-document wiki to 1000+ documents and from
small local models to 1M-context models without the operator reassessing
every knob. Hand-tuned constants are wrong at both ends and fail silently:
requests sized to exactly the window are refused by the provider, conversation
threads outgrow the window, compaction cannot shrink one oversized message,
and merging refuses wholesale on a redundant corpus.

**Required behaviour.**

- Size requests from the effective window minus answer reserve and safety,
  and from measured chars-per-token once samples exist.
- Price the riding conversation thread into the payload budget.
- On failure, adapt and retry — smaller payload, lowered trigger, relearned
  per-source cap — keeping the work done so far.
- Trigger compaction from the same budget arithmetic, and escalate when a
  single message cannot be shrunk.
- Report the inputs (window, measured rate, corpus units) with any budget
  decision, so a refusal explains itself.

**Forbidden.**

- New absolute constants in sizing paths.
- Per-vault tuning as the intended workflow; a knob is a last resort and must
  be justified in its issue.

## Where it lives in the code

- `src/kompend/llm/request_budget.py` — the `RequestBudget` arithmetic every
  payload and compaction decision derives from.
- `src/kompend/llm/throughput.py` — measured call samples and the fit that
  prices a request in seconds.
- `src/kompend/llm/context_window.py` — the discovered window, its answer
  reserve and history budget.
- `src/kompend/agents/base/context.py` — compaction presets as fractions of
  that budget, never of the raw window.
