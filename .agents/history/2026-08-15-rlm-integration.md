# RLM Integration Plan — KNOX (rewrite v2, verified against sources)

Created: 2026-08-14
Status: **ready to implement** (Phase 1). Supersedes the previous draft, which was
written against an API that does not exist in the published package.

## 0. Verification summary (what changed vs. v1)

| Claim in v1 | Reality (verified) |
|---|---|
| `pydantic_ai_rlm.RLM` class with `depth`/`max_iterations`/`compaction` | **Does not exist.** Package API: `create_rlm_agent(model, sub_model, code_timeout, custom_instructions, grounded)`, `run_rlm_analysis(...)`, `RLMDependencies(context, config)`, `RLMConfig(code_timeout=60, truncate_output_chars=50000, sub_model=None)`, `REPLEnvironment`, `create_rlm_toolset` (one tool: `execute_code`), `max_tool_calls=50`. |
| `rlm_query(prompt, model)` recursive calls | **Does not exist.** Only `llm_query(prompt)` when `sub_model` is set — and it is a **plain** `model_request_sync`, not recursive. See upstream issue vstorm-co/pydantic-ai-rlm#4 (open). |
| `answer` dict protocol | **Does not exist.** Agent returns `str` output. (rlm-minimal uses `FINAL(...)`/`FINAL_VAR(...)` text markers instead.) |
| `RLMCapability(AbstractCapability)` from the package | **Does not exist.** Package ships no capability hooks. We build the capability wrapper ourselves (this plan, Phase 1). |
| Compaction config fields (`rlm_compaction_threshold` etc.) | **Drop.** KNOX already ships compaction: `ContextManagerCapability` (pydantic-ai-summarization) is wired in `KnoxAgentBase._resolve_context_capability`, enabled via `llm.context_manager` (`"default"`/`"aggressive"`/`"sliding"`/`false`). RLM and compaction are orthogonal (see §4). |
| Files: `src/knox/llm/config.py`, `*_agent.md.j2` templates | **Wrong.** Config lives in `src/knox/core/config/_configs.py` (`LLMConfig`) + `keys.py` + `generate.py`. Templates: `templates/compilation/*.md.j2` (extraction_system/prompt, generation_system/prompt, wicer_probe/diagnose/evaluate) and `templates/query/`. |
| Sandbox blocks `open` | **Wrong.** REPL `SAFE_BUILTINS` includes `open` and `__import__` (code can do anything the process user can). Per decision: accepted as-is (§8). |

**Version compatibility (verified by import test in KNOX venv):** `pydantic-ai-rlm`
0.1.2 imports cleanly against pydantic-ai 2.30.0 (`pydantic_ai.direct`,
`pydantic_ai.toolsets.FunctionToolset`, `Agent`, `UsageLimits`, `ModelRequest`,
`RunContext`, `TextPart`). Its dep `pydantic-ai-slim[cli]>=0.1.0` is a wildcard —
no conflict.

**Model bridge (verified):** `build_model('ollama/...')` returns a `Model`
instance; `create_rlm_agent(model=<Model>, sub_model=<Model>)` constructs
successfully, and the REPL's `llm_query` path (`model_request_sync`) accepts
`Model | str` — so KNOX providers (Ollama, user providers, models-dev catalog)
work for both the main agent and the sub-model. The `str` type hints in the
package are cosmetic.

## 1. Problem

KNOX agents (query, generation, WiCER) pack large context blocks into one LLM
call (`QueryDeps.context`, `GenerationDeps.content`, `WicerProbeDeps.text`).
With local models (32–128K context) this hits context rot: more context → worse
replies. Extraction is **excluded** — Tier 2 chunking already bounds its input.

## 2. What RLM is (paper 2512.24601 + rlm-minimal reference impl)

- Root LM writes **Python code** instead of receiving the full document.
- Code runs in a REPL where the document is the `context` variable; the model
  slices/searches it programmatically.
- `llm_query(prompt)` delegates semantic analysis of a small slice to a
  sub-model. Recursive RLMs give the sub-model the same REPL + `llm_query`
  tools, depth-limited (rlm-minimal: `recursive_model`; upstream issue #4's
  proposed fix).
- Answer is returned via `FINAL(...)`/`FINAL_VAR(...)` text markers
  (rlm-minimal) or as the agent's structured output (pydantic-ai-rlm).
- Loop bounded by `max_iterations` (rlm-minimal) / `max_tool_calls`
  (pydantic-ai-rlm).

Whamp/pi-rlm adds the practical ladder worth encoding in prompts: **REPL for
structural queries → llm_query for semantic slices → subagent only for final
synthesis**. manojlds/pi-rlm contributes guardrail ideas (depth, node budget,
cycle detection) for the recursion phase.

## 3. Design decision (resolved): capability wrapping — option (b)

Per discussion: wrap the package's REPL/toolset into a KNOX `RLMCapability` on
`KnoxAgentBase` — the same agent gains code-exploration tools, closest to the
paper. Not option (a) (delegate whole stages to `run_rlm_analysis`), which would
lose structured outputs and duplicate the agent lifecycle.

### 3.1 What we own vs. what we reuse

| Layer | Source |
|---|---|
| `REPLEnvironment` (sandbox, context loading, `llm_query` internals) | reuse from `pydantic-ai-rlm` |
| Tool glue (`execute_code` tool bound to KNOX deps) | **own, ~80 LOC** — package toolset is typed for `RLMDependencies`; KNOX deps types differ (`QueryDeps`/`GenerationDeps`/…). Our toolset reads `context` from a configurable deps field and constructs `RLMDependencies` internally. |
| Capability (`RLMCapability(AbstractCapability)`) | **own** — usage tracking via `KnoxModel`-wrapped sub-model, `on_usage_update` callbacks |
| Recursion (`rlm_query`) | **own** (Phase 2) — implements upstream issue #4 fix |
| Compaction | **existing** — `ContextManagerCapability`; reference, don't restate |

### 3.2 Attachment point

`KnoxAgentBase._build_agent()` (src/knox/agents/base.py):

1. New field `rlm: RLMConfig | None = None` (default `None` → zero behavior change).
2. When set: build `RLMToolset(context_field=..., code_timeout=..., sub_model=KnoxModel(build_model(...)) or None)`; pass `toolsets=[toolset]` to `Agent(...)`.
3. Append RLM instructions to the system prompt (see §5.3).
4. Attach `RLMCapability` (usage hooks) alongside the existing `ContextManagerCapability` — both can be on simultaneously.

Context binding: the REPL `context` variable is loaded from the KNOX deps field
(`QueryDeps.context`, `GenerationDeps.content`, `WicerProbeDeps.text`) at first
`execute_code` call, cached per `deps.source_id` (mirrors the package's
`_repl_registry` keyed by deps id).

### 3.3 Interaction with compaction (explained)

The two mechanisms operate on different things:

- **`ContextManagerCapability`** compresses the **conversation history** when it
  nears the model's context budget. RLM code-execution results accumulate in
  the conversation too — compaction still applies to them.
- **RLM** keeps the **document data** out of the conversation entirely: the data
  lives in the REPL `context` variable and code returns only small slices.

They compose: RLM prevents rot at the source (data never enters messages),
compaction bounds the conversation (code output accumulation). No new
compaction fields; `rlm` config carries no `compaction_*` keys.

## 4. Config

In `LLMConfig` (src/knox/core/config/_configs.py), `keys.py`, `generate.py`
`DEFAULTS`:

```toml
[llm]
rlm = false              # false | "default" | "aggressive" | RLMConfig inline
rlm_sub_model = "ollama/gpt-oss:20b"   # optional; default = main model
rlm_code_timeout = 60    # seconds per execute_code
rlm_max_tool_calls = 50  # loop bound
rlm_max_depth = 2        # recursion depth (Phase 2; 0 = non-recursive)
```

Presets (in `generate.py` `DEFAULTS`): `default` = `{max_tool_calls: 50,
code_timeout: 60, max_depth: 2}`, `aggressive` = `{max_tool_calls: 100,
code_timeout: 120, max_depth: 3}`. `false`/`None` = disabled (today's behavior).

Per-agent override: `KnoxAgentBase.rlm` field; pipeline factory maps
`llm.rlm` → agent fields.

## 5. Phases

> Progress: Phase 1 ✅ Phase 2 ✅ Phase 3 ✅ Phase 4 ✅ (2025-08-14) — 1690
> unit tests green, ruff clean (0 new), ty clean on new code; live
> verification via `scripts/verify_rlm.py` PASSES against Qwen3.5-2B-MTP.

### Phase 1 — dependency + RLMToolset + RLMCapability (non-recursive) ✅

- `pyproject.toml`: optional extra `[rlm]` → `pydantic-ai-rlm>=0.1.2`.
- `src/knox/agents/rlm.py` (new):
  - `RLMToolset` — wraps `REPLEnvironment`; `execute_code` tool reading deps
    context; `llm_query` tool when sub_model set (usage-tracked via
    `KnoxModel`).
  - `RLMCapability(AbstractCapability)` — `on_usage_update` → existing
    `UsageTracker`.
  - `build_rlm_instructions(depth, max_tool_calls)` — prompt block.
- `src/knox/agents/base.py`: `rlm` field + wiring in `_build_agent()`.
- `src/knox/core/config/`: `rlm` fields in `LLMConfig` + keys + defaults.
- Unit tests: toolset context binding from KNOX deps; REPL executes against a
  synthetic context; `llm_query` sub-model routing; config parse + presets.
- Exit: `rlm=false` default, all existing tests green, no behavior change.

### Phase 2 — recursion (`rlm_query`), implementing upstream issue #4 ✅

Status: **done (2025-08-14)**. `rlm_query(prompt, context=None)` REPL
builtin spawns a child agent (same main model, own REPL over the given
slice) recursing up to `rlm_max_depth`; depth 0 children fall back to
`llm_query`-only. Guardrails: per-root-run node budget (16, `node_budget`),
cycle detection on (depth, prompt, context fingerprint), shared guard
across the tree. Usage flows through the KnoxModel-wrapped main model.
Config: `rlm_max_depth` (default 2, 0 = non-recursive).

- `RLMToolset` gains `rlm_query(prompt, model=None)` tool: builds a child agent
  (same toolset, `max_depth-1`, deps context = sub-slice) and runs it;
  falls back to plain `llm_query` at `max_depth=0`.
- Guardrails (from manojlds/pi-rlm): depth limit, per-run node budget,
  cycle detection on identical (prompt, depth) pairs.
- Unit tests: depth fallback, node budget, cycle detection, usage accounting.

### Phase 3 — pilot wiring (query agent first) ✅

Status: **done (2025-08-14)**. Wired `rlm`/`rlm_sub_model`/`rlm_max_depth`
from `compile.llm.*` into: QueryAgent (via `synthesize_results` +
`HybridRetriever`/`EnhancedRetriever.synthesize`, CLI `knox query`, MCP
query tool; `rlm_context_field="context"`), GenerationAgent
(`rlm_context_field="content"`), WicerProbeAgent (text, default field).
When RLM is enabled the inlined context is replaced by a placeholder in
rendered prompts (`RLM_CONTEXT_PLACEHOLDER`) so the model explores via
code — QueryAgent overrides both prompt renderers, so the swap lives in
its `render_system_prompt` too. ExtractionAgent intentionally not wired.

1. `QueryAgent` — biggest context pressure (search synthesis over many
   snippets), free-text output. Enable via config `llm.rlm="default"`.
2. `GenerationAgent` (`GenerationDeps.content`), `WicerProbeAgent` (`text`).
3. `ExtractionAgent` — **out of scope**: input is already chunked; RLM adds
   latency without benefit.

Template change (per agent, appended to system template — e.g.
`templates/query/synthesize_system.md.j2`):
append a `## RLM Mode` block (only when enabled) with the Whamp ladder: REPL
first for structural slicing → `llm_query` for semantic analysis of slices →
synthesize into the agent's structured output. Keep code concise; never dump
the whole context into one `llm_query`.

### Phase 4 — integration verification + docs ✅

Status: **done (2025-08-14)**.

- `scripts/verify_rlm.py` (not pytest — tests/AGENTS.md forbids real LLM
  calls in any test): runs QueryAgent with RLM off vs. on over a synthetic
  corpus (~4.3K tokens, fact at the tail) against a live Ollama model
  (default `Qwen3.5-2B-MTP`, override via `KNOX_TEST_RLM_MODEL`); prompt-
  shape checks run without a model call. Verified on this machine:
  rlm-off gives garbage (no fact), rlm-on answers correctly via REPL code
  search; conversation bounded (placeholder, not corpus).
- Model findings: LFM2.5-8B-A1B ignores the tool, gemma4:e2b emits code as
  text (no structured tool call), Qwen3.5-4B too slow; Qwen3.5-2B-MTP
  works. `build_rlm_instructions` made **directive** ("MUST call
  execute_code") — permissive "MAY" let weak models answer from the empty
  placeholder.
- Docs: `docs/guides/llm-config.md` — RLM section (what/why, config keys,
  presets, sub-model, depth, wiring, verification script).
- Note: Ollama serves full context (num_ctx=131072) — no hard truncation;
  the rlm-off failure is lost-in-the-middle retrieval failure, which is
  the same practical outcome and what the script asserts.

## 6. Open questions — resolved

| v1 question | Resolution |
|---|---|
| `pydantic-ai-rlm` availability/compat | **Verified**: 0.1.2, works with pydantic-ai 2.30.0. |
| Small local model support | Code execution is local (REPL); `llm_query`/main need a model — KNOX `build_model` covers Ollama/custom providers (verified). |
| Compaction interaction | **Orthogonal** — compaction stays with `ContextManagerCapability`; no new fields. |
| Delta-retry interaction | RLM code output is free text via tools; delta-retry applies to structured output extraction only — no conflict expected; verify in Phase 1 tests. |
| Persistent RLM conversations | Not needed — KNOX agents are single-turn; REPL state lives per `source_id` for the run. |
| Security of code execution | **Accepted as-is** (decision): REPL chdirs to a temp dir; `eval`/`exec`/`compile`/`globals` blocked; `open`/`__import__` allowed. Threat model: the code-writing model is the same trusted model; KNOX input is semi-trusted. Revisit if untrusted third-party content ever flows into agent deps. |
| Token tracking | `KnoxModel` wrap on main + sub-model; `on_usage_update` → `UsageTracker` (existing). |
| Error handling | REPL errors return as tool output (package behavior) → model self-corrects; agent-level failures still wrapped by existing `LLMOutputError` handling. |
| Recursion missing upstream | **We implement it** (Phase 2) per issue #4's proposed fix; Phase 1 ships non-recursive (still the bulk of the value). |

## 7. Success criteria

- [x] `uv add knox[rlm]` installs; package imports against pydantic-ai 2.30.0
- [x] `rlm=false` default: zero change to existing agents (all unit tests green)
- [x] Query agent with `rlm="default"` answers over a corpus > context window
      without the document text appearing in the message history (verified
      live via `scripts/verify_rlm.py` against Qwen3.5-2B-MTP)
- [x] `llm_query` routes to a KNOX-configured sub-model (Ollama verified) and
      usage is recorded (unit-tested via KnoxModel)
- [x] Phase 2: `rlm_query` recursion depth-limited with node budget + cycle
      detection; falls back to `llm_query` at max depth
- [x] Compaction (`context_manager`) and RLM both enabled without conflict
- [x] Ruff + `ty check src/knox` pass (0 new diagnostics)

## 8. Out of scope (say so if asked)

- Sandbox hardening (decision: not needed now).
- RLM for extraction (chunks already bounded).
- Replacing `ContextManagerCapability` — RLM is an addition, not a replacement.
