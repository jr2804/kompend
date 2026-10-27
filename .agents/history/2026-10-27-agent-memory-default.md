# Agent notebook on by default (2026-10-27)

## Why the defaults flipped

The agent notebook (`compile.llm.agent_memory`) shipped opt-in: commit
`a9f74350` (2026-10-22) made it opt-in "only where one agent sees many
outputs", because offering `read_memory` without a notebook can cost the run its
tool budget (an invented file name gets `ModelRetry`, and a spent budget kills
the run with the node left unsummarized). The vault backend
(`compile.llm.memory_store = "vault"`, 2026-10-26) then made the notebook
persist at `.kompend/agent-memory/` — but the feature stayed dark: a user had to
know two keys to get a notebook at all.

The risk that justified opt-in lives in the *small-model* case, not in the
notebook itself, so the notebook is on and the escape hatch stays one key away:
`compile.llm.agent_memory = false`.

## What landed

- `compile.llm.agent_memory` → `true`, `compile.llm.memory_store` → `"vault"`
  (`src/kompend/core/config/_configs.py`); `DEFAULTS` follows, since it is
  `KompendConfig().model_dump()`.
- `kompend.toml.example`, `docs/guides/llm-config.md`,
  `src/kompend/agents/AGENTS.md` and the `AgenticHarness` docstring carry the
  new defaults.
- Tests: `tests/unit/test_config.py` (both defaults),
  `tests/unit/test_agentic_harness.py` (the summariser offers the notebook with
  a default `LLMConfig`), `tests/unit/test_agentic_wiring.py` (the session
  store stays the explicit contrast case).

The harness flag stays per-agent (`AgenticHarness(memory=True)`): the config
decides *which stages* ask for a notebook, and the four stages that handle many
outputs in one session ask.

## Known gap (closed 2026-10-27)

The vault backend reached the summariser only; verification, remediation and
distillation kept the in-process store. The harness namespaces notebook paths
by agent name, so the stages never collided — wiring the other three to the
vault was plumbing, not a redesign. Closed by `notebook-vault-stages`
(bead `knox-lwvw`).
