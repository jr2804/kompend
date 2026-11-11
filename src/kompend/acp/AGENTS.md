# DOX — kompend.acp

<!-- on entry: loaded when agent enters src/kompend/acp -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

ACP (Agent Client Protocol) adapter — serve kompend to editors and AI agents over
stdio (`kompend agent`) with TOML-defined conversational agents. The session's
model comes from the workspace config (`compile.llm.default_model`). The client
cannot inject an endpoint: `providers/set` is not served yet — see
docs/integrations/acp-agent.md for what works today.

## Ownership

- `src/kompend/acp/` — agent definitions, registry, tool set, model selection,
  provider store, harness adapter, shipped agent TOML

## Architecture

| Module | Contents |
| ------ | -------- |
| `definitions.py` | `AgentDefinition`, `AgentTask`, `AgentLLMConfig`, `AgentPromptConfig` — the TOML schema |
| `registry.py` | `AgentRegistry` — discovery (built-in / vault / explicit), `active_for` resolution |
| `tools.py` | Native PydanticAI `FunctionToolset` wrapping the MCP server functions; `base_dir` injected by closure |
| `harness_adapter.py` | `build_kompend_agent`, `kompend_session_config` — per-session agent factory for `run_acp_stdio_sync` |
| `model_selection.py` | `ConfigModelSelector`, `NO_MODEL_MESSAGE` — the per-step model selector the session agent carries |
| `provider_store.py` | Thread-safe store for a `providers/set` endpoint; not reachable from the wire (see Local Contracts) |
| `builtin/*.toml` | Shipped agents: `generic`, `compile`, `query`, `capture` |

Protocol flow today: `initialize` → `session/new` → prompts. The protocol itself
is handled by `pydantic-ai-harness` (`run_acp_stdio_sync`), not by this package.
`/kompend <command>` prompts dispatch the corresponding CLI command; free text
goes to the active conversational agent.

## Local Contracts

- Import: `from kompend.acp.registry import AgentRegistry`,
  `from kompend.acp.definitions import AgentDefinition`.
- Agent definitions live in TOML, never in Python source (see `AGENTS.md`:
  "no text in code"). Built-ins ship in `builtin/*.toml`; user agents go in
  `<wiki>/.kompend/agents/*.toml`. The active agent per task is stored in
  `config.toml` under `[agents]`.
- `providers/set` is an RFD, not stable ACP, and is **not served**: the harness
  advertises no `providers` capability and its router forwards only
  `_`-prefixed methods to `ext_method`, so the route cannot be added through
  public API. Wire it as its own ticket, not inside an unrelated change.
- **The agent carries no model.** `build_kompend_agent` passes `model=None` plus
  a `SelectModel` capability; `ConfigModelSelector` resolves
  `compile.llm.default_model` once per session (from the session's `cwd`, so it
  outranks the process-cwd fallback) and reuses it for every later step. A
  session with nothing configured starts fine and fails its first prompt with
  `NO_MODEL_MESSAGE` (a re-export of `kompend.llm.messages.NO_MODEL_CONFIGURED_MESSAGE`)
  raised as `acp.RequestError` — never pydantic-ai's `UserError`: the
  connection forwards `RequestError` verbatim.
- Sessions keep their history across turns (`SessionState.history`); a turn is
  not committed when it is cancelled.
- Tools are the MCP server's functions — add a tool by adding an MCP tool, not a
  parallel implementation.
- `ProviderCredentials` is thread-safe (`threading.Lock`); nothing constructs it
  today, so its env writes are dead code until `providers/*` is wired.
- **Sessions cache their prefix.** A conversational session resends its whole
  history every turn, so `kompend_session_config` attaches pydantic-ai's
  `Caching()` capability unless `compile.llm.prompt_cache` is off. The agents in
  `kompend.agents` use the equivalent `cache` model setting instead.

## Work Guidance

- New built-in agent: add TOML under `builtin/` with a name, description, system
  prompt, and tool list; no Python change is required.
- New provider method (`providers/*`): not possible through public harness API
  yet — see Local Contracts; a change there needs the ticket, not a private hop
  into `acp/router.py`.
- Reaching the pipeline from a tool: call the core function or the CLI helper
  that the MCP tool calls — never shell out.

## Verification

- `uv run pytest tests/unit/test_agent.py tests/unit/test_agent_definitions.py
  tests/unit/test_acp_model_selection.py tests/unit/test_acp_session_caching.py -q`
  — provider store, credentials, agent install/remove, definitions, registry,
  model selection, session capabilities. The protocol itself lives in
  `pydantic-ai-harness` and is not covered here.
