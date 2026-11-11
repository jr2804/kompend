# DOX — kompend.acp

<!-- on entry: loaded when agent enters src/kompend/acp -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

ACP (Agent Client Protocol) adapter — serve kompend to editors and AI agents over
stdio (`kompend agent`) with TOML-defined conversational agents. The session's
model comes from the workspace config (`compile.llm.default_model`); an ACP
client can override the *endpoint* at runtime through `providers/set`.

## Ownership

- `src/kompend/acp/` — agent definitions, registry, tool set, model selection,
  provider store and wire routes, the ACP agent assembly, shipped agent TOML

## Architecture

| Module | Contents |
| ------ | -------- |
| `definitions.py` | `AgentDefinition`, `AgentTask`, `AgentLLMConfig`, `AgentPromptConfig` — the TOML schema |
| `registry.py` | `AgentRegistry` — discovery (built-in / vault / explicit), `active_for` resolution |
| `tools.py` | Native PydanticAI `FunctionToolset` wrapping the MCP server functions; `base_dir` injected by closure |
| `harness_adapter.py` | `build_kompend_agent`, `kompend_session_config` — per-session agent factory |
| `model_selection.py` | `ConfigModelSelector`, `NO_MODEL_MESSAGE` — configured-model fallback |
| `endpoint_model.py` | `EndpointOverrideSelector`, `build_endpoint_model` — prefer the injected endpoint |
| `provider_store.py` | `ProviderCredentials` — process-wide `providers/set` store; `LlmApiType`, `ProviderConfig` |
| `provider_wire.py` | `ProviderWire` handlers, `build_provider_router`, `serve_agent` — the JSON-RPC serving path |
| `connection_client.py` | `ConnectionClient` — the `acp.Client` the adapter streams through |
| `acp_agent.py` | `KompendACPAgent`, `build_acp_agent`, `serve`, `serve_sync` — the shipped entry point |
| `builtin/*.toml` | Shipped agents: `generic`, `compile`, `query`, `capture` |

Protocol flow: `initialize` (advertises `providers`) → `providers/list` →
`providers/set` (endpoint stored, next prompt uses it) → `session/new` →
`session/prompt` → `providers/disable` (back to the configured model).
`/kompend <command>` prompts dispatch the corresponding CLI command; free text
goes to the active conversational agent.

## Local Contracts

- Import: `from kompend.acp.registry import AgentRegistry`,
  `from kompend.acp.definitions import AgentDefinition`.
- Agent definitions live in TOML, never in Python source (see `AGENTS.md`:
  "no text in code"). Built-ins ship in `builtin/*.toml`; user agents go in
  `<wiki>/.kompend/agents/*.toml`. The active agent per task is stored in
  `config.toml` under `[agents]`.
- `providers/*` is served through public SDK API only: `serve` builds its own
  `acp.Connection` over `build_provider_router`, which is `build_agent_router`
  plus the three provider routes. `AgentSideConnection` is `@final` and installs
  its router internally, so it cannot carry the routes; the adapter gets its
  `acp.Client` from `ConnectionClient`, which mirrors the SDK translation layer.
  No private attribute of `acp` or `pydantic_ai_harness` may be touched.
- The `providers` capability in `initialize` must stay in lockstep with the
  served routes: `KompendACPAgent.initialize` advertises it because `serve`
  always registers them.
- **The agent carries no model.** `build_kompend_agent` passes `model=None` plus
  a `SelectModel` capability; `ConfigModelSelector` resolves
  `compile.llm.default_model` once per session (from the session's `cwd`, so it
  outranks the process-cwd fallback). A session with nothing configured starts
  fine and fails its first prompt with `NO_MODEL_MESSAGE` (a re-export of
  `kompend.llm.messages.NO_MODEL_CONFIGURED_MESSAGE`) raised as
  `acp.RequestError` — never pydantic-ai's `UserError`: the connection forwards
  `RequestError` verbatim.
- An injected endpoint replaces the model's *provider*, not its name: the ACP
  payload carries `apiType`/`baseUrl`/headers and no model id, so the model id
  still comes from `compile.llm.default_model`. `EndpointOverrideSelector` rebuilds
  only when the store changes; `build_endpoint_model` supports `anthropic`,
  `openai`, `azure` and rejects the rest.
- `ProviderCredentials` is thread-safe (`threading.Lock`) and deliberately does
  **not** write `KOMP_LLM_*` into `os.environ` — nothing in the pipeline reads
  those keys, so an injected endpoint reaches the agent only through the store.
- Sessions keep their history across turns (`SessionState.history`); a turn is
  not committed when it is cancelled.
- Tools are the MCP server's functions — add a tool by adding an MCP tool, not a
  parallel implementation.
- **Sessions cache their prefix.** A conversational session resends its whole
  history every turn, so `kompend_session_config` attaches pydantic-ai's
  `Caching()` capability unless `compile.llm.prompt_cache` is off. The agents in
  `kompend.agents` use the equivalent `cache` model setting instead.

## Work Guidance

- New built-in agent: add TOML under `builtin/` with a name, description, system
  prompt, and tool list; no Python change is required.
- New provider method: add a `ProviderWire` handler and one
  `router.route_request(...)` line in `build_provider_router`, keep the
  `providers` capability honest, and cover it in
  `tests/unit/test_acp_provider_wire.py`.
- New endpoint apiType: extend `build_endpoint_model` and keep its error message
  listing the supported set.
- Reaching the pipeline from a tool: call the core function or the CLI helper
  that the MCP tool calls — never shell out.

## Verification

- `uv run pytest tests/unit/test_agent.py tests/unit/test_agent_definitions.py
  tests/unit/test_acp_model_selection.py tests/unit/test_acp_session_caching.py
  tests/unit/test_acp_provider_wire.py -q` — provider store and wire routes,
  agent install/remove, definitions, registry, model selection, session
  capabilities.
- End to end: `.kompend/tmp/smoke_providers_wire.py` (real `kompend agent` over
  stdio JSON-RPC against a local OpenAI stand-in) shows `providers/set` then a
  prompt reaching the injected `baseUrl` with the injected auth header.
