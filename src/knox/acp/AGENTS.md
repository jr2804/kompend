# DOX — knox.acp

<!-- on entry: loaded when agent enters src/knox/acp -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

ACP (Agent Client Protocol) adapter — serve KNOX to editors and AI agents over
stdio (`knox agent`) with TOML-defined conversational agents. The client injects
its own LLM endpoint through `providers/set`, so KNOX carries no credentials of
its own. User-facing guide: `docs/integrations/acp-agent.md`.

## Ownership

- `src/knox/acp/` — agent definitions, registry, tool set, provider store,
  harness adapter, shipped agent TOML

## Architecture

| Module | Contents |
| ------ | -------- |
| `definitions.py` | `AgentDefinition`, `AgentTask`, `AgentLLMConfig`, `AgentPromptConfig` — the TOML schema |
| `registry.py` | `AgentRegistry` — discovery (built-in / vault / explicit), `active_for` resolution |
| `tools.py` | Native PydanticAI `FunctionToolset` wrapping the MCP server functions; `base_dir` injected by closure |
| `harness_adapter.py` | `build_knox_agent`, `knox_session_config` — per-session agent factory for `run_acp_stdio_sync` |
| `provider_store.py` | Thread-safe store for the `providers/set` endpoint; reconfigures env-var-based model resolution |
| `builtin/*.toml` | Shipped agents: `generic`, `compile`, `query`, `capture` |

Protocol flow: `initialize` (advertises `providers` capability) → `providers/list`
→ `providers/set` (endpoint stored, model reconfigured) → `session/new` →
prompts. `/knox <command>` prompts dispatch the corresponding CLI command; free
text goes to the active conversational agent. The protocol itself is handled by
`pydantic-ai-harness` (`run_acp_stdio_sync`), not by this package.

## Local Contracts

- Import: `from knox.acp.registry import AgentRegistry`,
  `from knox.acp.definitions import AgentDefinition`.
- Agent definitions live in TOML, never in Python source (see `AGENTS.md`:
  "no text in code"). Built-ins ship in `builtin/*.toml`; user agents go in
  `<wiki>/.knox/agents/*.toml`. The active agent per task is stored in
  `config.toml` under `[agents]`.
- `providers/set` is an RFD, not stable ACP. When it is never called, model
  resolution falls back to env vars → `config.toml` → `DEFAULTS`.
- Sessions are stateless: each prompt turn is independent, with no accumulated
  conversation history.
- Tools are the MCP server's functions — add a tool by adding an MCP tool, not a
  parallel implementation.
- `ProviderCredentials` is thread-safe (`threading.Lock`) and restores the
  previous environment on `disable()`.

## Work Guidance

- New built-in agent: add TOML under `builtin/` with a name, description, system
  prompt, and tool list; no Python change is required.
- New provider method (`providers/*`): extend the `ext_method` handler in the
  harness adapter and `provider_store.py`.
- Reaching the pipeline from a tool: call the core function or the CLI helper
  that the MCP tool calls — never shell out.

## Verification

- `uv run pytest tests/unit/test_agent.py tests/unit/test_agent_definitions.py -q`
  — provider store, credentials, agent install/remove, definitions, registry.
  The protocol itself lives in `pydantic-ai-harness` and is not covered here.
