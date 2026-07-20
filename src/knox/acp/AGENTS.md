# DOX — knox.acp

<!-- on entry: loaded when agent enters src/knox/acp -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

ACP (Agent Client Protocol) — define, register, and load TOML-defined
conversational agents for KNOX compilation stages. Enables user-customisable
agents without writing Python code.

## Ownership

- ``src/knox/acp/`` — ACP agent definitions, registry, provider store, harness

## Architecture

| Module | Contents |
|--------|----------|
| ``definitions.py`` | ``AgentDefinition``, ``AgentTask``, ``AgentLLMConfig``, ``AgentPromptConfig`` |
| ``registry.py`` | ``AgentRegistry`` — discovery (builtin/vault/explicit), ``active_for`` resolution |
| ``tools.py`` | ACP tool adapter — wraps agents as callable tools |
| ``harness_adapter.py`` | Harness integration — adapts ACP agents for pipeline use |
| ``provider_store.py`` | Provider model resolution for ACP agents |
| ``builtin/`` | Shipped agent TOML definitions |

## Rules

- Import: ``from knox.acp.registry import AgentRegistry``,
  ``from knox.acp.definitions import AgentDefinition``.
- Agent definitions live in TOML, never in Python source (see parent:
  "no text in code").
- Built-in agents ship in ``builtin/*.toml``; user agents go in
  ``<wiki>/.knox/agents/*.toml``.
