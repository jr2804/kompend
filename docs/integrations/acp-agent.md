# Editor Integration: ACP Agent

KOMPEND can run as an **ACP (Agent Client Protocol) agent** that editors like
opencode and Zed can connect to. This gives the editor's built-in AI agent
access to KOMPEND's compilation, query, and capture capabilities without
sharing API keys — the editor injects the LLM endpoint via ``providers/set``.

## How it works

```text
Editor ── ACP stdio ──→ kompend agent
                          │
                          ├── /kompend compile  → wiki_compile tool
                          ├── /kompend query    → wiki_query tool
                          ├── /kompend capture  → wiki_capture tool
                          └── free text      → conversational agent loop
```

1. The editor starts ``kompend agent`` as a subprocess
2. KOMPEND advertises its capabilities (``providers/set``, slash commands)
3. The editor calls ``providers/set`` to inject the LLM endpoint
4. The user types a prompt → KOMPEND routes it through a conversational agent
   that can call tools (compile, query, capture) as needed

## opencode integration

Add to your opencode configuration (``~/.config/opencode/config.json`` or
project-level ``.opencode.json``):

```json
{
  "agents": {
    "kompend": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/your/vault", "kompend", "agent"],
      "cwd": "/path/to/your/vault"
    }
  }
}
```

Replace ``/path/to/your/vault`` with the path to your KOMPEND vault (the
directory containing ``.kompend/`` and ``sources/``).

### With MCP companion server

To also expose KOMPEND's MCP tools to other AI agents in the editor, add
``--mcp-port``:

```json
{
  "agents": {
    "kompend": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/your/vault", "kompend", "agent", "--mcp-port", "8000"],
      "cwd": "/path/to/your/vault"
    }
  }
}
```

## Zed integration

Add to your Zed settings (``~/.config/zed/settings.json``):

```json
{
  "agent_servers": {
    "kompend": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/your/vault", "kompend", "agent"],
      "cwd": "/path/to/your/vault"
    }
  }
}
```

## Env-var fallback (no providers/set)

If your editor does not support the ``providers/set`` ACP extension, KOMPEND
falls back to its normal config precedence:

1. Environment variables
2. ``config.toml`` in the vault
3. Built-in defaults

Set these environment variables before starting the agent:

```bash
export KOMP_LLM_DEFAULT_MODEL="ollama-cloud/deepseek-v4-flash"
export KOMP_LLM_API_BASE="https://api.ollama.cloud/v1"
export KOMP_LLM_API_KEY="sk-..."
```

Or configure them permanently in ``config.toml``:

```toml
[compile.llm]
default_model = "ollama-cloud/deepseek-v4-flash"
```

## Available slash commands

Once connected, you can use these commands in the editor's chat:

| Command | Description |
|---------|-------------|
| ``/kompend compile`` | Compile sources into a wiki |
| ``/kompend query <question>`` | Query the compiled wiki |
| ``/kompend capture <note>`` | Capture a note into the knowledge base |
| ``/kompend status`` | Show vault health and cache stats |
| ``/kompend help`` | Show available commands |

## Selecting an agent

KOMPEND ships with several built-in conversational agents. Use ``kompend agents``
to manage them:

```bash
# List available agents
kompend agents list

# Set the active agent for compilation
kompend agents use compile compiler

# Install a custom agent
kompend agents install path/to/my-agent.toml

# Use an explicit agent file when starting the agent
kompend agent --agent-file path/to/my-agent.toml
```

## Troubleshooting

**Agent doesn't start:**

- Ensure ``uv`` is installed and on PATH
- Ensure the vault directory exists and has a ``.kompend/`` directory
- Run ``kompend doctor`` to check the vault health

**LLM not available:**

- The editor must call ``providers/set`` with a valid LLM endpoint
- Or set ``KOMP_LLM_DEFAULT_MODEL`` / ``KOMP_LLM_API_KEY`` env vars
- Run ``kompend doctor`` to verify LLM connectivity

**Slash commands not working:**

- Ensure the prompt starts with ``/kompend`` (with a space after ``kompend``)
- Check that the agent is running (``kompend agent --help`` for options)
