# Editor Integration: ACP Agent

kompend can run as an **ACP (Agent Client Protocol) agent** that editors like
opencode and Zed can connect to. This gives the editor's built-in AI agent
access to kompend's compilation, query, and capture capabilities without
sharing API keys — the model comes from kompend's own configuration (see [Configuring the model](#configuring-the-model)).

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
2. kompend advertises its capabilities (slash commands)
3. The editor creates a session; the LLM model is resolved per prompt from
   kompend's configuration
4. The user types a prompt → kompend routes it through a conversational agent
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

Replace ``/path/to/your/vault`` with the path to your kompend vault (the
directory containing ``.kompend/`` and ``sources/``).

### With MCP companion server

To also expose kompend's MCP tools to other AI agents in the editor, add
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

## Configuring the model

The ACP agent carries no model of its own: every prompt resolves the model from
kompend's normal config precedence — environment variables, then ``config.toml``
in the vault, then the built-in defaults.

Point kompend at a model before starting the agent:

```bash
export KOMP_LLM_DEFAULT_MODEL="ollama-cloud/deepseek-v4-flash"
```

Or configure it permanently in ``config.toml``:

```toml
[compile.llm]
default_model = "ollama-cloud/deepseek-v4-flash"
```

The endpoint follows the provider the model id names — a native provider, or a
``[providers.<name>]`` block with a ``base_url``:

```toml
[providers.ollama-cloud]
base_url = "https://ollama.com/v1"
api_key = "sk-..."
```

Without a model the agent still starts; a prompt then fails with
``No LLM model configured. Set compile.llm.default_model or run 'kompend config set'.``

> ``providers/set`` — letting the editor inject the endpoint at runtime — is not
> served yet; the editor cannot override the endpoint from the wire.

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

kompend ships with several built-in conversational agents. Use ``kompend agents``
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

- Set ``compile.llm.default_model`` (or the ``KOMP_LLM_DEFAULT_MODEL`` env var)
- Check the provider block the model id names (``[providers.<name>]`` with a
  ``base_url``)
- Run ``kompend doctor`` to verify LLM connectivity

**Slash commands not working:**

- Ensure the prompt starts with ``/kompend`` (with a space after ``kompend``)
- Check that the agent is running (``kompend agent --help`` for options)
