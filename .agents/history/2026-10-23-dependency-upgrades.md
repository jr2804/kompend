# Dependency upgrades (2026-10-23)

## Why anthropic was pinned below 1.0 (2026-10-18, resolved 2026-10-23)

pydantic-ai 2.31.1's Anthropic provider hands the SDK an `httpx.AsyncClient`,
while `anthropic>=1.0` moved to `httpx2` and rejects it at construction
(`TypeError: Invalid http_client argument`), and pydantic-ai-slim 2.31.1 still
declares `anthropic>=0.108.0` with no upper bound. `[tool.uv]
constraint-dependencies = ["anthropic<1.0"]` held the SDK on the 0.x line
(`httpx`, resolves to 0.125.0). The pydantic-ai 2.60.0 bump made the pin
unnecessary — it declares `anthropic>=1.12.1` and ships the `httpx2` client —
so the constraint is gone and the lock carries `anthropic` 2.1.0.

### Why `[tool.uv]` overrides openai and typer (2026-10-23)

Two transitive caps held the graph on outdated versions, so
`override-dependencies = ["openai>=3.26.0", "typer>=0.27.3"]` forces the floors
kompend wants:

- `mineru<=4.0.12` (arrives via `kompend[mineru]` → `parsecraft[mineru]`)
  requires `openai>=1.70.0,<3`, while pydantic-ai-slim ≥ 2.55 (every release
  through 2.60.0) requires `openai>=3.26.0` for the openai/openrouter/zai/xai
  extras. Without the override the resolver silently backtracks all the way to
  pydantic-ai-slim 2.31.1 rather than erroring loudly.
- `docling-slim<=2.142.1` requires `typer>=0.19.0,<0.27.0`, which dated the CLI
  at typer 0.26.8; docling is only reached through parsecraft's Python API, not
  its CLI.

Revisit both when MinerU declares openai 3 support and docling-slim raises its
typer cap. MinerU's openai usage is OpenAI-compatible VLM-server clients only,
so the openai 3.x runtime risk is confined to `kompend[mineru]`.

### Why sentence-transformers support is gone (2026-10-23)

The `pydantic-ai[sentence-transformers]` extra pulled the full native ML stack
(torch + transformers) and dominated import/startup time, and kompend never
needed a local Python embedding backend - the same models are reachable through
pydantic-ai's remote providers (`ollama:`, `openai:`, ...). The dependency is
now `pydantic-ai-slim` (no st extra) and `to_pydantic_ai_embedding_string()`
rejects the `sentence-transformers` prefix (`_KNOWN_PROVIDERS`), so a bare
HuggingFace model name fails fast with the provider-prefix error. Docs point at
`provider:model` embedding strings only (e.g. `ollama:granite-embedding:278m`);
embedding stays opt-in with no default model (`compile.embedding.model = ""`).
