# DOX — knox.llm

## Purpose

LLM client layer. All LLM calls go through `LLMClient` — never import litellm directly in business modules.

Custom litellm providers live in `knox.llm.providers/`.

## Format module

`knox.llm.format` is the single source of truth for LLM communication formats.
It owns `salvage_json` (moved from `client.py`), `encode_for_llm`, `decode_from_llm`,
`format_system_hint`, and `inject_format_hint`.

Supported formats (config key `compile.llm.data_format`):

| Value | Token reduction | Requires | Notes |
|-------|----------------|----------|-------|
| `"json"` | baseline | — | API-level `response_format: json_object` guarantee |
| `"toon"` | 30–60% | `uv add toon_format` | Best with frontier models for output |
| `"ison"` | 30–70% | `uv add ison-py` | Best with frontier models for output |

**Input benefit:** `encode_for_llm()` serializes context data (entity batches, concept
lists) in the chosen format before injection into prompt templates — reduces input
tokens, leaving more budget for model output and reducing truncation-related
malformed responses with small local models.

**Output behaviour:** When `data_format != "json"` and `response_format=JSON_RESPONSE_FORMAT`
is requested, `LLMClient.complete()` injects a format hint into the system message,
skips the API-level `response_format` constraint, and parses the response with
`decode_from_llm()`. Reliable output format switching requires capable models;
for local/small models the input-only benefit is safer.

**Call sites to update when adding new LLM context serialization:**
Use `encode_for_llm(data, config.llm.data_format)` instead of `json.dumps(data)`
for any Python data injected as a Jinja2 template variable for an LLM call.

## Provider submodule

`src/knox/llm/providers/` registers providers that litellm does not ship
natively. See `providers/AGENTS.md` for the full rules, patterns, and
test requirements. Brief summary:

- No hard-coded model names — use generic `<model>` in docs and `test-model` in tests.
- If the provider supports model discovery, use it; otherwise pass the user's model through as-is.
- Two patterns: **thin wrapper** (no discovery, known API format) or **discovery-based** (opencode.py style).
- Register in `_register_custom_providers()`, test all 4 methods, add tests to `tests/unit/test_providers.py`.

### Test hint

Provider registration is a side-effect of importing `knox.llm.providers`. Tests should mock the `httpx.get` call in `_discover_models` and set the relevant API key env var (or mock `_api_key`) before calling discovery methods.

### Error handling

Two broad `except Exception` sites in this package are **deliberate**, not
sloppy — see the "Error handling" rule in `src/knox/AGENTS.md`:

- `failover.py` `FailoverChain.run`: catches everything around the arbitrary
  `call(model)` and routes it through `classify_litellm_error` (guaranteed
  fallthrough → `TransientError`). Error-hierarchy trust boundary.
- `client.py` model-introspection: `litellm.get_model_info` raises bare
  `Exception` for unmapped models (its own API), so a narrow catch is
  impossible. Optional-feature probe, logged at WARNING.

Do not "clean up" these broad catches — they are required by the upstream
APIs. If a narrow base class becomes available (e.g. litellm ships
`LiteLLMException`), narrow then.

## Verification

- `uv run pytest tests/unit/test_providers.py -x` — provider tests pass
- `uv run python -c "import knox.llm.providers"` — auto-registration fires without error

---

## Child DOX Index

| Path | Scope |
|------|-------|
| `providers/AGENTS.md` | Custom litellm provider rules, patterns, registration, testing |
