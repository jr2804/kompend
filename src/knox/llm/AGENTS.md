# DOX — knox.llm

## Purpose

LLM client layer. All LLM calls go through `LLMClient` — never import litellm directly in business modules.

Custom litellm providers live in `knox.llm.providers/`.

## Format module

`knox.llm.format` is the single source of truth for LLM communication formats.
It owns `salvage_json` (moved from `client.py`), `encode_for_llm`.

Supported formats (`LlmDataFormat` enum, config key `compile.llm.data_format`):

| Value | Token reduction | Package | Scope |
|-------|----------------|---------|-------|
| `JSON` | baseline | stdlib | input + output |
| `TOON` | 30–60% | `toon_format` (core dep) | input only |
| `ISON` | 30–70% | `ison-py` → `ison_parser` (core dep) | input only |

**Input only.** TOON and ISON apply to context data serialized into prompt
templates (`encode_for_llm`). LLM output stays JSON — the
`response_format: json_object` API constraint is preserved so providers
enforce valid JSON at the protocol level. Models are not reliably
trained on TOON/ISON output syntax, and embedding format specs in every
call is unnecessary overhead.

**Call-site rule:** Use `encode_for_llm(data, config.llm.data_format)` instead
of `json.dumps(data)` for any Python data injected as a Jinja2 template
variable in an LLM call. Leave all `response_format=JSON_RESPONSE_FORMAT`
and `salvage_json()` usages unchanged.

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
  impossible. Optional-feature probe, logged at DEBUG (it is now an
  intermediate step — `models.dev` or the fallback give the final word, and
  the single WARNING lives in the fallback step only).

Do not "clean up" these broad catches — they are required by the upstream
APIs. If a narrow base class becomes available (e.g. litellm ships
`LiteLLMException`), narrow then.

## Context-window discovery

`_discover_context_window` in `client.py` resolves a model's token limits
through a 5-step first-hit chain (each step fills only the values still
missing): litellm static tables → Ollama runtime `/api/show` →
`models.dev` catalog → fallback (clamped, one WARNING) → user cap.

`models_dev.py` is the step that resolves **custom-provider models** (e.g.
Ollama Cloud) absent from litellm's static DB. It fetches the
`models.dev/catalog.json` and caches it in-memory (per process) **and** on
disk (`~/.config/knox/cache/models-dev-catalog.json`, shared across
invocations, 7-day TTL). Matching: exact key → provider-local stem →
normalized base + size fuzzy match. Force-refresh: `KNOX_MODELS_CACHE_TTL=0`
or call `refresh_catalog()`. Network failures are caught narrowly
(`httpx.HTTPError, ValueError`) and logged at DEBUG — the caller falls
through to its fallback path. Never raises.

## Verification

- `uv run pytest tests/unit/test_providers.py tests/unit/test_llm_client.py tests/unit/test_models_dev.py -x` — provider, client, and context-window tests pass
- `uv run python -c "import knox.llm.providers"` — auto-registration fires without error

---

## Child DOX Index

| Path | Scope |
|------|-------|
| `providers/AGENTS.md` | Custom litellm provider rules, patterns, registration, testing |
