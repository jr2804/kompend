"""Build native pydantic-ai Model instances and resolve model metadata.

kompend uses the ``provider/model`` convention (e.g. ``ollama/qwen3``,
``openai/gpt-5.2``, ``claudinio/claude-4``). This module translates such
a string into a native pydantic-ai :class:`~pydantic_ai.models.Model`.

Resolution strategy (first match wins):

1. **Native pydantic-ai provider** — :func:`pydantic_ai.providers.infer_provider_class`
   is the smart dispatcher: it returns the matching ``Provider`` subclass
   for every provider pydantic-ai ships a class for (ollama, openai,
   anthropic, google, deepseek, groq, mistral, cohere, bedrock, …). We
   construct that provider, then :func:`~pydantic_ai.models.infer_model`
   builds the typed Model subclass (``OllamaModel``, ``AnthropicModel``,
   ``OpenAIResponsesModel``, …). No hard-coded string comparison on our
   side — pydantic-ai's dispatch resolves both the provider class and
   the model class.

2. **models-dev catalog** — for providers pydantic-ai doesn't ship a
   class for (custom OpenAI-compatible mirrors like ``claudinio``,
   ``opencode``), the catalog's ``api`` URL and advertised ``env`` var
   build a generic :class:`~pydantic_ai.models.openai.OpenAIChatModel`
   on an :class:`~pydantic_ai.providers.openai.OpenAIProvider`.

3. Neither → ``ValueError``.

API keys and endpoints are **never** injected via environment mutation.
Defaults are passed as constructor arguments; user overrides are read
from the env vars the provider/catalog advertises.
"""

from __future__ import annotations

import os
import pkgutil
import re

import ollama
import pydantic_ai.providers
from models_dev import Provider, get_model_by_id, get_provider
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

# ── public API ─────────────────────────────────────────────────────────
from kompend.llm.user_providers import _user_providers, set_user_providers  # noqa: E402  (re-export for build_model)
from kompend.logging import get_logger

# Default base URL for local Ollama — ``ollama`` is not in the models-dev
# catalog, and OllamaProvider has no built-in default (it raises UserError
# when neither ``base_url`` nor ``OLLAMA_BASE_URL`` is given). pydantic-ai's
# OllamaModel uses the OpenAI-compatible endpoint, so ``/v1`` is required in
# the base URL (the OpenAI SDK appends ``/chat/completions`` to it).
OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

#: Ollama's server-wide default context length. Renamed from ``OLLAMA_NUM_CTX``
#: (the old name is gone from the server: ollama 0.35.0 has zero references to
#: it, one to this name), so reading the old one silently did nothing.
OLLAMA_CONTEXT_LENGTH_ENV = "OLLAMA_CONTEXT_LENGTH"

# Placeholder satisfies the OpenAI SDK's non-empty api_key requirement for
# local / keyless OpenAI-compatible endpoints.
_NO_KEY_PLACEHOLDER = "api-key-not-set"

__all__ = [
    "build_model",
    "is_known_provider",
    "model_supports_vision",
    "resolve_model_limits",
    "resolve_vlm_model",
    "set_user_providers",
]


#: Vision capability is cached per model string — the Ollama probe is a
#: live HTTP call to the local server and must not run on every pipeline
#: construction. The catalog path is cheap but the cache makes the whole
#: lookup idempotent.
_VISION_CACHE: dict[str, bool] = {}

logger = get_logger(__name__)


# ── path 1: native pydantic-ai provider ────────────────────────────────


class _NotNativeProvider(Exception):
    """Provider is not natively known to pydantic-ai."""


# ── path 2: models-dev catalog (OpenAI-compatible custom provider) ─────


class _NotInCatalog(Exception):
    """Provider is not in the models-dev catalog or has no API endpoint."""


def resolve_vlm_model(default_model: str | None, vlm_model: str | None) -> str | None:
    """Effective VLM model: explicit VLM wins, else default model if vision-capable.

    ``vlm_model`` is the configured VLM (``compile.llm.vlm_model``); when it
    is empty, the default model is reused as the captioning model only if
    :func:`model_supports_vision` confirms image input. Returns ``None`` when
    neither is usable — callers then skip VLM enrichment as before.
    """
    if vlm_model:
        return vlm_model
    if default_model and model_supports_vision(default_model):
        return default_model
    return None


def model_supports_vision(model_string: str) -> bool:
    """True when *model_string* can consume image input.

    Vision is a property of the *model entry* (its modality signature), not
    of the api_type/transport: the same OpenAI-compatible endpoint will
    accept images for a vision model and reject them for a text-only one.
    How the signature is discovered depends on the provider:

    - **models-dev catalog providers** (openai, google, anthropic, ...): the
      catalog entry's ``modalities.input`` lists ``"image"``.
    - **Ollama**: a live ``ollama.show()`` probe returns the server-side
      ``capabilities`` list, which includes ``"vision"`` for multimodal
      models (e.g. llava, qwen2.5-vl). The probe is cached per model string
      and a dead/absent server is treated as *no vision* (conservative: a
      wrong fallback would send images to a text-only model).
    - **Unknown providers / user-defined endpoints**: no static metadata,
      returns ``False`` unless the model is in the catalog.

    Used by :func:`resolve_vlm_model` to fall back to the default model when
    no explicit VLM is configured.
    """
    if model_string in _VISION_CACHE:
        return _VISION_CACHE[model_string]
    try:
        provider_id, model_name = _split(model_string)
    except ValueError:
        return False

    vision = _vision_from_catalog(provider_id, model_name)
    if vision is None and provider_id == "ollama":
        vision = _vision_from_ollama(model_name)
    _VISION_CACHE[model_string] = bool(vision)
    return bool(vision)


def _vision_from_catalog(provider_id: str, model_name: str) -> bool | None:
    """Vision flag from the models-dev catalog; ``None`` when not listed."""
    try:
        model = get_model_by_id(provider_id, model_name)
    except (KeyError, ValueError, TypeError):
        return None
    modalities = getattr(model, "modalities", None)
    if modalities is None or modalities.input is None:
        return None
    return "image" in modalities.input


def _vision_from_ollama(model_name: str) -> bool | None:
    """Vision flag via live ``ollama.show``; ``None`` on probe failure."""
    try:
        caps = ollama.show(model_name).capabilities
        return bool(caps and "vision" in caps)
    except Exception:
        logger.debug("ollama.show() vision probe failed for %s", model_name, exc_info=True)
        return None


def build_model(model_string: str) -> Model:
    """Build a native pydantic-ai Model from a kompend ``provider/model`` string."""
    provider_id, model_name = _split(model_string)

    # 1. Native pydantic-ai provider?
    try:
        return _build_native(provider_id, model_name)
    except _NotNativeProvider as exc:
        # Fall through to (2) user-defined providers.
        logger.debug("provider %r not native (%s); trying user-defined providers", provider_id, exc)

    # 2. User-defined [providers] from kompend.toml — explicit user config beats
    # the implicit catalog. A user redefining a catalog-known provider id
    # (e.g. minimax via an anthropic-protocol proxy) must win, otherwise the
    # catalog shadows their base_url/api_type and requests hit the wrong
    # endpoint.
    # Override-only blocks (no base_url, e.g. [providers.ollama] with just
    # max_connections) do not build models — fall through to the catalog.
    if provider_id in _user_providers and _user_providers[provider_id].get("base_url"):
        return _build_user_provider(provider_id, model_name)

    # 3. models-dev catalog (OpenAI-compatible custom provider)?
    try:
        return _build_catalog(provider_id, model_name)
    except _NotInCatalog as exc:
        logger.debug("provider %r not in catalog (%s)", provider_id, exc)

    # Build a helpful list of known providers for the error message.
    known = _discover_provider_ids()
    known_str = ", ".join(sorted(known))
    msg = (
        f"Unknown provider {provider_id!r}: not natively supported by "
        f"pydantic-ai and not in the models-dev catalog. "
        f"Model strings use the format 'provider/model' (e.g. 'ollama/{model_string}').\n"
        f"  Known providers: {known_str}\n"
        f"  Got: {model_string!r} -> provider={provider_id!r}, model={model_name!r}"
    )
    raise ValueError(msg)


def is_known_provider(provider_id: str) -> bool:
    """Whether *provider_id* resolves without a user-defined endpoint.

    True when pydantic-ai ships a native provider class for it, or the
    models-dev catalog has an entry. Distinguishes an override-only
    ``[providers.<name>]`` block (valid for a known provider — e.g.
    ``[providers.ollama]`` with only ``context_window``) from a typo'd
    provider id that can never build a model.
    """
    if provider_id in _discover_provider_ids():
        return True
    try:
        get_provider(provider_id)
    except (KeyError, ValueError):
        return False
    return True


def resolve_model_limits(model_id: str) -> tuple[int, int]:
    """Resolve ``(max_input, max_output)`` token limits from the models-dev catalog.

    Raises:
        ValueError: If *model_id* has no ``/`` or ``:`` separator.
        KeyError: If the provider or model is not in the catalog.
        LookupError: If the model entry has no limit data.
    """
    provider_id, model_name = _split(model_id)
    model = get_model_by_id(provider_id, model_name)  # raises KeyError

    limit = model.limit
    if limit is None or limit.context is None:
        raise LookupError(f"No limit data for {model_id!r}")

    return limit.context, limit.output or limit.context


# ── parsing ────────────────────────────────────────────────────────────


def _discover_provider_ids() -> set[str]:
    """Discover all native provider IDs by scanning pydantic_ai.providers submodules.

    No hard-coded provider list — the scan discovers whatever pydantic-ai
    ships in its installed version. ``sentence_transformers`` is excluded
    because it's an embedding-only backend, not an LLM provider.
    """
    return {m.name for m in pkgutil.iter_modules(pydantic_ai.providers.__path__) if not m.name.startswith("_") and m.name != "sentence_transformers"}


def _split(model_string: str) -> tuple[str, str]:
    """Split ``provider/model`` or ``provider:model`` into *(provider, model_name)*.

    The first ``/`` or ``:`` (whichever comes first) is the separator; the
    model name may itself contain slashes or colons.
    """
    slash = model_string.find("/")
    colon = model_string.find(":")
    candidates = [p for p in (slash, colon) if p > 0]
    if not candidates:
        raise ValueError(f"Invalid model string {model_string!r}. Use 'provider/model' or 'provider:model'.")
    pos = min(candidates)
    return model_string[:pos], model_string[pos + 1 :]


def _build_native(provider_id: str, model_name: str) -> Model:
    """Build via pydantic-ai's native provider/model dispatch.

    Raises :class:`_NotNativeProvider` when pydantic-ai has no class for
    *provider_id* (caller falls through to the catalog path).
    """
    try:
        provider_cls = infer_provider_class(provider_id)
    except ValueError as exc:
        raise _NotNativeProvider from exc

    # Ollama is a well-known local provider with no models-dev catalog entry
    # and no built-in default base_url. Provide the local default so kompend
    # works out of the box; OLLAMA_BASE_URL still wins when set.
    if provider_cls is OllamaProvider:
        return _build_ollama(model_name)

    provider = provider_cls()
    return infer_model(
        f"{provider_id}:{model_name}",
        provider_factory=lambda _name: provider,
    )


def _build_ollama(model_name: str) -> OllamaModel:
    """Construct an OllamaModel with a sane local default base_url.

    ``OLLAMA_BASE_URL`` (the env var pydantic-ai reads) wins when set;
    otherwise we fall back to the local default ``http://127.0.0.1:11434``
    so a stock local install works without any configuration.

    The model's context window is passed per-request via ``options.num_ctx``
    (Ollama's OpenAI-compatible endpoint honors it). Without this, Ollama
    runs at its server default (~4096) and silently truncates long prompts
    from the middle — the model sees prompt head/tail but not the content.
    """
    base_url = os.getenv("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL
    provider = OllamaProvider(base_url=base_url)
    ctx = _query_ollama_num_ctx(model_name)
    if ctx is None:
        return OllamaModel(model_name, provider=provider)
    settings: ModelSettings = ModelSettings(extra_body={"options": {"num_ctx": ctx}})
    return OllamaModel(model_name, provider=provider, settings=settings)


def _build_catalog(provider_id: str, model_name: str) -> Model:
    """Build a generic OpenAI-compatible model from models-dev catalog data.

    The catalog's ``api`` field is the OpenAI-compatible base URL; the
    advertised ``env`` var carries the API key. Raises
    :class:`_NotInCatalog` when the provider is absent or has no ``api``.
    """
    try:
        catalog = get_provider(provider_id)
    except KeyError as exc:
        raise _NotInCatalog from exc

    if not catalog.api:
        raise _NotInCatalog(f"Provider {provider_id!r} is in the catalog but has no API URL.")

    base_url = _normalize_base_url(catalog.api)
    api_key = _catalog_api_key(catalog)
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )


def _normalize_base_url(api_url: str) -> str:
    """Ensure the catalog ``api`` URL ends with ``/v1``."""
    base = api_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _catalog_api_key(catalog: Provider) -> str:
    """API key from the first advertised env var that is set, else a placeholder."""
    for env_var in catalog.env:
        value = os.environ.get(env_var)
        if value:
            return value
    return _NO_KEY_PLACEHOLDER


def _build_user_provider(provider_id: str, model_name: str) -> Model:
    """Build a model from a user-defined provider config.

    The provider config is registered via :func:`set_user_providers` and
    must have a ``base_url`` key. ``api_type`` selects the wire protocol:

    - ``openai-chat`` (default) — ``/v1/chat/completions``
    - ``openai-responses`` — ``/v1/responses``
    - ``anthropic`` — ``/v1/messages``
    """
    cfg = _user_providers[provider_id]
    base_url = cfg.get("base_url", "")
    if not base_url:
        raise ValueError(f"User-defined provider {provider_id!r} has no base_url configured.")

    raw_key = cfg.get("api_key")
    api_key = _resolve_env_ref(raw_key) if raw_key else _NO_KEY_PLACEHOLDER
    api_type = cfg.get("api_type") or "openai-chat"

    if api_type == "anthropic":
        provider = AnthropicProvider(base_url=base_url, api_key=api_key)
        return AnthropicModel(model_name, provider=provider)
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    if api_type == "openai-responses":
        return OpenAIResponsesModel(model_name, provider=provider)
    return OpenAIChatModel(model_name, provider=provider)


# ── path 3: user-defined provider (from [providers] in kompend.toml) ──────


def _resolve_env_ref(value: str) -> str:
    """Resolve ``${VAR_NAME}`` references in a string to their env var value.

    Returns the original string if the env var is unset or the pattern
    doesn't match.
    """

    def _replace(m: re.Match) -> str:
        var = m.group(1) or ""
        return os.environ.get(var, m.group(0))

    return re.sub(r"\$\{(\w+)\}", _replace, value)


def _query_ollama_num_ctx(model_name: str) -> int | None:
    """Query Ollama for the model's context window via ``ollama.show()``.

    Priority (most restrictive wins when multiple are set):
    1. ``OLLAMA_CONTEXT_LENGTH`` environment variable (server-wide default).
    2. ``context_length`` from ``ollama.ps()`` — the context the server
       actually loaded the model with. GGUF metadata is aspirational;
       Ollama caps the slot by available KV memory at load time. Without
       this step kompend sizes prompts against e.g. 262144 while the server
       serves 65536 — oversized prompts then get message-truncated
       server-side (dropping the user turn), which breaks chat templates
       that require a user message ("No user query found in messages",
       HTTP 500).
    3. ``parameters.num_ctx`` from ``ollama.show()`` - Modelfile override
       (what Ollama is actually configured to enforce at runtime).
    4. ``modelinfo`` from ``ollama.show()`` - native GGUF metadata.
       Prefers ``original_context_length`` (base context before RoPE scaling)
       over the scaled ``context_length`` when both are present.

    Returns ``None`` if none are available or the Ollama queries fail.
    """
    # 1. Global env var.
    env_ctx: int | None = None
    env_val = os.environ.get(OLLAMA_CONTEXT_LENGTH_ENV)
    if env_val:
        try:
            env_ctx = int(env_val)
            if env_ctx <= 0:
                env_ctx = None
        except ValueError:
            logger.warning("%s=%r is not a valid integer", OLLAMA_CONTEXT_LENGTH_ENV, env_val, exc_info=True)
    # 2. Context the server actually loaded the model with (ollama.ps()).
    loaded_ctx: int | None = None
    # 3. Modelfile parameters from ollama.show() - what Ollama enforces.
    param_ctx: int | None = None
    # 4. Native GGUF metadata from ollama.show().
    native_ctx: int | None = None
    try:
        for ps_model in ollama.ps().models:
            for ps_id in (ps_model.model, ps_model.name):
                if ps_id is not None and (ps_id == model_name or ps_id.split(":")[0] == model_name.split(":", maxsplit=1)[0]):
                    loaded_ctx = ps_model.context_length
                    break
            else:
                continue
            break
    except Exception:
        logger.debug("ollama.ps() failed for %s", model_name, exc_info=True)
    try:
        resp = ollama.show(model_name)
        # 2a. Parse parameters string for num_ctx (Modelfile override).
        params_str: str = resp.parameters or ""
        for line in params_str.split("\n"):
            stripped = line.strip()
            if stripped.startswith("num_ctx") and stripped[len("num_ctx") :].strip().isdigit():
                param_ctx = int(stripped[len("num_ctx") :].strip())
                break
        # 3a. Native GGUF metadata.
        model_info = resp.modelinfo or {}
        # First pass: look for original_context_length (base context before RoPE)
        for key, val in model_info.items():
            if key.endswith(".original_context_length") and isinstance(val, int) and val > 0:
                native_ctx = val
                break
        # Second pass: fall back to scaled context_length
        if native_ctx is None:
            for key, val in model_info.items():
                if key.endswith(".context_length") and isinstance(val, int) and val > 0:
                    native_ctx = val
                    break
    except Exception:
        logger.debug("ollama.show() failed for %s", model_name, exc_info=True)
    candidates = [c for c in (env_ctx, loaded_ctx, param_ctx, native_ctx) if c is not None]
    return min(candidates) if candidates else None
