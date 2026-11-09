"""LLM, embedding, and provider endpoint models (``[compile.llm]``, ``[providers]``)."""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import BeforeValidator, Field

from kompend.core.config.enums import ContextManagerPreset, ExtractionStrategy, LlmApiType, LlmMemoryStore, ReasoningEffort
from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar
from kompend.core.vocab import LlmDataFormat

_CONTEXT_OFF_WORDS: Final[frozenset[str]] = frozenset({"", "off", "false", "none", "no"})


class RlmConfig(_KompendBaseModel):
    """RLM (recursive research-loop) agent configuration (``[compile.llm.rlm]``).

    ``enable`` turns the capability on; the remaining fields set the
    execution budget directly.
    """

    enable: bool = False
    max_tool_calls: int = 50
    code_timeout: float = 60.0
    sub_model: str | None = None
    max_depth: int = 2
    node_budget: int | None = None


class LLMConfig(_KompendBaseModel):
    """Configuration for LLM models, timeouts, and fallback behavior."""

    default_model: str | None = Field(default=None, validation_alias=EnvVar.LLM_DEFAULT_MODEL)
    vlm_model: str | None = Field(default=None, validation_alias=EnvVar.VLM_MODEL)
    fallback_chain: list[str] = Field(default_factory=list, validation_alias=EnvVar.LLM_FALLBACK_CHAIN)
    timeout: float = Field(default=120.0, validation_alias=EnvVar.LLM_TIMEOUT)
    fallback_max_input: int = Field(default=32768, validation_alias=EnvVar.LLM_FALLBACK_MAX_INPUT)
    fallback_max_output: int = Field(default=4096, validation_alias=EnvVar.LLM_FALLBACK_MAX_OUTPUT)
    context_window: int | None = Field(
        default=None, validation_alias=EnvVar.LLM_CONTEXT_WINDOW, description="Context window length override/cap applied to every discovered model"
    )
    #: How many tokens of content one request may carry. It sizes a window, it
    #: never removes anything: a stage whose payload does not fit splits it into
    #: more requests (extraction batches, generation segments, summary and probe
    #: windows). Unset means the context window decides alone. Distinct from
    #: ``context_window``, which caps what the model will *accept*; this caps
    #: what it should be *given*, which is a property of the model, not of the
    #: provider.
    content_window_tokens: int | None = Field(
        default=None,
        validation_alias=EnvVar.LLM_CONTENT_WINDOW_TOKENS,
        description="Tokens of content one request may carry; larger payloads are split, never cropped",
    )
    data_format: LlmDataFormat = Field(default=LlmDataFormat.JSON, validation_alias=EnvVar.LLM_DATA_FORMAT)
    #: Whole-request retries after a *transient* failure (a timeout, a dropped
    #: connection, a 429/5xx). Distinct from ``tool_retries`` (a tool raising
    #: ``ModelRetry``) and from ``output_retries`` (a schema violation): those
    #: fix the request, a transient retry only repeats it. It lives on the agent
    #: base, so every stage - generation, lint, extraction, verification - obeys
    #: one policy instead of each loop inventing backoff.
    transient_retries: int = Field(default=2, validation_alias=EnvVar.LLM_TRANSIENT_RETRIES)
    #: First backoff in seconds for a transient retry; doubled per attempt.
    transient_backoff: float = Field(default=5.0, validation_alias=EnvVar.LLM_TRANSIENT_BACKOFF)
    #: Mark the conversation prefix for provider-side prompt caching. Only
    #: Anthropic-protocol models act on it (``anthropic_cache_messages``,
    #: the proxy-safe variant); a multi-pass agentic loop re-sends the same
    #: prefix, so this turns repeated prefill into cache reads. Disable it
    #: for a gateway that rejects the ``cache_control`` field.
    prompt_cache: bool = Field(default=True, validation_alias=EnvVar.LLM_PROMPT_CACHE)
    #: Output-validation retry budget for plain (single-shot) agents: how
    #: often a model answer may fail schema validation before the run fails.
    output_retries: int = Field(default=2, validation_alias=EnvVar.LLM_OUTPUT_RETRIES)
    output_retries_enabled: bool = Field(default=True, validation_alias=EnvVar.LLM_OUTPUT_RETRIES_ENABLED)
    #: Per-turn output-validation retry budget for tool-based (agentic) agents.
    #: Defaults higher than ``output_retries`` because agentic agents run
    #: multi-turn loops (fetch concepts → read segment → generate) where each
    #: turn can be invalidated; 2 retries is too tight for the typical
    #: small-model interaction.  Does NOT govern tool-call retries — those
    #: have their own budget (``tool_retries``).
    agentic_output_retries: int = Field(default=4, validation_alias=EnvVar.LLM_AGENTIC_OUTPUT_RETRIES)
    #: Tool-call retry budget for every agent (a tool that keeps raising
    #: ``ModelRetry`` — a missing file, an invalid argument).  Separate from
    #: the output-validation budgets so a flaky tool can be given more trials
    #: without also granting a model more schema-validation attempts.
    #: Defaults to 3: one hallucinated tool call must not abort a run — the
    #: ``ModelRetry`` feedback is usually enough for the model to correct
    #: itself.
    tool_retries: int = Field(default=3, validation_alias=EnvVar.LLM_TOOL_RETRIES)
    #: Offer the agentic memory capability (a per-agent notebook the model
    #: can write to and read from) to the stages where ONE agent instance
    #: handles MANY outputs — summarisation, verification, remediation and
    #: distillation. Stages that build a fresh agent per unit (generation,
    #: capture, query, indexing) gain nothing from a notebook: the memory
    #: dies with the agent.  ON by default: a stage that sees many outputs
    #: in one session is the only place a notebook pays off.
    agent_memory: bool = Field(default=True, validation_alias=EnvVar.LLM_AGENT_MEMORY)
    #: Backend for the agent notebook. ``vault`` (default) persists it under
    #: the vault's agent-memory directory: a ``MEMORY.md`` index per agent
    #: (injected into prompts, capped) plus topic files behind the memory
    #: tools. The agent writes that notebook itself, so it is not part of the
    #: summary cache key - editing it leaves cached summaries as they are.
    #: ``session`` keeps it in process - fresh per stage run.  The vault
    #: backend backs every notebook stage - summarisation, verification,
    #: remediation and distillation.
    memory_store: LlmMemoryStore = Field(default=LlmMemoryStore.VAULT, validation_alias=EnvVar.LLM_MEMORY_STORE)
    #: Rolling window (in output tokens) for the short-term tok/s rate.
    throughput_window_tokens: int = Field(default=1000, validation_alias=EnvVar.LLM_THROUGHPUT_WINDOW_TOKENS)
    delta_retry_enabled: bool = Field(default=True, validation_alias=EnvVar.LLM_DELTA_RETRY_ENABLED)
    batch_failure_threshold_ratio: float = Field(default=0.5, validation_alias=EnvVar.LLM_BATCH_FAILURE_THRESHOLD_RATIO)
    batch_failure_threshold_min: int = Field(default=5, validation_alias=EnvVar.LLM_BATCH_FAILURE_THRESHOLD_MIN)
    reasoning_effort: ReasoningEffort | None = Field(default=None, validation_alias=EnvVar.LLM_REASONING_EFFORT)
    context_manager: Annotated[ContextManagerPreset | bool, BeforeValidator(_coerce_context_manager)] = Field(
        default=ContextManagerPreset.DEFAULT,
        validation_alias=EnvVar.LLM_CONTEXT_MANAGER,
    )
    #: RLM recursive research-loop configuration (see :class:`RlmConfig`);
    #: ``None`` disables. Only the group form is accepted — no scalar
    #: shorthand.
    rlm: RlmConfig | None = None
    #: How a multi-chunk source is extracted (see the "Extraction strategies"
    #: section in docs/guides/llm-config.md). ``incremental`` (default) keeps
    #: one conversation per source with per-batch caching; ``reduce`` extracts
    #: batches independently and merges once - the bounded choice for small
    #: windows and small models; ``agentic`` walks sections with a
    #: consolidated memory digest - best coherence for long sources, but it
    #: needs a comfortable window (rule of thumb >= ~48-64k) or it aborts
    #: rather than crop the digest.
    extraction_strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.INCREMENTAL,
        validation_alias=EnvVar.LLM_EXTRACTION_STRATEGY,
    )
    slug_max_length: int = Field(default=80, validation_alias=EnvVar.LLM_SLUG_MAX_LENGTH)
    batch_retry_attempts: int = Field(default=3, validation_alias=EnvVar.LLM_BATCH_RETRY_ATTEMPTS)
    min_segment_tokens: int = Field(default=512, validation_alias=EnvVar.LLM_MIN_SEGMENT_TOKENS)
    default_chars_per_token: float = Field(default=3.85, validation_alias=EnvVar.LLM_DEFAULT_CHARS_PER_TOKEN)


class EmbeddingConfig(_KompendBaseModel):
    """Configuration for embedding models.

    Vector dimensionality is detected from the model at runtime via the
    embedding provider — no ``dim`` config value is needed.
    """

    model: str = Field(default="", validation_alias=EnvVar.LLM_EMBEDDING_MODEL)
    batch_size: int = Field(default=10, validation_alias=EnvVar.EMBEDDING_BATCH_SIZE)


class ProviderModelConfig(_KompendBaseModel):
    """Per-model settings within a provider (``[providers.<name>.models."<model>"]``)."""

    context_window: int | None = Field(
        default=None, description="Context window length for this model (finest granularity, wins over provider-level and catalog values)"
    )


class ProviderEndpoint(_KompendBaseModel):
    """A user-defined provider endpoint.

    Defined under ``[providers.<name>]`` in ``kompend.toml`` (recommended:
    globally in ``~/.config/kompend/config.toml``, since providers carry
    credentials):

    .. code-block:: toml

        [providers.my-llama]
        base_url = "http://localhost:8081/v1"
        api_key = "${MY_API_KEY}"          # optional; env var reference
        api_type = "openai-chat"           # openai-chat | openai-responses | anthropic
        context_window = 32768             # default context window for all its models

        [providers.my-llama.models."mistral-nemo"]
        context_window = 131072            # per-model override (finest granularity)

    The provider name (``my-llama``) becomes the ``provider`` part in
    ``provider/model`` strings, e.g. ``my-llama/mistral-nemo``.

    A section may omit ``base_url`` entirely to act as an **override block**
    for an integrated provider (e.g. ``[providers.ollama]`` with only
    ``max_connections``) — settings apply, model building still routes
    through the native/catalog path.
    """

    base_url: str | None = Field(
        default=None,
        description="Provider base URL (e.g. http://localhost:8081/v1). None = override-only block for an integrated provider",
    )
    api_key: str | None = Field(default=None, description="API key or env var reference like ${VAR_NAME}")
    api_type: LlmApiType = Field(
        default=LlmApiType.OPENAI_CHAT,
        description="Wire protocol the endpoint speaks: openai-chat (/v1/chat/completions), openai-responses (/v1/responses), or anthropic (/v1/messages)",
    )
    context_window: int | None = Field(
        default=None, description="Default context window length for all models on this provider (overrides auto-discovery; per-model entries win)"
    )
    models: dict[str, ProviderModelConfig] = Field(default_factory=dict, description="Per-model settings keyed by model name")
    max_connections: int | None = Field(
        default=None,
        description="Max parallel LLM connections to this provider (client-side). Clips compile.workers when set. None = no limit",
    )


def _coerce_context_manager(value: object) -> object:
    """Normalize a raw ``compile.llm.context_manager`` value for the schema.

    Strings are matched against the off-words (→ ``False``) and the preset
    members; anything else passes through for pydantic to validate.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _CONTEXT_OFF_WORDS:
            return False
        return ContextManagerPreset(normalized)
    return value
