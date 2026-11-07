"""Model checks: connectivity, structured-output capability, context windows.

Every check here talks to a provider, so it is the slow half of the doctor.
A model is never swapped out between two of its own checks: the runner groups
them per model and the embedder runs last, because loading it evicts the
completion models from Ollama's memory.
"""

from __future__ import annotations

from time import monotonic
from typing import Final

from kompend.agents.doctor import DoctorAgent, DoctorDeps
from kompend.core.config import LLMConfig, LlmDataFormat
from kompend.diagnostics.base import CheckResult
from kompend.diagnostics.capabilities import (
    probe_compact_format,
    probe_native_structured_output,
    probe_prompted_structured_output,
    probe_raw_json,
)
from kompend.diagnostics.providers import OLLAMA_PROVIDER_ID
from kompend.llm import model_slug, provider_id_of
from kompend.llm.context_window import discover_context_window
from kompend.llm.models import (
    _query_ollama_num_ctx,
    build_model,
    requires_output_ceiling,
    resolve_model_limits,
)
from kompend.llm.usage import UsageTracker, tokens_per_second
from kompend.rag.embedding_pydantic import build_embedder
from kompend.rag.vector_store import to_pydantic_ai_embedding_string

#: Smallest input budget a window may leave before the doctor calls it unusable.
MIN_INPUT_TOKENS: Final[int] = 2048

#: The connectivity ping: one word keeps the measured tok/s honest (a long
#: answer would be dominated by generation length, not by throughput).
PING_PROMPT: Final[str] = "Respond with exactly one word: ok"

#: The text the embedder is asked to embed. Any string works; keeping it short
#: keeps the check about reachability, not about tokenising a document.
EMBED_PING_TEXT: Final[str] = "ping"


def ping_model(
    model: str | None,
    label: str,
    *,
    verbose: bool = False,
    timeout: float = 60.0,
    retries: int | None = None,
) -> CheckResult:
    """Ping a completion-type model (LLM or VLM) through the doctor agent."""
    name = f"connectivity-{label}"
    if not model:
        if label == "llm":
            # A missing LLM is a hard failure — a compile cannot run without one.
            return CheckResult(name, False, "No llm model configured — skipped")
        return CheckResult(name, True, f"No {label} model configured — skipped", warning=True)

    try:
        tracker = UsageTracker()
        agent = DoctorAgent(
            model_string=model,
            timeout=timeout,
            retries=2 if retries is None else retries,
            usage_tracker=tracker,
        )
        deps = DoctorDeps(ping=PING_PROMPT)
        # Warm-up run: the first call pays model load (cold start) and would
        # otherwise dominate the reported throughput. Discard it and measure
        # the second, warm run.
        agent.run(deps)
        tokens_before = tracker.completion_tokens
        start = monotonic()
        result = agent.run(deps)
        elapsed = monotonic() - start
        rate = tokens_per_second(tracker.completion_tokens - tokens_before, elapsed)
        rate_s = f" ({rate:.1f} tok/s over {elapsed:.1f}s)" if rate else ""
        if result and result.strip():
            return CheckResult(name, True, f"{label} OK ({model}){rate_s}")
        return CheckResult(name, False, f"{label} returned empty response")
    except Exception as exc:  # noqa: BLE001 — a failed ping is the fact being reported
        return CheckResult(name, False, f"{label} failed: {exc}")


def ping_embed_model(model: str | None) -> CheckResult:
    """Ping an embedding model and report its dimension count."""
    name = "connectivity-embed"
    if not model:
        return CheckResult(name, True, "No embed model configured — skipped", warning=True)

    try:
        pydantic_ai_string = to_pydantic_ai_embedding_string(model)
        embedder = build_embedder(pydantic_ai_string)
        result = embedder.embed_documents_sync(EMBED_PING_TEXT)
        dims = len(result.embeddings[0]) if result.embeddings else 0
        return CheckResult(name, True, f"embed OK ({pydantic_ai_string}, {dims} dims)")
    except Exception as exc:  # noqa: BLE001 — a failed embed is the fact being reported
        return CheckResult(name, False, f"embed failed: {exc}")


def check_model_capabilities(
    model: str | None,
    label: str,
    *,
    timeout: float = 120.0,
    retries: int | None = None,
    data_format: LlmDataFormat = LlmDataFormat.JSON,
    full: bool = True,
) -> list[CheckResult]:
    """Probe how the model produces structured output.

    The first probe — native tool output — is the capability every typed
    kompend agent depends on. The extra probes tell the user what else the
    model manages: a model that fails natively but answers a schema in the
    prompt is still usable, and a model that can emit the configured compact
    format saves context on every structured payload.
    """
    name = f"structured-output-{label}"
    if not model:
        return [CheckResult(name, True, f"No {label} model configured — skipped")]
    budget = 1 if retries is None else retries
    probes = [probe_native_structured_output(model, timeout=timeout, retries=budget)]
    if full:
        probes.append(probe_prompted_structured_output(model, timeout=timeout, retries=budget))
        probes.append(probe_raw_json(model, timeout=timeout, retries=budget))
        if data_format in (LlmDataFormat.TOON, LlmDataFormat.ISON):
            probes.append(probe_compact_format(model, data_format, timeout=timeout, retries=budget))
    results: list[CheckResult] = []
    for probe in probes:
        check_name = name if probe.name == "structured-output" else f"{probe.name}-{label}"
        results.append(CheckResult(check_name, probe.ok, f"{probe.detail} (model={model})", warning=probe.skipped))
    return results


def check_context_window(
    model: str | None,
    label: str,
    context_window: int | None = None,
    llm_config: LLMConfig | None = None,
) -> CheckResult:
    """Discover and report the context window for a model."""
    name = f"context-window-{label}"
    if not model:
        return CheckResult(name, True, f"No {label} model configured — skipped")

    try:
        window = discover_context_window(model, llm_config=llm_config, context_window_cap=context_window)
    except Exception as exc:  # noqa: BLE001 — an unresolvable window is the fact being reported
        return CheckResult(name, False, f"{label} context window discovery failed: {exc}")
    source_label = window.source.value if hasattr(window.source, "value") else str(window.source)
    if window.source == "fallback":
        return CheckResult(
            name,
            True,
            f"{label}: {window.max_input} in / {window.max_output} out tokens ({source_label} — model not found in registry)",
        )
    return CheckResult(
        name,
        True,
        f"{label}: {window.max_input} in / {window.max_output} out tokens (source: {source_label})",
    )


def check_window_fit(
    model: str | None,
    context_window: int | None = None,
    llm_config: LLMConfig | None = None,
) -> CheckResult:
    """Check that the discovered window still leaves room for a request.

    kompend sends no ``max_tokens``: the answer is bounded by the window minus
    its prompt, and the planners reserve a share of the window for it
    (``ContextWindow.answer_reserve``). The only thing that can be wrong is a
    window too small to carry both.
    """
    name = "window-fit"
    if not model:
        return CheckResult(name, True, "No model - skipped")

    try:
        window = discover_context_window(model, context_window_cap=context_window, llm_config=llm_config)
    except Exception as exc:  # noqa: BLE001 — an unresolvable window is the fact being reported
        return CheckResult(name, False, f"Cannot discover context window: {exc}")

    reserve = window.answer_reserve
    input_budget = window.input_budget()
    if input_budget < MIN_INPUT_TOKENS:
        return CheckResult(
            name,
            False,
            f"Window {window.max_input} tokens leaves {input_budget} for input after the {reserve}-token answer reserve - increase the model's context window",
        )
    return CheckResult(name, True, f"{window.max_input} ctx: {reserve} answer reserve, {input_budget} input")


def check_server_window(model: str | None, context_window: int | None = None) -> CheckResult:
    """Warn when the configured window exceeds what the server enforces.

    A local Ollama server started with ``-c 32768`` serves 32768 tokens no
    matter what the GGUF metadata or the configuration claim. kompend learns
    the server's real limit from the provider when a request is rejected (and
    re-plans against it), but the doctor can say so up front: it pings the
    model first, so the server has it loaded and reports the context it
    actually serves.

    Only Ollama models are checked — a cloud provider's limit is the one the
    catalog reports.
    """
    name = "server-window"
    if not model or provider_id_of(model) != OLLAMA_PROVIDER_ID:
        return CheckResult(name, True, "Not an Ollama model - skipped")
    try:
        server_ctx = _query_ollama_num_ctx(model_slug(model))
    except Exception as exc:  # noqa: BLE001 — an unreachable daemon is reported, not raised
        return CheckResult(name, True, f"Ollama not reachable ({exc}) - skipped")
    if server_ctx is None:
        return CheckResult(name, True, "Ollama reports no context limit - skipped")

    try:
        effective = discover_context_window(model, context_window_cap=context_window)
    except Exception:  # noqa: BLE001 — without a window there is nothing to compare
        return CheckResult(name, True, f"Server serves {server_ctx} tokens - skipped")

    if effective.max_input <= server_ctx:
        return CheckResult(name, True, f"Server serves {server_ctx} tokens, kompend plans for {effective.max_input}")

    return CheckResult(
        name,
        True,
        f"Server serves {server_ctx} tokens but kompend plans for {effective.max_input} - "
        f"start the server with a matching -c {server_ctx}, or lower compile.llm.context_window. "
        f"Requests that do not fit are rejected and re-planned automatically.",
        warning=True,
    )


def check_output_ceiling(model: str | None, label: str) -> CheckResult:
    """Report whether the provider protocol demands an explicit output ceiling.

    kompend sends no ``max_tokens`` for OpenAI-compatible endpoints or Ollama.
    The Anthropic protocol rejects an unset value (pydantic-ai then uses 4096),
    so there it computes one per request: window minus the prompt, minus a
    margin, clamped by the model's own output limit when the catalog knows it.
    """
    name = f"output-ceiling-{label}"
    if not model:
        return CheckResult(name, True, f"No {label} model configured - skipped")
    try:
        native = build_model(model)
    except Exception as exc:  # noqa: BLE001 — an unbuildable model is a fact to report
        return CheckResult(name, False, f"Cannot build the model: {exc}")

    if not requires_output_ceiling(native):
        return CheckResult(name, True, "No ceiling sent - the provider default applies")

    try:
        _, limit = resolve_model_limits(model)
    except (KeyError, LookupError, ValueError):
        limit = None
    clamp = f"clamped by the model's {limit}-token output limit" if limit else "catalog knows no output limit for this model"
    return CheckResult(name, True, f"Protocol requires max_tokens: computed per request (window - prompt - margin), {clamp}")


__all__ = [
    "EMBED_PING_TEXT",
    "MIN_INPUT_TOKENS",
    "PING_PROMPT",
    "check_context_window",
    "check_model_capabilities",
    "check_output_ceiling",
    "check_server_window",
    "check_window_fit",
    "ping_embed_model",
    "ping_model",
]
