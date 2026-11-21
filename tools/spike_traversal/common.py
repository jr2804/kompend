"""Shared contract for the traversal spike (knox-u84v.6).

Two variants run against the same tree, the same model, the same task
statement and the same budget:

- **A — planner + worker**: the shipped path. ``NodeSummarizer`` with
  ``TraversalMode.AUTO`` asks a planner per level and executes the plan.
- **B — navigating session**: one agentic session that walks the tree with
  ``list_children`` / ``browse`` / ``summarize_node`` / ``skip`` tools.

Everything both variants must agree on lives here: the model, the task
statement, the budget, and the measurement recorder. A variant that changes
one of these breaks the comparison, so the shared values are constants and
the recorder is the only way to record usage.

The spike is measurement code: it is never imported by ``src/`` and it
changes nothing in the pipeline.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from kompend.core.models.trees import ChunkTree
from kompend.core.templates import get_template_loader
from kompend.llm.models import build_model, set_user_providers

#: The model both variants use — one model, so the comparison is about the
#: traversal shape and not about the model.
MODEL = "minimax/MiniMax-M3.1-Flash"

#: The model that answers the traversal questions. Variant A asks it per
#: level; variant B asks it inside its session. Same model, same questions.
DECISION_MODEL = MODEL

#: Most nodes a variant may summarize. The budget is a contract for both
#: variants: variant A spends it on ``EXPAND`` verdicts, variant B on the
#: nodes it summarizes.
BUDGET_NODES = 60

#: Most content windows a variant may send. A window is one request's worth
#: of node content.
BUDGET_WINDOWS = 200

#: The shipped traversal task statement — the wiki's job, identical for both
#: variants. Read from the pipeline's own template so neither variant gets a
#: private paraphrase of the task.
TASK_TEMPLATE = "chunking/summarizer_traversal_task.md"

#: What a request was for.
RequestKind = Literal["decision", "summarize", "other"]


def task_statement() -> str:
    """The task statement both variants are judged against."""
    return get_template_loader().get_text(TASK_TEMPLATE)


def configure_provider() -> None:
    """Register the endpoint both variants call.

    The credentials come from the environment; nothing is written to disk.
    """
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not base_url or not token:
        msg = "ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN must be set — the spike calls the Anthropic-compatible endpoint"
        raise RuntimeError(msg)
    set_user_providers(
        {
            "minimax": {
                "base_url": base_url,
                "api_key": token,
                "api_type": "anthropic",
                "context_window": 200_000,
            }
        }
    )


@dataclass(frozen=True)
class PlantedFact:
    """A question a reader would ask, and the answer planted in the tree."""

    question: str
    answer: str
    node_id: str
    chapter: str


@dataclass(frozen=True)
class SyntheticDoc:
    """The tree the spike runs on, and what the checks need to know about it."""

    tree: ChunkTree
    facts: tuple[PlantedFact, ...]
    #: Subtree roots whose content is boilerplate — a skip is correct here.
    boilerplate_ids: frozenset[str]
    #: Subtree roots that carry a planted fact — skipping one loses an answer.
    answer_bearing_ids: frozenset[str]
    #: The chapter headings the root summary may name.
    chapter_headings: tuple[str, ...]


@dataclass
class VariantResult:
    """What one variant produced.

    Attributes:
        variant: The variant's name (``a`` / ``b``).
        summaries: Summary per node id.
        skips: Skip reason per node id.
        counter: The measurement.
        notes: Free-form notes for the run log.
    """

    variant: str
    summaries: dict[str, str]
    skips: dict[str, str]
    counter: Counter
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """The result as plain data."""
        return {
            "variant": self.variant,
            "summaries": dict(self.summaries),
            "skips": dict(self.skips),
            "notes": list(self.notes),
            **self.counter.as_dict(),
        }


@dataclass
class Counter:
    """What the spike measures.

    Attributes:
        requests: Total model requests.
        by_kind: Requests per kind (``decision`` / ``summarize`` / ``other``).
        windows: Content windows sent — one window is one request's worth of
            node content, never a cropped one.
        prompt_tokens: Total input tokens the provider reported.
        completion_tokens: Total output tokens the provider reported.
        peak_prompt_tokens: The largest input token count of a single request.
            The provider counts the whole conversation on every request, so
            this is the peak context a variant carried.
        wall_time_s: Wall clock of the measured phase.
        phase: The kind the next request counts as, when the variant does
            not classify the response itself.
    """

    requests: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    windows: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    peak_prompt_tokens: int = 0
    wall_time_s: float = 0.0
    phase: str = "other"

    def record(self, *, kind: RequestKind, input_tokens: int, output_tokens: int) -> None:
        """Record one model request."""
        self.requests += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        self.prompt_tokens += input_tokens
        self.completion_tokens += output_tokens
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, input_tokens)

    def record_window(self, count: int = 1) -> None:
        """Record *count* content windows sent."""
        self.windows += count

    def as_dict(self) -> dict[str, Any]:
        """The measurement as plain data."""
        return {
            "requests": self.requests,
            "decision_requests": self.by_kind.get("decision", 0),
            "summarize_requests": self.by_kind.get("summarize", 0),
            "other_requests": self.by_kind.get("other", 0),
            "windows": self.windows,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "peak_prompt_tokens": self.peak_prompt_tokens,
            "wall_time_s": round(self.wall_time_s, 2),
        }


class CountingModel(Model):
    """A model wrapper that counts every request the variant makes.

    The wrapper changes nothing about the request: it delegates, then reads
    the provider's own usage. The input token count of a request is the whole
    conversation the provider saw, so the maximum of those is the peak
    context the variant carried.
    """

    def __init__(
        self,
        inner: Model,
        counter: Counter,
        *,
        classify: Callable[[ModelResponse], RequestKind] | None = None,
    ) -> None:
        self._inner = inner
        self._counter = counter
        self._classify = classify

    @property
    def model_name(self) -> str:
        """The wrapped model's name."""
        return self._inner.model_name

    @property
    def base_url(self) -> str | None:
        """The wrapped model's base URL."""
        return self._inner.base_url

    @property
    def provider(self) -> str:
        """The wrapped model's provider."""
        return self._inner.provider

    @property
    def system(self) -> str:
        """The wrapped model's system name (the OpenTelemetry ``gen_ai.system``)."""
        return self._inner.system

    @property
    def profile(self) -> str:
        """The wrapped model's profile."""
        return self._inner.profile

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> ModelResponse:
        """Delegate the request and record its usage."""
        response = await self._inner.request(messages, model_settings, model_request_parameters)
        usage = response.usage
        kind = self._classify(response) if self._classify is not None else self._phase_kind()
        self._counter.record(
            kind=kind,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return response

    async def request_stream(  # noqa: ANN201 — pydantic-ai's generator signature
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
        run_context: Any | None = None,
    ) -> Any:
        """Delegate the streaming request and record its usage."""
        async with self._inner.request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            async for event in stream:
                yield event
        usage = stream.response.usage
        kind = self._classify(stream.response) if self._classify is not None else self._phase_kind()
        self._counter.record(
            kind=kind,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    def _phase_kind(self) -> RequestKind:
        """The kind the current phase asks for."""
        if self._counter.phase in ("decision", "summarize"):
            return self._counter.phase  # type: ignore[return-value]
        return "other"


def build_counted_model(counter: Counter, *, classify: Callable[[ModelResponse], RequestKind] | None = None) -> CountingModel:
    """Build the measured model both variants call."""
    return CountingModel(build_model(MODEL), counter, classify=classify)


class Stopwatch:
    """Wall clock of a measured phase."""

    def __init__(self, counter: Counter) -> None:
        self._counter = counter
        self._started = 0.0

    def __enter__(self) -> Stopwatch:
        """Start the clock."""
        self._started = time.monotonic()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the clock and record the elapsed time."""
        self._counter.wall_time_s += time.monotonic() - self._started
