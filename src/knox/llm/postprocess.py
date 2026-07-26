"""LLM response post-processing pipeline.

Modular, composable post-processors for JSON repair, surrogate scrubbing,
thinking suppression, and usage tracking. Each processor has a single
responsibility and can be tested/extended independently.

The pipeline is pluggable: add new processors without touching ``KnoxModel``,
and compose them in any order via ``ResponsePipeline``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.messages import ModelRequest, TextPart, ThinkingPart, ToolCallPart

from knox.llm.debug import messages_to_debug_dicts, save_debug_prompt
from knox.llm.format import salvage_json
from knox.llm.usage import UsageTracker

logger = logging.getLogger(__name__)


# ── Core protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class ResponsePostprocessor(Protocol):
    """Single-responsibility post-processing step."""

    def __call__(self, response: ModelResponse) -> None:
        """Apply transformation in-place to response."""
        ...


# ── Helper functions ────────────────────────────────────────────────────


def _scrub_dict_strings(data: dict[str, Any], scrub_fn: Callable[[str], str]) -> dict[str, Any]:
    """Recursively apply scrub_fn to all string values in a dict.

    Non-string values (int, bool, None, list, nested dict) are passed through
    unchanged. Lists are walked recursively to scrub strings inside them.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = scrub_fn(value)
        elif isinstance(value, dict):
            result[key] = _scrub_dict_strings(value, scrub_fn)
        elif isinstance(value, list):
            result[key] = [_scrub_value(item, scrub_fn) for item in value]
        else:
            result[key] = value
    return result


def _scrub_value(value: Any, scrub_fn: Callable[[str], str]) -> Any:
    """Apply scrub_fn to strings; pass through other types."""
    if isinstance(value, str):
        return scrub_fn(value)
    if isinstance(value, dict):
        return _scrub_dict_strings(value, scrub_fn)
    if isinstance(value, list):
        return [_scrub_value(item, scrub_fn) for item in value]
    return value


# ── Post-processor implementations ──────────────────────────────────────


class SurrogateScrubber:
    """Remove lone UTF-16 surrogate characters from all text parts.

    Small Ollama models can emit lone surrogates (U+D800–U+DFFF) in tool
    call content. Python strings allow them internally, but ``json.dumps``
    chokes with ``UnicodeEncodeError: surrogates not allowed``. This
    processor scrubs them before any JSON serialization occurs.
    """

    _SURROGATE_REMOVER = dict.fromkeys(range(0xD800, 0xE000))
    """Translation table that maps every UTF-16 surrogate to ``None`` (removal)."""

    def __call__(self, response: ModelResponse) -> None:
        """Scrub surrogates from all text in response parts."""

        def _scrub(text: str) -> str:
            return text.translate(self._SURROGATE_REMOVER)

        for part in response.parts:
            if isinstance(part, ToolCallPart) and isinstance(part.args, dict):
                part.args = _scrub_dict_strings(part.args, _scrub)
            elif isinstance(part, (TextPart, ThinkingPart)):
                part.content = _scrub(part.content)


class ToolCallRepair:
    """Run json_repair schema-guided repair on tool-call args.

    Small local models produce tool-call arguments with invalid JSON
    (truncation, bad escapes, stray control chars) and type mismatches
    (``"related": ""`` instead of ``[]``, garbled string fragments as page
    dicts). pydantic-ai's built-in validator uses pydantic's strict
    ``validate_json`` which rejects all of these, triggering a retry that
    wastes context window and usually produces worse output.

    This processor intercepts the raw args *before* pydantic-ai validates
    them and repairs the JSON using ``salvage_json`` with schema guidance.
    """

    def __init__(self, schema_resolver: Callable[[], Any | None]) -> None:
        """Initialize with a callable that returns the result schema."""
        self._get_schema = schema_resolver

    def __call__(self, response: ModelResponse) -> None:
        """Repair tool-call args in-place."""
        schema = self._get_schema()
        if schema is None:
            return

        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue
            if not isinstance(part.args, dict):
                continue
            repaired = salvage_json(
                json.dumps(part.args, ensure_ascii=False),
                schema=schema,
            )
            if isinstance(repaired, dict):
                part.args = repaired


class ThinkingPartRepair:
    """Convert ThinkingPart/TextPart with valid JSON to ToolCallPart.

    Small Ollama models frequently return structured output as raw text
    (``TextPart``) instead of a tool call (``ToolCallPart``). PydanticAI's
    ``AutoOutputSchema`` has two validation paths:

    - **Tool path:** receives ``ToolCallPart.args`` as a dict, calls
      ``validate_python(dict)`` — always works.
    - **Text path:** receives ``TextPart.content`` as a string, calls
      ``validate_json(string)`` — fails on double-encoded JSON because
      ``validate_json`` parses the outer string literal and gets a Python
      ``str``, not a ``dict``, producing ``Input should be an object``.

    This processor intercepts the response *before* PydanticAI validates
    it, parses the text content via ``salvage_json``, and converts the
    ``TextPart`` to a ``ToolCallPart`` with the parsed dict as ``args``.
    This forces the tool validation path, which always works correctly.

    Only applies to agents with structured output (``output_type`` is not
    ``str``). Free-text agents keep the original ``TextPart``.
    """

    def __init__(self, schema_resolver: Callable[[], Any | None]) -> None:
        """Initialize with a callable that returns the result schema."""
        self._get_schema = schema_resolver

    def __call__(self, response: ModelResponse) -> None:
        """Convert ThinkingPart/TextPart with JSON to ToolCallPart."""
        schema = self._get_schema()
        is_structured = schema is not None

        for i, part in enumerate(response.parts):
            if not isinstance(part, ThinkingPart):
                continue
            content = part.content
            if not content:
                continue
            repaired = salvage_json(content)
            if isinstance(repaired, (dict, list)):
                new_part = TextPart(content=json.dumps(repaired, ensure_ascii=False))
                response.parts[i] = new_part  # ty: ignore[invalid-assignment]
                logger.info(
                    "Converted ThinkingPart to TextPart (%d chars of JSON)",
                    len(new_part.content),
                )

        has_tool_call = any(isinstance(p, ToolCallPart) for p in response.parts)

        for i, part in enumerate(response.parts):
            if not isinstance(part, TextPart):
                continue
            content = part.content
            if not isinstance(content, str) or not content:
                continue
            repaired = salvage_json(content)
            if repaired is None:
                continue
            if not isinstance(repaired, (dict, list)):
                continue

            if is_structured and not has_tool_call and isinstance(repaired, dict):
                response.parts[i] = ToolCallPart(  # ty: ignore[invalid-assignment]
                    tool_name="final_result",
                    args=repaired,
                )
                logger.info(
                    "Converted TextPart to ToolCallPart (%d chars -> %d chars)",
                    len(content),
                    len(json.dumps(repaired, ensure_ascii=False)),
                )
            else:
                part.content = json.dumps(repaired, ensure_ascii=False)
                logger.info(
                    "Repaired TextPart content via salvage_json (%d chars -> %d chars)",
                    len(content),
                    len(part.content),
                )


class UsageRecorder:
    """Record token usage from a response onto the usage tracker.

    Reads ``response.usage`` and records input/output tokens via the
    canonical ``UsageTracker`` interface.
    """

    def __init__(self, tracker: UsageTracker | None, model_name: str) -> None:
        """Initialize with usage tracker and model identifier."""
        self._tracker = tracker
        self._model_name = model_name

    def __call__(self, response: ModelResponse) -> None:
        """Record usage data from response if available."""
        if self._tracker is None:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._tracker.record(
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self._model_name,
        )


class DebugSaver:
    """Dump call to disk when KNOX_DEBUG_LLM_DIR is set.

    Saves the complete message-history + response to a timestamped JSON
    file for post-mortem debugging of model output.
    """

    def __init__(
        self,
        agent: Any,
        model_name: str,
        messages: Sequence[ModelMessage],
    ) -> None:
        """Initialize with agent ref, model name, and request messages.

        The *agent* argument is a pydantic-ai ``Agent`` instance with
        ``stage`` and ``current_source_id`` properties (protocol defined
        in ``knox.llm.debug``).  Its ``_last_debug_path`` attribute is
        set after a successful save.
        """
        self._agent = agent
        self._model_name = model_name
        self._messages = messages

    def __call__(self, response: ModelResponse) -> None:
        """Save debug dump if directory is configured."""
        content = ""
        for part in response.parts:
            if isinstance(part, TextPart):
                content = part.content
                break
        finish_reason = getattr(response, "finish_reason", None) or "stop"

        user_turns = sum(1 for m in self._messages if isinstance(m, ModelRequest))
        assistant_turns = sum(1 for m in self._messages if isinstance(m, ModelResponse))

        all_dicts = messages_to_debug_dicts(list(self._messages) + [response])

        filepath = save_debug_prompt(
            self._agent,
            self._model_name,
            all_dicts,
            {"model": self._model_name},  # nocheck: debug log dict, not wiki metadata
            content,
            str(finish_reason),
            user_turns=user_turns,
            assistant_turns=assistant_turns,
        )
        if filepath is not None and hasattr(self._agent, "_last_debug_path"):
            self._agent._last_debug_path = filepath


# ── Pipeline composition ─────────────────────────────────────────────────


class ResponsePipeline:
    """Compose multiple post-processors into a single pipeline.

    Processors are executed in the order they are added. This allows
    fine-grained control over the processing sequence (e.g., scrub surrogates
    before JSON repair to avoid ``json.dumps`` crashes).
    """

    def __init__(self, processors: Sequence[ResponsePostprocessor]) -> None:
        """Initialize with a sequence of processors."""
        self._processors = list(processors)

    def __call__(self, response: ModelResponse) -> None:
        """Apply all processors to the response in order."""
        for processor in self._processors:
            if processor is not None:
                processor(response)

    def extend(self, processors: Sequence[ResponsePostprocessor]) -> None:
        """Add more processors to the end of the pipeline."""
        self._processors.extend(processors)

    def insert(self, index: int, processor: ResponsePostprocessor) -> None:
        """Insert a processor at a specific position."""
        self._processors.insert(index, processor)


# ── Plugin system ────────────────────────────────────────────────────────


# Global plugin registry — accumulates decorator-registered processors.
_registry: dict[str, type[ResponsePostprocessor]] = {}


def register_postprocessor(name: str) -> Callable[[type[ResponsePostprocessor]], type[ResponsePostprocessor]]:
    """Decorator that registers a post-processor class for discovery.

    Usage::

        @register_postprocessor("my_repair")
        class MyRepair:
            def __call__(self, response: ModelResponse) -> None: ...
    """

    def decorator(processor_class: type[ResponsePostprocessor]) -> type[ResponsePostprocessor]:
        _registry[name] = processor_class
        return processor_class

    return decorator


def get_registered_processor(name: str) -> type[ResponsePostprocessor] | None:
    """Retrieve a registered post-processor by name."""
    return _registry.get(name)


def list_registered_processors() -> list[str]:
    """List all registered post-processor names."""
    return list(_registry.keys())


def pipeline_from_registry(names: Sequence[str], **kwargs: Any) -> ResponsePipeline:
    """Build a pipeline from registered processor names.

    Pass constructor keyword arguments as ``<name>__<kwarg>=value``,
    e.g. ``ToolCallRepair__schema_resolver=fn`` to supply the schema
    resolver to ``ToolCallRepair``.
    """
    processors = []
    for name in names:
        processor_class = get_registered_processor(name)
        if processor_class is None:
            raise ValueError(f"Unknown post-processor: {name}")
        instance_kwargs = {}
        prefix = f"{name}__"
        for key, val in kwargs.items():
            if key.startswith(prefix):
                instance_kwargs[key[len(prefix) :]] = val
        processors.append(processor_class(**instance_kwargs))
    return ResponsePipeline(processors)


# Register built-in processors
register_postprocessor("surrogate_scrub")(SurrogateScrubber)
register_postprocessor("tool_call_repair")(ToolCallRepair)
register_postprocessor("thinking_part_repair")(ThinkingPartRepair)
register_postprocessor("usage_recorder")(UsageRecorder)
register_postprocessor("debug_saver")(DebugSaver)


# ── Default pipeline factory ────────────────────────────────────────────


def default_pipeline(
    *,
    schema_resolver: Callable[[], Any | None],
    agent: Any | None,
    model_name: str,
    usage_tracker: UsageTracker | None,
    messages: list[ModelMessage],
) -> ResponsePipeline:
    """Create the default post-processing pipeline used by KnoxModel.

    Order matters:
    1. SurrogateScrubber — remove surrogates before JSON operations
    2. ToolCallRepair — repair tool-call args using schema guidance
    3. ThinkingPartRepair — convert ThinkingPart/TextPart to ToolCallPart
    4. UsageRecorder — record input/output token counts
    5. DebugSaver — save debug dump if configured
    """
    processors: list[ResponsePostprocessor] = [
        SurrogateScrubber(),
        ToolCallRepair(schema_resolver=schema_resolver),
        ThinkingPartRepair(schema_resolver=schema_resolver),
        UsageRecorder(tracker=usage_tracker, model_name=model_name),
    ]
    if agent is not None:
        processors.append(DebugSaver(agent=agent, model_name=model_name, messages=messages))
    return ResponsePipeline(processors)
