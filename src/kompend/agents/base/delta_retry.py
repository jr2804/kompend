"""Delta-retry salvage: recover partial outputs on retry.

When an agentic agent produces an output that almost validates — a few
fields broken in an otherwise good object, or the requested list of
objects with some entries invalid — pydantic-ai would discard the whole
answer and retry from scratch. These helpers let the retry hook merge
the valid parts of the previous answer into the next attempt instead,
salvaging what the model already got right.
"""

from __future__ import annotations

import json
import typing
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart

from kompend.llm.format import salvage_json
from kompend.types import JSONDict, JSONValue

if TYPE_CHECKING:
    from kompend.agents.base.agent import KompendAgentBase

# ── Delta-retry helpers ────────────────────────────────────────────────


def _get_response_text(response: ModelResponse) -> str | None:
    """Extract the text content from a ModelResponse."""
    for part in response.parts:
        if isinstance(part, TextPart) and part.content:
            return part.content
    return None


def _set_response_text(response: ModelResponse, data: str) -> None:
    """Replace the text content of a ModelResponse with JSON text."""
    for part in response.parts:
        if isinstance(part, TextPart):
            part.content = data
            return
    response.parts = [*response.parts, TextPart(content=data)]


def _build_delta_message(parsed: JSONValue, schema: type[BaseModel]) -> str | None:
    """Build a delta-retry message describing what needs fixing.

    Examines the parsed (but invalid) output against the schema and
    produces a message that tells the model exactly which parts are
    broken and what the expected structure looks like.
    """
    issues: list[str] = []

    # Check if the model returned a list when the schema expects an object.
    if isinstance(parsed, list):
        expected = _format_schema_fields(schema)
        issues.append(f"- You returned a JSON array, but the expected output is a single JSON object with fields {expected}. Return one object, not a list.")

    if isinstance(parsed, dict):
        # Check each field against the schema
        for field_name, field_info in schema.model_fields.items():
            _check_field(parsed, field_name, field_info, issues)

    if not issues:
        return None

    issue_text = "\n".join(issues)
    return (
        "[CORRECTION REQUEST — This is NOT a new task. Your previous output had "
        "the following structural issues. Fix them and return the COMPLETE corrected "
        "output, keeping all valid parts unchanged.]\n\n"
        f"{issue_text}\n\n"
        "Return the COMPLETE corrected output as a single valid JSON object matching "
        "the expected schema exactly. Do not add explanations, markdown formatting, "
        "or extra fields. If this keeps failing, your model may not support structured "
        "JSON output — consider disabling delta-retry in the config "
        "(`compile.llm.delta_retry_enabled = false`)."
    )


def _check_field(
    parsed: dict[str, JSONValue],
    field_name: str,
    field_info: FieldInfo,
    issues: list[str],
) -> None:
    """Compare one parsed output field against its schema declaration."""
    if field_name not in parsed:
        issues.append(f"- missing field '{field_name}'")
        return
    value = parsed[field_name]
    inner_type = _get_list_inner_type(field_info.annotation)
    if not (inner_type and isinstance(value, list)):
        return
    expected = _format_schema_fields(inner_type)
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            issues.append(f"- {field_name}[{i}]: expected an object with fields {expected}, got {type(item).__name__} ({json.dumps(str(item))})")


def _get_list_inner_type(annotation: Any) -> type[BaseModel] | None:
    """Extract the inner pydantic-model type from a ``list[X]`` annotation."""
    origin = typing.get_origin(annotation)
    if origin is not list:
        return None
    args = typing.get_args(annotation)
    inner = args[0] if args else None
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return inner
    return None


def _format_schema_fields(model: object) -> str:
    """Format the field names of a Pydantic model for display."""
    if isinstance(model, type) and issubclass(model, BaseModel):
        return "{" + ", ".join(model.model_fields) + "}"
    return "{...}"


def _merge_valid_data(previous: JSONValue, current: JSONDict, schema: type[BaseModel]) -> JSONDict | None:
    """Merge valid data from a previous retry attempt with the current one.

    When the model returns fewer items on retry (e.g., only the corrected
    pages), merge them with the previously valid items by replacing the
    tail of the previous list with the new items.
    """
    if not isinstance(previous, dict):
        return None

    merged = dict(current)
    for field_name in schema.model_fields:
        if field_name not in previous or field_name not in current:
            continue
        prev_val = previous[field_name]
        curr_val = current[field_name]
        if isinstance(prev_val, list) and isinstance(curr_val, list) and len(curr_val) < len(prev_val):
            # New response has fewer items — replace the tail of previous
            merged[field_name] = prev_val[: len(prev_val) - len(curr_val)] + curr_val

    return merged


def _is_cache_rejection(exc: BaseException) -> bool:
    """True when an error says the provider does not accept the cache field."""
    status = getattr(exc, "status_code", None)
    if status not in (400, 404, 422):
        return False
    text = str(exc).lower()
    return "cache" in text or "unknown field" in text or "extra inputs" in text


def make_delta_retry_hook(agent: KompendAgentBase) -> Hooks:
    """Create an ``after_model_request`` hook for delta-retry.

    When the model returns partially valid structured output, this hook
    extracts the valid parts and raises ``ModelRetry`` with a message
    that tells the model exactly which parts need fixing and what the
    expected structure looks like.  On retry, valid parts from previous
    attempts are accumulated and merged.

    Only fires for agents with ``supports_delta_retry = True`` and a
    Pydantic ``result_type`` (structured output).
    """
    state = agent._delta_retry_state
    hooks = Hooks()

    @hooks.on.after_model_request
    async def _delta_retry_hook(
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        # Only handle structured output (TextPart with JSON)
        schema = agent.result_type
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            return response  # text output — nothing to salvage
        text = _get_response_text(response)
        if not text:
            return response

        # Try to parse with schema-aware salvage
        parsed = salvage_json(text, schema=schema)
        if parsed is None:
            return response  # Can't parse — let PydanticAI handle it

        # Try to validate against the schema
        try:
            validated = schema.model_validate(parsed)
        except Exception:
            # Validation failed — build a delta message
            step = ctx.run_step
            delta_msg = _build_delta_message(parsed, schema)
            if delta_msg:
                # Store the partial data for potential merge on retry
                state[step] = parsed
                raise ModelRetry(delta_msg)
            return response

        # Fully valid — replace response text with corrected JSON so
        # PydanticAI validates the fixed version, not the original broken one.
        # salvage_json may have restructured fields, added defaults, or coerced
        # types — without this, PydanticAI re-validates the original text.
        corrected_json = validated.model_dump_json()
        if corrected_json != text:
            _set_response_text(response, corrected_json)

        # Merge with any accumulated state from previous retries
        step = ctx.run_step
        if step in state:
            merged = _merge_valid_data(state[step], validated.model_dump(mode="json"), schema)
            if merged is not None:
                _set_response_text(response, json.dumps(merged, ensure_ascii=False, default=str))
            del state[step]
        return response

    return hooks
