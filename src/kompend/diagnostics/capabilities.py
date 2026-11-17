"""Model capability probes for ``kompend doctor``.

Each probe answers one narrow question about a model: can it produce the
structured output kompend needs, and in which shape? The results are facts
the doctor prints — never exceptions that abort the run.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from time import monotonic
from typing import Final

from PIL import Image, ImageDraw
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.output import NativeOutput, OutputSpec, PromptedOutput
from pydantic_ai.settings import ModelSettings

from kompend.core.config._configs import LlmDataFormat
from kompend.llm.format import decode_from_llm, encode_for_llm, format_note
from kompend.llm.models import build_model
from kompend.preprocessing.images import ImageMediaType
from kompend.types import JSONList

#: Longest exception line a probe result carries into the doctor output.
_MAX_DETAIL_CHARS: Final[int] = 160

#: Edge length of the probe image. A vision encoder refuses degenerate sizes
#: (a 1x1 PNG is answered with HTTP 400), so this is small but real.
_PROBE_IMAGE_SIDE: Final[int] = 64

#: The records every compact-format probe asks the model to reproduce.
#: Uniform, short fields — the shape compact formats are built for.
_PROBE_RECORDS: JSONList = [
    {"title": "session affinity", "aliases": ["sticky sessions"], "score": 4},
    {"title": "write-ahead log", "aliases": ["wal"], "score": 5},
    {"title": "read replica", "aliases": [], "score": 3},
    {"title": "leader election", "aliases": ["quorum"], "score": 4},
]


@dataclass(frozen=True)
class CapabilityProbe:
    """Outcome of a single capability probe."""

    name: str
    ok: bool
    detail: str
    skipped: bool = False


class _ProbeAnswer(BaseModel):
    """Minimal schema every structured-output probe asks for."""

    ok: bool
    message: str


def probe_native_structured_output(model: str, *, timeout: float, retries: int) -> CapabilityProbe:
    """Can the model fill a schema through a native tool call?"""
    name = "structured-output"
    try:
        agent = _agent(
            model,
            NativeOutput(_ProbeAnswer),
            timeout=timeout,
            retries=retries,
            name=f"doctor-{name}",
        )
        output, elapsed = _timed(agent, "Reply with ok=true and message='hello'.")
    except Exception as exc:
        if _is_unsupported_protocol(exc):
            # Not a model defect: the provider's protocol has no native output
            # mode, so pydantic-ai refuses the request before it is sent.
            # kompend then uses the tool/prompted mode, which the next probes
            # verify - report it as information, not as a failure.
            return CapabilityProbe(
                name,
                True,
                "provider has no native structured-output mode; kompend uses tool/prompted output",
                skipped=True,
            )
        return CapabilityProbe(name, False, f"native tool output failed: {_short(exc)}")
    if isinstance(output, _ProbeAnswer) and output.ok:
        return CapabilityProbe(name, True, f"native tool output works ({elapsed:.1f}s)")
    return CapabilityProbe(name, False, f"native tool output returned an invalid answer ({elapsed:.1f}s)")


def _is_unsupported_protocol(exc: BaseException) -> bool:
    """True when pydantic-ai rejects native output for this provider."""
    return "native structured output is not supported" in str(exc).lower()


def probe_prompted_structured_output(model: str, *, timeout: float, retries: int) -> CapabilityProbe:
    """Can the model fill a schema that is described in the prompt?"""
    name = "structured-output-prompted"
    try:
        agent = _agent(
            model,
            PromptedOutput(_ProbeAnswer),
            timeout=timeout,
            retries=retries,
            name=f"doctor-{name}",
        )
        output, elapsed = _timed(agent, "Reply with ok=true and message='hello'.")
    except Exception as exc:
        return CapabilityProbe(name, False, f"schema-in-prompt output failed: {_short(exc)}")
    if isinstance(output, _ProbeAnswer) and output.ok:
        return CapabilityProbe(name, True, f"schema-in-prompt output works ({elapsed:.1f}s)")
    return CapabilityProbe(name, False, f"schema-in-prompt output returned an invalid answer ({elapsed:.1f}s)")


def probe_raw_json(model: str, *, timeout: float, retries: int) -> CapabilityProbe:
    """Can the model emit a raw JSON object on the text path?"""
    name = "structured-output-json"
    try:
        agent = _agent(
            model,
            str,
            timeout=timeout,
            retries=retries,
            name=f"doctor-{name}",
        )
        output, elapsed = _timed(agent, 'Reply with only this JSON object: {"ok": true, "message": "hello"}')
    except Exception as exc:
        return CapabilityProbe(name, False, f"raw JSON output failed: {_short(exc)}")
    text = output if isinstance(output, str) else str(output)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return CapabilityProbe(name, False, f"raw JSON output is not valid JSON: {_short(exc)}")
    if isinstance(parsed, dict) and parsed.get("ok") is True:
        return CapabilityProbe(name, True, f"raw JSON output works ({elapsed:.1f}s)")
    return CapabilityProbe(name, False, f"raw JSON output has the wrong shape ({elapsed:.1f}s)")


def probe_compact_format(
    model: str,
    fmt: LlmDataFormat,
    *,
    timeout: float,
    retries: int,
) -> CapabilityProbe:
    """Can the model emit the configured compact format, and does it decode?"""
    name = f"structured-output-{fmt.value}"
    spec = format_note(fmt)
    prompt = f"Repeat the following records using exactly this notation. Answer with the records only, no explanation and no code fence.\n\n{spec}"
    try:
        agent = _agent(
            model,
            str,
            timeout=timeout,
            retries=retries,
            name=f"doctor-{name}",
        )
        output, elapsed = _timed(agent, prompt)
    except Exception as exc:
        return CapabilityProbe(name, False, f"{fmt.value} output failed: {_short(exc)}")
    text = output if isinstance(output, str) else str(output)
    decoded = decode_from_llm(text, fmt)
    if decoded is None:
        return CapabilityProbe(name, False, f"{fmt.value} output did not decode ({elapsed:.1f}s)")
    json_len = len(json.dumps(_PROBE_RECORDS, ensure_ascii=False))
    return CapabilityProbe(
        name,
        True,
        f"{fmt.value} output decodes ({len(text)} chars vs {json_len} chars of JSON, {elapsed:.1f}s)",
    )


def probe_vision(model: str, *, timeout: float, retries: int) -> CapabilityProbe:
    """Can the model take an image at all?

    The connectivity ping is text only, so it can pass while every figure
    fails: Ollama's ``/v1/chat/completions`` answers HTTP 400 "invalid image
    input" for a container it will not decode. This probe sends one real
    image part through the same path the captioner uses and asks for anything
    back - the question is acceptance, not caption quality.
    """
    name = "vision"
    try:
        agent: Agent[None, str] = Agent(
            build_model(model),
            output_type=str,
            retries=retries,
            model_settings=ModelSettings(timeout=timeout),
            name=f"doctor-{name}",
        )
        prompt: list[str | BinaryContent] = [
            "Reply with the single word OK.",
            BinaryContent(_probe_image_png(), media_type=ImageMediaType.PNG.value),
        ]
        start = monotonic()
        result = agent.run_sync(prompt)
        elapsed = monotonic() - start
    except Exception as exc:
        return CapabilityProbe(name, False, f"image input failed: {_short(exc)}")
    if (result.output or "").strip():
        return CapabilityProbe(name, True, f"image input accepted ({elapsed:.1f}s)")
    return CapabilityProbe(name, False, f"image input accepted but the answer was empty ({elapsed:.1f}s)")


def _probe_image_png() -> bytes:
    """A small, non-degenerate PNG for the vision probe.

    Built rather than embedded so the bytes stay readable: 64x64 white with a
    black diagonal. Size matters more than content — Ollama answers a 1x1 PNG
    with HTTP 400 "Failed to load image or audio file", which would make the
    probe a false negative on a perfectly working vision endpoint.
    """
    image = Image.new("RGB", (_PROBE_IMAGE_SIDE, _PROBE_IMAGE_SIDE), (255, 255, 255))
    ImageDraw.Draw(image).line([(0, 0), (_PROBE_IMAGE_SIDE - 1, _PROBE_IMAGE_SIDE - 1)], fill=(0, 0, 0), width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def probe_tool_use(model: str, *, timeout: float, retries: int) -> CapabilityProbe:
    """Can the model call a tool at all?

    One trivial tool, one instruction, one verdict: the probe passes when the
    model made the call. Every kompend session agent (query, write,
    remediation, verification, summariser, distiller, context chunking) needs
    this — a model without it can only run single-request tasks.
    """
    name = "tool-use"
    calls: list[str] = []

    def _ping() -> str:
        # No ``RunContext`` parameter on purpose: ``tool_plain`` registers a
        # context-free tool, and annotating the context here makes pydantic-ai
        # refuse the schema ("RunContext annotations can only be used with
        # tools that take context") - which failed for every model, not the
        # one under test.
        calls.append("ping")
        return "pong"

    try:
        agent: Agent[None, str] = Agent(
            build_model(model),
            output_type=str,
            retries=retries,
            model_settings=ModelSettings(timeout=timeout),
            name=f"doctor-{name}",
        )
        agent.tool_plain(_ping)
        _output, elapsed = _timed(agent, "Call the ping tool once, then repeat exactly what it returned.")
    except Exception as exc:
        return CapabilityProbe(name, False, f"tool call failed: {_short(exc)}")
    if calls:
        return CapabilityProbe(name, True, f"model called a tool ({elapsed:.1f}s)")
    return CapabilityProbe(name, False, "model answered without calling the tool")


def _agent[OutputT](
    model: str,
    output_type: OutputSpec[OutputT],
    *,
    timeout: float,
    retries: int,
    name: str,
) -> Agent[None, OutputT]:
    """Build a probe agent bounded only by its timeout.

    No ``max_tokens``: a reply is bounded by the model's context window, and
    the probe asks for a tiny answer anyway (see ``_PROBE_RECORDS``).
    """
    return Agent(
        build_model(model),
        output_type=output_type,
        retries=retries,
        model_settings=ModelSettings(timeout=timeout),
        name=name,
    )


def _timed[OutputT](agent: Agent[None, OutputT], prompt: str) -> tuple[OutputT, float]:
    """Run *prompt* and return the output plus the elapsed seconds."""
    start = monotonic()
    result = agent.run_sync(prompt)
    return result.output, monotonic() - start


def measure_compact_formats() -> CapabilityProbe:
    """Compare the encoded size of a representative payload per format.

    Deterministic — no model call. It answers "how much context does the
    configured format save?" even for a model that cannot be reached.
    """
    name = "input-format-size"
    sizes = {fmt.value: len(encode_for_llm(_PROBE_RECORDS, fmt)) for fmt in (LlmDataFormat.JSON, LlmDataFormat.TOON, LlmDataFormat.ISON)}
    json_size = sizes[LlmDataFormat.JSON.value]
    detail = ", ".join(f"{fmt} {size} chars" for fmt, size in sizes.items())
    best = min(sizes, key=lambda key: sizes[key])
    saving = 0 if json_size == 0 else round(100 * (json_size - sizes[best]) / json_size)
    return CapabilityProbe(name, True, f"{detail} — {best} saves {saving}% over JSON")


def _short(exc: BaseException) -> str:
    """One-line rendering of an exception for a probe result.

    Every non-empty line is kept, joined by ``" / "``: a pydantic-ai error
    reads ``Error generating schema for _ping:`` on its first line and puts
    the actual reason on the second, so keeping only the head would print a
    colon and nothing else.
    """
    lines = [line.strip() for line in str(exc).strip().splitlines() if line.strip()]
    rendered = " / ".join(lines) if lines else exc.__class__.__name__
    return rendered[:_MAX_DETAIL_CHARS]
