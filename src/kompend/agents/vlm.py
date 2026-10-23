"""VlmAgent — PydanticAI agent for VLM image captioning.

Routes through the unified :class:`KompendAgentBase` → :class:`KompendModel` →
``build_model()`` path like every other agent, passing image bytes via a
:class:`pydantic_ai.messages.BinaryContent` part in the user prompt.

Reasoning is forced off (``reasoning_effort="none"``) — image captioning
never benefits from thinking traces, and uncontrolled reasoning was the
root cause of VLM timeouts and reasoning-polluted captions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from pydantic_ai.messages import BinaryContent

from kompend.agents.base import KompendAgentBase
from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.debug import save_debug_prompt
from kompend.llm.format import strip_reasoning
from kompend.logging import get_logger


#: Image extensions the VLM agent recognises. Keys are a :class:`StrEnum` so
#: the table is keyed by typed members; lookups still match bare suffix
#: strings because ``StrEnum`` is-a ``str``. Excludes ``.svg`` because
#: pydantic-ai's :class:`ImageMediaType` only accepts the raster MIME set
#: (jpeg/png/gif/webp) — ``.bmp`` is intentionally kept here even though
#: pydantic-ai rejects it: the lookup falls through to ``_DEFAULT_MEDIA_TYPE``
#: (``image/png``) before pydantic-ai ever sees the value.
class _ImageExt(StrEnum):
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    GIF = ".gif"
    WEBP = ".webp"
    BMP = ".bmp"


_MIME_BY_EXT: dict[_ImageExt, str] = {
    _ImageExt.PNG: "image/png",
    _ImageExt.JPG: "image/jpeg",
    _ImageExt.JPEG: "image/jpeg",
    _ImageExt.GIF: "image/gif",
    _ImageExt.WEBP: "image/webp",
    _ImageExt.BMP: "image/bmp",
}

# pydantic-ai requires a concrete ImageMediaType; this fallback covers the
# rare case of an unrecognised extension.
_DEFAULT_MEDIA_TYPE = "image/png"

logger = get_logger(__name__)


@dataclass
class VlmDeps:
    """Deps for VLM captioning."""

    prompt: str
    image_file: str
    source_id: str = "vlm"


class VlmAgent(KompendAgentBase[str, VlmDeps]):
    """PydanticAI agent for VLM image captioning.

    Overrides :meth:`run` to attach the image as a ``BinaryContent`` part
    alongside the text prompt, since the standard single-string user
    prompt cannot carry image bytes.
    """

    template_name: ClassVar[str] = ""
    system_template_name: ClassVar[str] = ""

    def __post_init__(self) -> None:
        # Image captioning never benefits from reasoning — force it off so
        # the model doesn't burn time on thinking traces that pollute
        # captions and cause timeouts on slow local vision models.
        self.reasoning_effort = "none"
        super().__post_init__()

    def run(self, deps: VlmDeps, *, message_history: list[Any] | None = None) -> str:  # noqa: PLR6301
        """Run VLM captioning with image input via the unified model path."""
        self._current_source_id = deps.source_id
        image_file = Path(deps.image_file)
        if not image_file.is_file():
            msg = f"Image not found: {image_file}"
            raise FileNotFoundError(msg)

        media_type = _MIME_BY_EXT.get(image_file.suffix.lower(), _DEFAULT_MEDIA_TYPE)
        image_bytes = image_file.read_bytes()

        # User prompt is a sequence: instruction text + image part. pydantic-ai
        # forwards the BinaryContent to the native model's image handling.
        user_prompt: list[str | BinaryContent] = [
            deps.prompt,
            BinaryContent(image_bytes, media_type=media_type),
        ]

        if self._agent is None:
            self._agent = self._build_agent()

        try:
            result = self._agent.run_sync(user_prompt, deps=deps)
        except Exception as exc:
            logger.warning("VLM call failed for '%s': %s", deps.image_file, exc)
            raise LLMOutputError(f"VLM failed for model '{self.model_string}': {exc}") from exc

        caption = strip_reasoning(result.output).strip()

        save_debug_prompt(
            self,
            self.model_string,
            [{"role": "user", "parts": [{"type": "text", "content": deps.prompt}]}],
            {"model": self.model_string, "image_bytes": len(image_bytes)},  # nocheck: debug log dict, not wiki metadata
            caption,
            "stop",
        )
        return caption
