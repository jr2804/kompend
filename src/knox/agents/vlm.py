"""VlmAgent — PydanticAI agent for VLM image captioning.

Routes through the unified :class:`KnoxAgentBase` → :class:`KnoxModel` →
``build_model()`` path like every other agent, passing image bytes via a
:class:`pydantic_ai.messages.BinaryContent` part in the user prompt.

Reasoning is forced off (``reasoning_effort="none"``) — image captioning
never benefits from thinking traces, and uncontrolled reasoning was the
root cause of VLM timeouts and reasoning-polluted captions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai.messages import BinaryContent

from knox.agents.base import KnoxAgentBase
from knox.agents.debug import save_debug_prompt
from knox.core.utils.exceptions import LLMOutputError
from knox.llm.format import strip_reasoning
from knox.llm.usage import UsageTracker
from knox.logging import get_logger

logger = get_logger(__name__)

_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# pydantic-ai requires a concrete ImageMediaType; this fallback covers the
# rare case of an unrecognised extension.
_DEFAULT_MEDIA_TYPE = "image/png"


@dataclass
class VlmDeps:
    """Deps for VLM captioning."""

    prompt: str
    image_path: str
    source_id: str = "vlm"
    timeout: float = 120.0


class VlmAgent(KnoxAgentBase[str, VlmDeps]):
    """PydanticAI agent for VLM image captioning.

    Overrides :meth:`run` to attach the image as a ``BinaryContent`` part
    alongside the text prompt, since the standard single-string user
    prompt cannot carry image bytes.
    """

    template_name: str = ""
    system_template_name: str = ""
    result_type: type = str
    deps_type: type[VlmDeps] = VlmDeps

    def __init__(
        self,
        model_string: str,
        *,
        timeout: float = 120.0,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        # Image captioning never benefits from reasoning — force it off so
        # the model doesn't burn time on thinking traces that pollute
        # captions and cause timeouts on slow local vision models.
        super().__init__(
            model_string,
            timeout=timeout,
            reasoning_effort="none",
            usage_tracker=usage_tracker,
        )

    def run(self, deps: VlmDeps, *, message_history: list[Any] | None = None) -> str:  # noqa: PLR6301
        """Run VLM captioning with image input via the unified model path."""
        self._current_source_id = deps.source_id
        image_file = Path(deps.image_path)
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
            logger.warning("VLM call failed for '%s': %s", deps.image_path, exc)
            raise LLMOutputError(f"VLM failed for model '{self._model_string}': {exc}") from exc

        caption = strip_reasoning(result.output).strip()

        save_debug_prompt(
            self,
            self._model_string,
            [{"role": "user", "content": deps.prompt}],
            {"model": self._model_string, "image_bytes": len(image_bytes)},
            caption,
            "stop",
        )
        return caption
