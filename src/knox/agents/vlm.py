"""VlmAgent — PydanticAI agent for VLM image captioning.

Special case: handles image input via base64 data URLs in the message
content, which PydanticAI's text-only prompt system doesn't support.
Overrides ``run()`` to construct the litellm call directly.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import litellm

from knox.agents.base import KnoxAgentBase
from knox.agents.litellm_model import _save_debug_prompt, _messages_to_litellm
from knox.core.utils.exceptions import LLMOutputError
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


@dataclass
class VlmDeps:
    """Deps for VLM captioning."""

    prompt: str
    image_path: str
    source_id: str = "vlm"
    timeout: float = 120.0


class VlmAgent(KnoxAgentBase[str, VlmDeps]):
    """PydanticAI agent for VLM image captioning.

    Overrides ``run()`` to construct the litellm call with image content
    directly, since PydanticAI's text-only prompt system doesn't support
    image attachments.
    """

    template_name: str = ""
    system_template_name: str = ""
    result_type: type = str
    deps_type: type[VlmDeps] = VlmDeps

    def render_user_prompt(self, deps: VlmDeps) -> str:
        return deps.prompt

    def render_system_prompt(self, deps: VlmDeps) -> str:
        return ""

    def run(self, deps: VlmDeps) -> str:
        """Run VLM captioning with image input."""
        image_file = Path(deps.image_path)
        if not image_file.is_file():
            msg = f"Image not found: {image_file}"
            raise FileNotFoundError(msg)

        raw_bytes = image_file.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        mime = _MIME_BY_EXT.get(image_file.suffix.lower(), "image/png")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": deps.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ]

        kwargs: dict = {
            "model": self._litellm_model,
            "messages": messages,
            "timeout": self._timeout,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens

        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            logger.warning("VLM call failed for '%s': %s", deps.image_path, exc)
            raise LLMOutputError(f"VLM failed for model '{self._litellm_model}': {exc}") from exc

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        # Track usage
        if self._usage_tracker is not None:
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._usage_tracker.record(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    model=self._litellm_model,
                )

        _save_debug_prompt(self._litellm_model, messages, kwargs, content, finish_reason)
        return content.strip()
