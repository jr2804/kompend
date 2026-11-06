"""The one home for provider/model string parsing.

Kompend model strings come in a ``provider/model`` slash form
(``ollama/llama3``, ``openai/gpt-4o:2024-11-20``), a ``provider:model``
colon form (``ollama:qwen3:latest``, ``ollama:granite-embedding:278m``),
and a provider-implied bare form (``llama3`` — a model resolved by name
alone, with no provider stated). Model names may carry further separators,
including Hugging Face ids (``ollama/hf.co/Qwen/Qwen3-4B``).

The provider/model separator is the first ``/`` or ``:`` in the string,
whichever comes first; a separator at position 0 (empty provider) is not
a separator. The provider part is everything before it, the model part
everything after it, verbatim. The bare form has no provider
(``provider_id_of`` returns ``None``) and the whole string as the model
part. These are the same rules the build path (``kompend.llm.models``)
splits by, so every consumer of a model string answers "which provider is
this model" identically.
"""

from __future__ import annotations

__all__ = ["model_slug", "provider_id_of"]


def provider_id_of(model_string: str) -> str | None:
    """Return the provider part of a model string, or ``None`` when bare."""
    provider, _ = _split_model_id(model_string)
    return provider


def model_slug(model_string: str) -> str:
    """Return the model part of a model string — the string minus its provider prefix."""
    _, model = _split_model_id(model_string)
    return model


def _split_model_id(model_string: str) -> tuple[str | None, str]:
    """Split *model_string* at its first ``/`` or ``:``, whichever comes first."""
    candidates = [pos for pos in (model_string.find("/"), model_string.find(":")) if pos > 0]
    if not candidates:
        return None, model_string
    pos = min(candidates)
    return model_string[:pos], model_string[pos + 1 :]
