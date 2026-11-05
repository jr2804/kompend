"""Import-light LLM constants shared across modules.

Importing :mod:`kompend.llm.models` pulls the full pydantic-ai provider
stack (SDK imports dominate interpreter startup). Modules that only need
these string constants — RAG embedders, test fixtures — import them from
here so their import chains stay light.
"""

from __future__ import annotations

# Default base URL for local Ollama — ``ollama`` is not in the models.dev
# catalog, and OllamaProvider has no built-in default (it raises UserError
# when neither ``base_url`` nor ``OLLAMA_BASE_URL`` is given). pydantic-ai's
# OllamaModel uses the OpenAI-compatible endpoint, so ``/v1`` is required in
# the base URL (the OpenAI SDK appends ``/chat/completions`` to it).
OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

#: Ollama's server-wide default context length. ``OLLAMA_NUM_CTX`` is not an
#: alias — the server ignores it, so reading it silently does nothing.
OLLAMA_CONTEXT_LENGTH_ENV = "OLLAMA_CONTEXT_LENGTH"
