"""Per-request model selection for the ACP session agent.

The ACP adapter serves a client that brings its own LLM endpoint, so the agent
it builds carries no model of its own. pydantic-ai infers a model eagerly in
``Agent.__init__``, which is why the adapter used to name a placeholder —
and why every session died with ``Unknown model: <placeholder>``. Instead the
model is chosen per request step by pydantic-ai's ``SelectModel`` capability:
:class:`ConfigModelSelector` reads ``compile.llm.default_model`` from the
session's workspace config on the first step and hands the built model to every
later step of that session.

Resolution is deliberately lazy: the server must start with nothing configured
(an editor may configure an endpoint before it prompts), so a missing model
raises :class:`acp.RequestError` from the first step of the first prompt — the
client sees the house sentence, not an internal error. Nothing is cached on a
failed attempt, so a later prompt picks up a model that was configured in the
meantime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import acp
from pydantic_ai.models import Model, ModelSelectionContext

from kompend.core.config import resolve_config
from kompend.llm.messages import NO_MODEL_CONFIGURED_MESSAGE
from kompend.llm.models import build_model

#: JSON-RPC code the client receives with the house sentence. The sentence is
#: the payload of an ``error.message``; the ACP connection forwards a
#: ``RequestError`` verbatim and wraps anything else in "Internal error".
_INTERNAL_ERROR: Final[int] = -32603

#: The house sentence, kept under its own name because the ACP rail and the
#: tests name it here. The wording lives in :mod:`kompend.llm.messages`.
NO_MODEL_MESSAGE: Final[str] = NO_MODEL_CONFIGURED_MESSAGE


class ConfigModelSelector:
    """Select the configured model for a session rooted at *base_dir*.

    Callable per the ``ModelSelector`` contract, so an instance can be handed to
    ``SelectModel``. The model is resolved once per session: a per-step rebuild
    would construct a new provider and HTTP client for every step of every
    turn. A failed resolution caches nothing, so the next step (or the next
    prompt) retries against whatever the config says then.

    Args:
        base_dir: Workspace the session is rooted at — the same directory
            :func:`kompend.acp.harness_adapter.kompend_session_config` passes to
            the tool set and the config engine.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._model: Model | None = None

    def __call__(self, ctx: ModelSelectionContext[None]) -> Model:
        """Return the model for the request step described by *ctx*."""
        del ctx  # every step of a session uses the same workspace model
        if self._model is None:
            self._model = _configured_model(self.base_dir)
        return self._model


def _configured_model(base_dir: Path) -> Model:
    """Build the model named by ``compile.llm.default_model`` for *base_dir*.

    Raises:
        acp.RequestError: the client-visible failure for a session with no model
            configured, or with a model id the model builder rejects.
    """
    model_id = resolve_config(base_dir=base_dir).kompend.compile.llm.default_model
    if not model_id:
        raise acp.RequestError(_INTERNAL_ERROR, NO_MODEL_MESSAGE)
    try:
        return build_model(model_id)
    except ValueError as exc:
        # A configured but unusable model id is a different failure from a
        # missing one: name what the config says rather than the house sentence.
        raise acp.RequestError(_INTERNAL_ERROR, str(exc)) from exc


__all__ = ["NO_MODEL_MESSAGE", "ConfigModelSelector"]
