"""House sentences the LLM layer hands to every surface.

One condition, one sentence. Six surfaces reported a missing model in six
different wordings (some naming the env var, some the config key, some neither),
so a reader had to work out which one applied to the command they had just
run. The sentence below names both the config key that sets a model and the
command that writes it, and every surface raises this constant instead of
spelling out its own.

This module imports nothing: the CLI, the MCP tools and the ACP adapter all
reach it without a cycle, and none of them pulls in pydantic-ai to read a
string.
"""

from __future__ import annotations

from typing import Final

#: What every surface says when no LLM model is configured. It names the config
#: key that holds the model and the command that sets it, because the two
#: alternative ways to fix it (a config file and an environment override) are
#: not obvious from the failure itself.
NO_MODEL_CONFIGURED_MESSAGE: Final[str] = "No LLM model configured. Set compile.llm.default_model or run 'kompend config set'."

__all__ = ["NO_MODEL_CONFIGURED_MESSAGE"]
