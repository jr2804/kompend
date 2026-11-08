"""Lint/evaluation models (``[lint]``)."""

from __future__ import annotations

from pydantic import Field

from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar
from kompend.core.vocab import LintOutput


class LintConfig(_KompendBaseModel):
    """Lint configuration. ``enabled=None`` means all structural rules."""

    enabled: list[str] | None = None
    fuzzy_threshold: float = Field(default=0.85, validation_alias=EnvVar.LINT_FUZZY_THRESHOLD)
    #: Per-call timeout for the LLM lint rules. They are long-context JSON
    #: requests (a batch of page summaries, or a whole page for clarity and
    #: factual checks), so a slow provider needs more than the compile default:
    #: a timed-out rule loses its findings instead of retrying the call.
    llm_timeout: float = Field(default=300.0, validation_alias=EnvVar.LINT_LLM_TIMEOUT)
    #: How much of a lint report reaches the terminal. ``quiet`` (default) prints
    #: the verdict and the path of the full report; ``summary`` adds one line per
    #: failing rule; ``full`` prints every finding and every passing rule. The
    #: full report is always written to ``.kompend/lint-log/``, and
    #: ``--verbose-lint`` / ``--quiet-lint`` override this per run.
    output: LintOutput = Field(default=LintOutput.QUIET, validation_alias=EnvVar.LINT_OUTPUT)
    model: str | None = Field(
        default=None,
        validation_alias=EnvVar.LINT_MODEL,
        description="Model override for LLM-powered lint steps (factual-consistency, clarity-score, ...). Falls back to compile.llm.default_model when unset.",
    )
