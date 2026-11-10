"""Decision-model checks: the offline result-type grammar and the live ping.

A decision model is a different endpoint family from the chat models the rest
of the doctor pings: it speaks the SystemOne protocol (``/v1/systemone``) and
answers typed questions instead of writing text. Both of its failure classes
are silent in a compile run — a refused result type degrades every judge tier
to its fallback, and a dead endpoint degrades it to the deterministic
classifier — so the doctor asks both questions directly.

- :func:`check_decision_schema` is offline: it builds the request the model
  would build for every shipped result type and lets the decision grammar
  judge it, with no endpoint and no api key involved.
- :func:`check_decision_connectivity` is one live request through the same
  agent the pipeline uses, so the reported cause is the real one (401, 404,
  the zstd frame, a timeout, a refused connection).
"""

from __future__ import annotations

from importlib import import_module
from time import monotonic
from typing import Any, Final

from pydantic import BaseModel
from pydantic_ai.exceptions import UserError

from kompend.agents.decision_probe import probe_decision_model
from kompend.core.config import DecisionConfig
from kompend.core.utils.exceptions import LLMOutputError
from kompend.diagnostics.base import CheckResult
from kompend.llm.decision import DECISION_RESULT_TYPES, decision_model_string

#: Names of the two decision checks. Kebab-case and stable: the doctor JSON,
#: the ``doctor`` MCP tool output and every saved report key on them.
DECISION_SCHEMA_CHECK: Final[str] = "decision-schema"
DECISION_CONNECTIVITY_CHECK: Final[str] = "connectivity-decision"

#: The module that owns the decision grammar. Imported by name so a renamed or
#: removed private helper becomes a reported warning instead of an import
#: failure at doctor start-up.
_GRAMMAR_MODULE: Final[str] = "pydantic_ai.models.decision"

#: The module that builds the output tool the grammar reads.
_OUTPUT_MODULE: Final[str] = "pydantic_ai._output"

#: What a failed ping may raise, as a closed set: the agent base maps every
#: provider failure to :class:`LLMOutputError`; the grammar and the model
#: builder raise ``UserError``/``ValueError``; a broken provider module raises
#: ``ImportError``; a refused connection surfaces as ``OSError``. The tuple is
#: explicit because the check reports a failure as a fact and must not absorb a
#: bug of its own into a green tick.
_BASE_FAILURES: Final[tuple[type[BaseException], ...]] = (LLMOutputError, UserError, ValueError, ImportError, OSError)


def check_decision_schema() -> CheckResult:
    """Validate every shipped decision result type against the real grammar.

    The grammar decides whether a result type can be asked at all, before any
    request is sent: a list of models is refused outright, while a list of
    string options is accepted and then collapses verdicts. Both make a tier
    degrade without an error, which is exactly what this check reports — one
    verdict per type, with the grammar's own stated cause.
    """
    name = DECISION_SCHEMA_CHECK
    try:
        grammar = import_module(_GRAMMAR_MODULE)
        output = import_module(_OUTPUT_MODULE)
        limits = grammar._Limits(choice_options=None, score_levels=None, requires_instructions=False, questions=None)
        build_toolset = output.OutputToolset.build
    except (ImportError, AttributeError) as exc:
        return CheckResult(name, True, f"decision grammar could not be validated: pydantic-ai internals changed ({exc})", warning=True)

    accepted: list[str] = []
    refusals: list[str] = []
    questions = 0
    for label, result_type in DECISION_RESULT_TYPES:
        tool = _output_tool(build_toolset, result_type)
        if grammar._expressible(tool, None, limits):
            accepted.append(label)
            questions += len(grammar._questions(grammar._fields(tool), tool, None, limits, label=None))
        else:
            refusals.append(f"{label}: {_refusal_cause(grammar, tool, limits)}")
    if refusals:
        return CheckResult(name, False, "decision grammar refuses " + "; ".join(refusals))
    listed = ", ".join(accepted)
    return CheckResult(name, True, f"{len(accepted)} decision result types accepted, {questions} typed questions ({listed})")


def check_decision_connectivity(config: DecisionConfig | None) -> CheckResult:
    """Ping the configured decision model once, through the SystemOne endpoint.

    A disabled ``[decision]`` block, or an enabled one with no model, is a
    legitimate configuration: the pipeline keeps its deterministic fallbacks,
    so the check is a warning, not a failure. A configured model that fails is
    a failure — every judge tier silently loses its decisions when it does.
    """
    name = DECISION_CONNECTIVITY_CHECK
    model_string = decision_model_string(config)
    if model_string is None:
        if config is None or not config.enabled:
            return CheckResult(name, True, "decision model disabled — skipped", warning=True)
        return CheckResult(name, True, "decision model enabled but no model configured — skipped", warning=True)

    timeout = config.timeout if config is not None else 10.0
    start = monotonic()
    try:
        result = probe_decision_model(model_string, timeout=timeout)
    except _ping_failures() as exc:
        return CheckResult(name, False, f"decision model {model_string} failed: {exc}")
    elapsed = monotonic() - start
    answered = f"decision model OK ({model_string}) via SystemOne ({elapsed:.1f}s)"
    if not result.replied:
        return CheckResult(name, True, f"{answered} — it answered no to a true statement", warning=True)
    return CheckResult(name, True, answered)


def _ping_failures() -> tuple[type[BaseException], ...]:
    """Every exception a ping may raise, resolved without importing the SDK eagerly.

    The TypeSafe SDK rejects a missing api key in its own constructor, before
    any request: that is the very failure the ``connectivity-decision`` check
    exists to name, so its exception type joins the tuple — imported here, not
    at module level, so ``kompend doctor`` does not pay the vendor import for
    a decision model that is switched off.
    """
    try:
        sdk = import_module("typesafe_sdk")
    except ImportError:
        return _BASE_FAILURES
    return (*_BASE_FAILURES, sdk.TypeSafeError)


def _output_tool(build_toolset: Any, result_type: type[BaseModel]) -> Any:
    """The output tool the decision grammar reads for *result_type*."""
    return build_toolset([result_type])._tool_defs[0]


def _refusal_cause(grammar: Any, tool: Any, limits: Any) -> str:
    """The grammar's own words for why it refuses *tool*."""
    try:
        grammar._questions(grammar._fields(tool), tool, None, limits, label=None)
    except UserError as exc:
        return str(exc)
    return "refused with no stated cause"


__all__ = ["DECISION_CONNECTIVITY_CHECK", "DECISION_SCHEMA_CHECK", "check_decision_connectivity", "check_decision_schema"]
