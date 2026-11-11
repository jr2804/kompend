"""Decision-model checks: the offline result-type grammar and the live ping.

A decision model is a different endpoint family from the chat models the rest
of the doctor pings: it speaks the SystemOne protocol (``/v1/systemone``) and
answers typed questions instead of writing text. Both of its failure classes
are silent in a compile run — a refused result type degrades every judge tier
to its fallback, and a dead endpoint degrades it to the deterministic
classifier — so the doctor asks both questions directly.

- :func:`check_decision_schema` is offline: it builds the request the model
  would build for every shipped result type — the four static ones in
  :data:`kompend.llm.decision.DECISION_RESULT_TYPES` and one representative of
  the dynamic split-boundary family — and lets the decision grammar judge it
  against the caps the shipped model enforces, with no endpoint and no api key
  involved.
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

#: The shipped decision model, read by name for the caps its endpoint enforces.
_DECISION_MODEL_MODULE: Final[str] = "pydantic_ai.models.typesafe"

#: The public factory of the dynamic split-boundary verdict model.
_SPLITTING_MODULE: Final[str] = "kompend.preprocessing.splitting"

#: The packet length a split-boundary verdict is validated at: ``SourceSplitter``
#: defaults ``min_pages`` to six, so six is the narrowest packet that reaches a
#: judge, and the verdict then carries the five candidate cut points.
_BOUNDARY_PAGES: Final[int] = 6

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
    request is sent: a list of models is refused outright, a list of string
    options is accepted and then collapses verdicts, and a pick-one wider than
    the endpoint's cap is refused too. All three make a tier degrade without an
    error, which is exactly what this check reports — one verdict per type, with
    the grammar's own stated cause.

    The dynamic split-boundary family cannot be enumerated: it mints one boolean
    per candidate cut, so its shape varies per packet. One representative is
    validated instead, so a regression that only shows on wider packets is not
    reported as clean for the whole family.
    """
    name = DECISION_SCHEMA_CHECK
    try:
        grammar = import_module(_GRAMMAR_MODULE)
        output = import_module(_OUTPUT_MODULE)
        limits = _shipped_limits(grammar)
        build_toolset = output.OutputToolset.build
    except (ImportError, AttributeError) as exc:
        return CheckResult(name, True, f"decision grammar could not be validated: pydantic-ai internals changed ({exc})", warning=True)

    family: list[tuple[str, type[BaseModel]]] = list(DECISION_RESULT_TYPES)
    unchecked: str | None = None
    try:
        family.extend(_split_boundary_family())
    except (ImportError, AttributeError, ValueError) as exc:
        # The splitting package is optional for a doctor run on a tree where it is
        # not installed; an unvalidated family is a warning, never a failure.
        unchecked = f"the dynamic split-boundary verdict went unchecked ({exc})"

    accepted: list[str] = []
    refusals: list[str] = []
    questions = 0
    for label, result_type in family:
        tool = _output_tool(build_toolset, result_type)
        if grammar._expressible(tool, None, limits):
            accepted.append(label)
            questions += len(grammar._questions(grammar._fields(tool), tool, None, limits, label=None))
        else:
            refusals.append(f"{label}: {_refusal_cause(grammar, tool, limits)}")
    suffix = f" — {unchecked}" if unchecked is not None else ""
    if refusals:
        return CheckResult(name, False, "decision grammar refuses " + "; ".join(refusals) + suffix)
    listed = ", ".join(accepted)
    message = f"{len(accepted)} decision result types accepted, {questions} typed questions ({listed})"
    return CheckResult(name, True, message + suffix, warning=unchecked is not None)


def _shipped_limits(grammar: Any) -> Any:
    """The ``_Limits`` the shipped decision model enforces, read from its class variables.

    The real request-time limits come from the model class — a profile may refine
    them, and the class's values are the fallback where it leaves them out — so
    they are read here rather than repeated: Jev picks from at most 255 options
    and scores at most 10 rubric levels today, and an upstream change to those
    moves this check with it instead of letting it accept a type the endpoint
    would refuse with a 400. The class inherits the grammar's own defaults, so a
    cap it stops declaring falls back to the unbounded base by itself.
    """
    model_class = import_module(_DECISION_MODEL_MODULE).TypeSafeModel
    return grammar._Limits(
        choice_options=model_class.max_choice_options,
        score_levels=model_class.max_score_levels,
        requires_instructions=model_class.requires_instructions,
        questions=model_class.max_questions,
    )


def _split_boundary_family() -> list[tuple[str, type[BaseModel]]]:
    """Two representatives of the dynamic split-boundary verdict family.

    Imported lazily: the doctor must not pull the preprocessing machinery in at
    start-up for a check that is only about the result shape. The verdict would
    otherwise go unchecked for the whole family, and a regression that only
    shows on wider packets would be reported as clean — so the narrowest packet
    that reaches a judge is validated, and so is the widest window a long packet
    is ever asked about.
    """
    module = import_module(_SPLITTING_MODULE)
    return [
        (f"split-boundary-verdict({_BOUNDARY_PAGES} pages)", module.boundary_verdict_model(_BOUNDARY_PAGES)),
        (
            f"split-boundary-window({module.MAX_QUESTIONS_PER_REQUEST} transitions)",
            module.window_verdict_model(1, module.MAX_QUESTIONS_PER_REQUEST),
        ),
    ]


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
