"""The judge machinery behind one parse run.

``execute`` (parsecraft's pipeline) takes a ``RoutingJudge``: either the
configured judge resolved from its spec, or the deterministic rule table
that fills in when no judge is configured, the spec fails to resolve, or a
live judge dies on its first call. This module owns that resolution and
degradation policy as free functions of explicit values; the parser keeps
the state (the memo slots) and delegates.

The degradation contract (do not simplify): ``resolve_judge`` validates the
*spec*, never the *endpoint*, so a dead judge surfaces as a ``RoutingError``
from the first ``rank`` call. A judge failure must cost a degraded plan, not
a failed ingest — retry exactly once with the deterministic rule table, warn
once per run, and never absorb ``NoEligibleBackendError`` (no judge can rank
candidates that do not exist — that is a genuine routing dead-end) or a
failure from the deterministic judge itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from parsecraft.backends import BackendRegistry
from parsecraft.backends.protocol import SourceDocument
from parsecraft.environment import EnvironmentInfo
from parsecraft.pipeline import PipelineResult, execute
from parsecraft.pipeline.analysis import AnalysisResult
from parsecraft.routing import NoEligibleBackendError, RoutingConstraints, RoutingError
from parsecraft.routing.judge import DeterministicJudge, RoutingJudge
from parsecraft.routing.judge_providers import JudgeError, resolve_judge

from kompend.core.types import ReasonCode
from kompend.logging import get_logger
from kompend.preprocessing.parsers.parsecraft_routing import ParsecraftRoutingConfig

#: Characters of a judge/SDK error message kept in a log line. The SDK wraps
#: provider errors with the full HTTP exchange; the first clause names the
#: cause and the rest is noise in a per-file warning.
_JUDGE_ERROR_EXCERPT_CHARS: Final[int] = 160

logger = get_logger(__name__)


def execute_with_judge(
    analysis: AnalysisResult,
    registry: BackendRegistry,
    constraints: RoutingConstraints,
    source: SourceDocument,
    *,
    judge: RoutingJudge,
    degraded: bool,
    on_degrade: Callable[[BaseException], RoutingJudge],
) -> PipelineResult:
    """Run ``execute`` with the configured judge, degrading to the rule table once.

    A dead judge endpoint must cost a degraded plan, not a failed ingest —
    therefore: retry exactly once with the deterministic rule table, then let
    the caller remember the verdict for the rest of the run so every later
    file skips the doomed call.

    ``NoEligibleBackendError`` is re-raised: no judge can rank candidates
    that do not exist, so it is a genuine routing dead-end and the caller
    must see it (this is why the handler cannot simply catch
    ``RoutingError``). A failure from the deterministic judge is re-raised
    for the same reason — the retry is not a general-purpose error
    absorber.
    """
    try:
        return execute(analysis, registry, constraints, source, judge=judge)
    except NoEligibleBackendError:
        raise
    except RoutingError as exc:
        if degraded or isinstance(judge, DeterministicJudge):
            raise
        fallback = on_degrade(exc)
        return execute(analysis, registry, constraints, source, judge=fallback)


def get_judge(
    settings: ParsecraftRoutingConfig,
    environment: EnvironmentInfo,
    *,
    judge: RoutingJudge | None,
    fallback: RoutingJudge | None,
) -> tuple[RoutingJudge, RoutingJudge | None, RoutingJudge | None]:
    """Resolve the routing judge, or hand back the memo the caller already holds.

    Three exit paths, in order:

    1. Neither ``PARSECRAFT_JUDGE_MODEL`` nor ``[judge] model`` names a
       judge (the default): return a fresh :class:`DeterministicJudge`
       every call. Cheap (in-process rule table) and avoids sharing state
       between documents.
    2. A spec is configured: resolve it through parsecraft's
       :func:`resolve_judge` and cache the verdict. A second parse reuses
       the same judge instance, so the SDK's intra-instance memo (per
       page class) survives across the run. The spec is parsecraft's —
       kompend neither parses nor names its provider.
    3. Resolver raised (spec error, missing credential, unreachable
       endpoint, missing `decision` capability on the ollama model):
       warn once and fall back to :class:`DeterministicJudge`. A dead
       judge endpoint must never fail an ingest — the fallback is a
       degraded parse, not a hard error.

    Resolution happens inside the caller's ``config_scope`` (see
    ``ParsecraftParser.parse``), so the spec and the provider endpoint are
    read from the run's configuration. ``resolve_judge`` validates the
    *spec*, never the *endpoint* (a cloud provider accepts any model token
    and 400s on the first call). The call-time failure is handled by
    :func:`execute_with_judge`.

    Returns ``(judge, judge_memo, fallback_memo)``: the judge to use plus
    the two memo values the caller stores on the parser instance.
    """
    if judge is not None:
        return judge, judge, fallback
    if fallback is not None:
        return fallback, judge, fallback
    spec = settings.judge_spec
    if spec is None:
        rule_table = deterministic_judge(settings, environment)
        return rule_table, judge, rule_table
    try:
        resolved = resolve_judge(
            spec,
            preference=settings.routing.preference,
        )
    except JudgeError as exc:
        # Every resolution failure is a JudgeError subclass: JudgeSpecError
        # (malformed spec), JudgeProviderUnavailableError (no loader / missing
        # credential), JudgeProviderLoadError (extra not installed).
        logger.warning(
            "parsecraft judge %s could not be resolved (%s: %s); falling back to the deterministic rule table.",
            spec,
            type(exc).__name__,
            str(exc)[:_JUDGE_ERROR_EXCERPT_CHARS],
            extra={"reason_code": ReasonCode.JUDGE_DEGRADED},
        )
        rule_table = deterministic_judge(settings, environment)
        return rule_table, judge, rule_table
    return resolved, resolved, fallback


def degrade_judge(settings: ParsecraftRoutingConfig, environment: EnvironmentInfo, exc: BaseException) -> DeterministicJudge:
    """Warn that a live judge failed at call time; return the rule-table fallback.

    The caller drops the memo slots; the warning is the once-per-run record.
    """
    logger.warning(
        "parsecraft judge %s failed at call time (%s: %s); falling back to the deterministic rule table for the rest of this run.",
        settings.judge_spec or "(unresolved)",
        type(exc).__name__,
        str(exc)[:_JUDGE_ERROR_EXCERPT_CHARS],
        extra={"reason_code": ReasonCode.JUDGE_DEGRADED},
    )
    return deterministic_judge(settings, environment)


def deterministic_judge(settings: ParsecraftRoutingConfig, environment: EnvironmentInfo) -> DeterministicJudge:
    """The rule-table judge carrying the configured preference axis and the host's extra set.

    parsecraft's table takes the smallest declared ``estimated_vram_gb``
    inside a family for SPEED *and* BALANCED (balanced is the neutral
    default) and the largest for QUALITY; preference never reorders across
    families. Speed-vs-balanced is therefore a configured value only,
    never a routing difference — a three-way behavioural test cannot
    exist, only a wiring test plus the speed/quality flip.

    The host's installed extras ride along (parsecraft A10): a candidate
    whose ``convert_extra`` is present on this host ranks native instead
    of extra-backed. The set comes from the same probe that feeds
    ``RoutingConstraints``, so rule table and planner cannot disagree
    about the host. Omitting it would tell parsecraft "no host facts" —
    the honest shape for a caller that has none, and a systematic
    over-demotion of every extra-gated convert for one that has them
    (knox-j5ig).
    """
    return DeterministicJudge(settings.routing.preference, installed_extras=environment.installed_extras)
