"""Construct the page-range splitter from resolved settings.

This is the one place that turns configuration into a judge, so wiring a
different judge (another model family, a local service, a fixture in tests) is
a change here alone: the splitter, the pipeline seam and the ledger contract do
not know which judge decided.

Two questions are answered here, both by raising rather than guessing:

* which model asks the boundary questions — ``splitting.judge_model`` first,
  then the ``[decision]`` model when that section is enabled;
* which categories the judge may assign — ``splitting.rules_file`` or the
  shipped list.

When no model resolves, the splitter is built with :class:`NoDecisionModel`
instead of being dropped: the run then records "asked, and there was nothing to
ask" per source, which is how an operator learns that splitting is on but has
no judge. A source is never split on a guess.
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.config._configs import DecisionConfig, SplittingConfig
from kompend.llm.decision import decision_model_string
from kompend.preprocessing.splitting._elements import SourceSplitter
from kompend.preprocessing.splitting._judge import (
    DecisionModelSplitJudge,
    NoDecisionModel,
)
from kompend.preprocessing.splitting._rules import load_rules


def build_splitter(
    splitting: SplittingConfig,
    decision: DecisionConfig,
    *,
    base_dir: Path,
) -> SourceSplitter | None:
    """Build the splitter a run should use, or ``None`` when splitting is off."""
    if not splitting.enabled:
        return None
    model_string = resolve_judge_model(splitting, decision)
    judge = DecisionModelSplitJudge(model_string, timeout=splitting.judge_timeout) if model_string is not None else NoDecisionModel()
    rules = load_rules(resolve_rules_file(splitting, base_dir))
    return SourceSplitter(
        judge=judge,
        rules=rules,
        min_pages=splitting.min_pages,
        max_page_chars=splitting.max_page_chars,
    )


def resolve_judge_model(splitting: SplittingConfig, decision: DecisionConfig) -> str | None:
    """Return the ``provider/model`` that finds boundaries, or ``None``.

    ``splitting.judge_model`` wins outright; otherwise the ``[decision]`` model
    applies when that section is enabled — the boundary question is a typed
    decision like the others, and a vault that configured one judge has
    configured the judge.
    """
    if splitting.judge_model:
        return splitting.judge_model
    return decision_model_string(decision)


def resolve_rules_file(splitting: SplittingConfig, base_dir: Path) -> Path | None:
    """Resolve ``splitting.rules_file`` against the vault root.

    ``None`` means "use the shipped list". A relative path is read from the
    vault, an absolute one as given — the same rule the rest of the config
    follows for vault-relative paths.
    """
    raw = splitting.rules_file
    if raw is None:
        return None
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else base_dir / candidate


__all__ = ["build_splitter", "resolve_judge_model", "resolve_rules_file"]
