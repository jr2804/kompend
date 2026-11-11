"""The agent error boundary: what may escape ``run()``.

Every pipeline stage catches one type — ``LLMOutputError`` — and decides from
its subclass and ``__cause__`` whether to retry, skip or abort. That contract
only holds while the base package maps failures in exactly the declared places,
so these tests pin both halves: the mapping itself, and the fact that no new
broad catch appears in ``agents/base/**`` to route a failure around it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Final

import pytest

from kompend.agents.base import KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps
from kompend.core.config._configs import LLMConfig
from kompend.core.utils.exceptions import LLMContextOverflowError, LLMOutputError, LLMTransientError

_BASE_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "kompend" / "agents" / "base"

#: How many broad catches each module of ``agents/base/**`` may contain.
#:
#: ``agent.py`` (2) and ``retries.py`` (1) are THE mapping boundary: each one
#: converts whatever the provider raised into a typed ``LLMOutputError`` (or
#: re-raises it), so nothing bare reaches a stage. ``delta_retry.py`` (1) is the
#: salvage probe: a schema-validation failure becomes ``ModelRetry`` with a
#: delta message. A module absent here may not catch broadly at all — a new
#: entry is a new place a failure can lose its cause, so add one only with a
#: reason and a line in ``src/kompend/agents/AGENTS.md``.
_ALLOWED_BROAD_CATCHES: Final[dict[str, int]] = {
    "agent.py": 2,
    "retries.py": 1,
    "delta_retry.py": 1,
}


_DEPS = PipelineAgentDeps(text="source body", source_id="src-1")


class _FailingAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """Raises a fixed exception from the model call, whatever the input."""

    def __init__(self, error: BaseException, **kwargs: Any) -> None:
        super().__init__(model_string="ollama/noop", **kwargs)
        self._error = error

    def _invoke(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error


class _FakeModelAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """Runs the real retry loop against a model object that always fails."""

    def __init__(self, error: BaseException, **kwargs: Any) -> None:
        super().__init__(model_string="ollama/noop", **kwargs)
        self._error = error
        self.calls = 0

    def _build_agent(self) -> Any:
        calls = self

        class _Run:
            def run_sync(self, *_args: Any, **_kwargs: Any) -> Any:
                calls.calls += 1
                raise calls._error

        return _Run()


class TestMappingBoundary:
    def test_a_plain_failure_emerges_as_llm_output_error(self) -> None:
        """Anything the provider raises is an ``LLMOutputError`` by the time it leaves."""
        agent = _FailingAgent(ValueError("the model answered nonsense"))

        with pytest.raises(LLMOutputError) as caught:
            agent.run(_DEPS, user_prompt="summarise this")

        assert type(caught.value) is LLMOutputError

    def test_the_cause_is_the_original_exception(self) -> None:
        """The stage must be able to tell transport from semantics from ``__cause__``."""
        original = ValueError("the model answered nonsense")
        agent = _FailingAgent(original)

        with pytest.raises(LLMOutputError) as caught:
            agent.run(_DEPS, user_prompt="summarise this")

        assert caught.value.__cause__ is original

    def test_run_with_messages_maps_the_same_way(self) -> None:
        """The second entry point shares the boundary; neither may drift."""
        original = ValueError("the model answered nonsense")
        agent = _FailingAgent(original)

        with pytest.raises(LLMOutputError) as caught:
            agent.run_with_messages(_DEPS, user_prompt="summarise this")

        assert caught.value.__cause__ is original

    def test_a_context_overflow_keeps_the_provider_limit(self) -> None:
        """The subclass carries the number the server named, not just the message."""

        class _Overflow(Exception):
            status_code = 400

            def __str__(self) -> str:
                return "request exceeds the available context size (32768 tokens)"

        agent = _FailingAgent(_Overflow())

        with pytest.raises(LLMContextOverflowError) as caught:
            agent.run(_DEPS, user_prompt="summarise this")

        assert caught.value.stated_limit == 32768
        assert caught.value.__cause__ is not None


class TestTransientRetry:
    def test_a_transient_failure_retries_then_surfaces_typed(self) -> None:
        """Retried while the policy allows, then raised as ``LLMTransientError``."""
        original = TimeoutError("read timed out")
        agent = _FakeModelAgent(original, llm=_llm(retries=2))

        with pytest.raises(LLMTransientError) as caught:
            agent.run(_DEPS, user_prompt="summarise this")

        assert agent.calls == 3, "one attempt plus two configured retries"
        assert type(caught.value) is LLMTransientError
        assert caught.value.__cause__ is original

    def test_a_semantic_failure_is_not_retried(self) -> None:
        """Repeating a request the model already answered the same way only burns budget."""
        agent = _FakeModelAgent(ValueError("bad JSON"), llm=_llm(retries=2))

        with pytest.raises(LLMOutputError):
            agent.run(_DEPS, user_prompt="summarise this")

        assert agent.calls == 1

    def test_the_retry_budget_follows_the_config(self) -> None:
        """The loop is the only retry policy; its count comes from ``compile.llm``."""
        agent = _FakeModelAgent(TimeoutError("read timed out"), llm=_llm(retries=4))

        with pytest.raises(LLMTransientError):
            agent.run(_DEPS, user_prompt="summarise this")

        assert agent.calls == 5


class TestOnlyTheBoundaryCatchesBroadly:
    def test_base_package_has_exactly_the_known_broad_catches(self) -> None:
        """A new broad catch in ``base/**`` would route a failure around the mapping."""
        found = {path.name: len(_broad_catches(path)) for path in sorted(_BASE_PACKAGE.glob("*.py"))}
        unexpected = {name: count for name, count in found.items() if _ALLOWED_BROAD_CATCHES.get(name, 0) != count}

        assert not unexpected, (
            f"agents/base/** broad catches changed: {unexpected}; every entry needs a reason in "
            "_ALLOWED_BROAD_CATCHES and a line in src/kompend/agents/AGENTS.md"
        )


def _llm(*, retries: int = 2) -> LLMConfig:
    """Config with no real backoff — the test measures attempts, not waits."""
    return LLMConfig(transient_retries=retries, transient_backoff=0.0)


def _broad_catches(path: Path) -> list[int]:
    """Return the line numbers of every bare/``Exception``/``BaseException`` handler."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = node.type
        is_broad = caught is None or (isinstance(caught, ast.Name) and caught.id in {"Exception", "BaseException"})
        if is_broad:
            found.append(node.lineno)
    return found
