"""Unit tests for the reference-extraction flow (compilation/flows/references.py).

Focus: context-window-aware windowing of oversized reference sections —
the flow must split a section that exceeds the discovered budget into
line-aligned windows, merge the per-window results, and keep references
parsed from windows that succeeded before a failure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from kompend.agents.reference_extraction import ReferenceExtractionAgent
from kompend.compilation.flows import references as ref_flow
from kompend.compilation.flows.references import run_reference_extraction
from kompend.core.models import Chunk, ElementType, ResolvedChunk
from kompend.core.models.references import Reference
from kompend.core.text.windows import split_text_windows
from kompend.llm.context_window import ContextSource, ContextWindow

SOURCE = "doc.docx"


@dataclass
class FakeOutput:
    references: list = field(default_factory=list)


class FakeAgent:
    """Records window sizes; emits one reference per successful call."""

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.window_chars: list[int] = []
        self.fail_on = fail_on or set()
        self.calls = 0
        self.init_kwargs: dict = {}

    def run(self, deps) -> FakeOutput:  # noqa: ANN001 — test stub
        self.calls += 1
        if self.calls in self.fail_on:
            msg = f"simulated failure on call {self.calls}"
            raise RuntimeError(msg)
        self.window_chars.append(len(deps.text))
        return FakeOutput(references=[Reference(id=f"r{self.calls}", raw=deps.text[:20])])


class FakeStore:
    def __init__(self) -> None:
        self.saved: dict[str, list] = {}
        self.save_calls = 0

    def add_source(self, source: str, refs: list) -> int:
        self.saved[source] = refs
        return len(refs)

    def save(self) -> None:
        self.save_calls += 1

    def source_count(self) -> int:
        return len(self.saved)


class TestSplitWindows:
    def test_short_text_is_single_window(self) -> None:
        text = "aa\nbb\ncc"
        assert split_text_windows(text, 100) == [text]

    def test_windows_never_exceed_max_chars(self) -> None:
        text = "\n".join("x" * 10 for _ in range(20))
        windows = split_text_windows(text, 25)
        assert len(windows) > 1
        assert all(len(w) <= 25 for w in windows)
        assert "".join(windows) == text

    def test_oversized_line_is_hard_split(self) -> None:
        windows = split_text_windows("y" * 100, 30)
        assert [len(w) for w in windows] == [30, 30, 30, 10]


def test_small_section_runs_in_one_window(flow_env: Callable[[FakeAgent], FakeStore]) -> None:
    agent = FakeAgent()
    store = flow_env(agent)
    text = "Ref one\nRef two\n"
    budget = ref_flow._reference_char_budget(_compiler())

    run_reference_extraction(_compiler(), _chunks(text), store)

    assert agent.window_chars == [len(text)]
    assert agent.window_chars[0] <= budget
    assert store.saved[SOURCE]
    assert all(r.citing_source == SOURCE for r in store.saved[SOURCE])
    assert store.save_calls == 1


def test_oversized_section_split_into_line_aligned_windows(flow_env: Callable[[FakeAgent], FakeStore]) -> None:
    agent = FakeAgent()
    store = flow_env(agent)
    text = "\n".join(f"[{i}] Author, Title, Journal 20{i % 10}, pp. 1-99." for i in range(120))
    compiler = _compiler(max_input=1_500)
    budget = ref_flow._reference_char_budget(compiler)

    run_reference_extraction(compiler, _chunks(text), store)

    assert agent.calls > 1
    assert all(size <= budget for size in agent.window_chars)
    # Whole lines survive: joining the windows reconstructs the section.
    assert len(store.saved[SOURCE]) == agent.calls
    assert all(r.citing_source == SOURCE for r in store.saved[SOURCE])


def test_window_failure_keeps_references_from_other_windows(flow_env: Callable[[FakeAgent], FakeStore]) -> None:
    agent = FakeAgent(fail_on={2})
    store = flow_env(agent)
    text = "\n".join(f"Reference entry number {i} with some detail." for i in range(120))
    compiler = _compiler(max_input=1_500)

    run_reference_extraction(compiler, _chunks(text), store)

    assert agent.calls > 2  # the failed window does not stop the remaining ones
    assert len(store.saved[SOURCE]) == agent.calls - 1  # every successful window contributes
    assert store.save_calls == 1  # partial results are still persisted


def test_agent_receives_configured_limits(flow_env: Callable[[FakeAgent], FakeStore]) -> None:
    agent = FakeAgent()
    store = flow_env(agent)
    compiler = _compiler(max_input=1_500, output_retries=3)

    run_reference_extraction(compiler, _chunks("Ref one\n"), store)

    assert "max_tokens" not in agent.init_kwargs
    assert agent.init_kwargs["context_window"] == 1_500
    assert agent.init_kwargs["timeout"] == 30.0
    assert agent.init_kwargs["delta_retry_enabled"] is True
    # The retry budget is not pinned at the construction site - the agent
    # derives it from the llm config (tool budget and output budget are
    # separate knobs).
    assert "retries" not in agent.init_kwargs
    assert agent.init_kwargs["llm"] is compiler._config.llm
    # The salvage hook is a class-level opt-in of the agent, not a
    # constructor argument (the reference agent enables it).
    assert ReferenceExtractionAgent.supports_delta_retry is True


def test_agent_retry_budget_comes_from_the_config(
    flow_env: Callable[[FakeAgent], FakeStore],
) -> None:
    agent = FakeAgent()
    store = flow_env(agent)
    compiler = _compiler(output_retries=4, tool_retries=7, output_retries_enabled=False)

    run_reference_extraction(compiler, _chunks("Ref one\n"), store)

    llm = agent.init_kwargs["llm"]
    assert llm.output_retries_enabled is False
    assert llm.tool_retries == 7


def test_every_window_is_processed(flow_env: Callable[[FakeAgent], FakeStore]) -> None:
    """No window is skipped to save cost — every window of an oversized section is parsed."""
    agent = FakeAgent()
    store = flow_env(agent)
    text = "\n".join(f"Reference entry number {i} with some detail." for i in range(120))
    compiler = _compiler(max_input=1_500)

    run_reference_extraction(compiler, _chunks(text), store)

    assert agent.calls > 2
    assert len(store.saved[SOURCE]) == agent.calls


def test_window_budget_follows_the_answer_reserve() -> None:
    """The per-window size is the smaller of the input room and the answer room."""
    compiler = _compiler(max_input=65_536)
    cw = compiler._context_window

    budget = ref_flow._reference_char_budget(compiler)
    answer_chars = ref_flow.tokens_to_chars(
        cw.answer_reserve // ref_flow._REFERENCE_JSON_EXPANSION,
        chars_per_token=4.0,
    )

    assert budget == min(ref_flow.tokens_to_chars(cw.input_budget(), chars_per_token=4.0), answer_chars)


def test_reference_answer_room_comes_from_the_window() -> None:
    """No configured ceiling: the answer room is a share of the model window."""
    assert ref_flow._reference_answer_tokens(_compiler(max_input=32_768)) == 8192
    assert ref_flow._reference_answer_tokens(_compiler(max_input=4_096)) == 1024


@pytest.fixture
def flow_env(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeAgent], FakeStore]:
    """Bypass heading detection and agent construction; returns a store factory."""

    def install(agent: FakeAgent) -> FakeStore:
        monkeypatch.setattr(ref_flow, "excise_references", lambda text: ("", text))

        def factory(**kwargs) -> FakeAgent:  # noqa: ANN003 — test stub mirrors the agent constructor
            agent.init_kwargs = kwargs
            return agent

        monkeypatch.setattr(ref_flow, "ReferenceExtractionAgent", factory)
        return FakeStore()

    return install


def test_no_model_detected_but_not_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ref_flow, "excise_references", lambda text: ("", text))
    compiler = _compiler()
    compiler._model = ""  # noqa: SLF001 — no model configured
    store = FakeStore()

    run_reference_extraction(compiler, _chunks("Ref one\n"), store)

    assert store.saved == {}
    assert store.save_calls == 0


def _chunks(text: str) -> list[ResolvedChunk]:
    chunk = Chunk(
        id="c1",
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[0],
        source_file=Path(SOURCE),
    )
    return [ResolvedChunk(chunk=chunk)]


def test_reference_char_budget_stays_positive() -> None:
    budget = ref_flow._reference_char_budget(_compiler(max_input=600))
    assert budget >= 1


def _compiler(
    max_input: int = 65_536,
    max_output: int | None = None,
    output_retries: int = 2,
    tool_retries: int = 3,
    output_retries_enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        _model="ollama/test",
        _timeout=30.0,
        _context_window=ContextWindow(
            max_input=max_input,
            max_output=max_input if max_output is None else max_output,
            source=ContextSource.OLLAMA,
        ),
        _budget=SimpleNamespace(),
        _config=SimpleNamespace(
            llm=SimpleNamespace(
                default_chars_per_token=4.0,
                output_retries=output_retries,
                tool_retries=tool_retries,
                output_retries_enabled=output_retries_enabled,
                reasoning_effort=None,
                delta_retry_enabled=True,
            ),
            paths=SimpleNamespace(base_dir=Path(".")),
        ),
        _usage_tracker=None,
    )
