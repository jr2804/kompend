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

from kompend.compilation.flows import references as ref_flow
from kompend.compilation.flows.references import run_reference_extraction
from kompend.core.models import Chunk, ElementType, ResolvedChunk
from kompend.core.models.references import Reference
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
        assert ref_flow._split_windows(text, 100) == [text]

    def test_windows_never_exceed_max_chars(self) -> None:
        text = "\n".join("x" * 10 for _ in range(20))
        windows = ref_flow._split_windows(text, 25)
        assert len(windows) > 1
        assert all(len(w) <= 25 for w in windows)
        assert "\n".join(windows) == text

    def test_oversized_line_is_hard_split(self) -> None:
        windows = ref_flow._split_windows("y" * 100, 30)
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
    compiler = _compiler(max_input=1_500, ref_output_max=400)
    budget = ref_flow._reference_char_budget(compiler)

    run_reference_extraction(compiler, _chunks(text), store)

    assert agent.calls > 1
    assert all(size <= budget for size in agent.window_chars)
    # Whole lines survive: joining the windows reconstructs the section.
    assert len(store.saved[SOURCE]) == agent.calls
    assert all(r.citing_source == SOURCE for r in store.saved[SOURCE])


def test_window_failure_keeps_earlier_references(flow_env: Callable[[FakeAgent], FakeStore]) -> None:
    agent = FakeAgent(fail_on={2})
    store = flow_env(agent)
    text = "\n".join(f"Reference entry number {i} with some detail." for i in range(120))
    compiler = _compiler(max_input=1_500, ref_output_max=400)

    run_reference_extraction(compiler, _chunks(text), store)

    assert agent.calls == 2  # failed on the second window and stopped
    assert len(store.saved[SOURCE]) == 1  # window 1 results survive
    assert store.save_calls == 1  # partial results are still persisted


@pytest.fixture
def flow_env(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeAgent], FakeStore]:
    """Bypass heading detection and agent construction; returns a store factory."""

    def install(agent: FakeAgent) -> FakeStore:
        monkeypatch.setattr(ref_flow, "excise_references", lambda text: ("", text))
        monkeypatch.setattr(ref_flow, "ReferenceExtractionAgent", lambda model_string: agent)
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
    budget = ref_flow._reference_char_budget(_compiler(max_input=600, ref_output_max=500))
    assert budget >= 1


def _compiler(max_input: int = 65_536, ref_output_max: int = 8192) -> SimpleNamespace:
    return SimpleNamespace(
        _model="ollama/test",
        _timeout=30.0,
        _context_window=ContextWindow(max_input=max_input, max_output=max_input, source=ContextSource.OLLAMA),
        _budget=SimpleNamespace(reference_output_max=ref_output_max),
        _config=SimpleNamespace(
            llm=SimpleNamespace(default_chars_per_token=4.0),
            paths=SimpleNamespace(base_dir=Path(".")),
        ),
    )
