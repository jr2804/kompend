"""Tests for the agentic distill session's resume behaviour.

Regression: one session covered every tag group, so a session that exhausted
its request budget lost the pages it *had* saved — kompend fell back to a
deterministic merge of every group and threw the LLM's work away. A resumed
session keeps the pages from every previous round and carries only the groups
still missing one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded

from kompend.agents.distill_agentic import make_agentic_distill_agent
from kompend.agents.types import DistillAgentDeps, DistillPage
from kompend.capture import distill as distill_mod
from kompend.capture.distill import CapturedNote, _distill_agentic

TIMESTAMP = "20261019T120000Z"


class TestDistillSessionResume:
    def test_resumes_for_the_groups_without_a_page(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        seen: list[list[str]] = []

        def _run(deps: DistillAgentDeps) -> str:
            tags = [group["tag"] for group in deps.groups]
            seen.append(tags)
            if len(seen) == 1:
                deps.pages["alpha"] = DistillPage(title="Alpha", content="synthesised alpha", tags=["alpha"])
                raise UsageLimitExceeded("The next request would exceed the request_limit of 12.")
            for tag in tags:
                deps.pages[tag] = DistillPage(title=tag.capitalize(), content=f"synthesised {tag}", tags=[tag])
            return ""

        monkeypatch.setattr(agent, "run", _run)

        written = _distill_agentic(_groups("alpha", "beta"), [], agent, TIMESTAMP, tmp_path)

        assert len(seen) == 2
        assert seen[0] == ["alpha", "beta"]
        assert seen[1] == ["beta"]  # only the group still missing a page
        assert len(written) == 2

    def test_stops_when_a_session_saves_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        seen: list[list[str]] = []

        def _run(deps: DistillAgentDeps) -> str:
            seen.append([group["tag"] for group in deps.groups])
            raise UsageLimitExceeded("The next request would exceed the request_limit of 12.")

        monkeypatch.setattr(agent, "run", _run)

        written = _distill_agentic(_groups("alpha", "beta"), [], agent, TIMESTAMP, tmp_path)

        assert len(seen) == 1, "a round that saves nothing must not be re-run"
        assert len(written) == 2  # both groups still get a deterministic page

    def test_pages_saved_outside_the_payload_do_not_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tag the session was never asked about is not progress."""
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        seen: list[list[str]] = []

        def _run(deps: DistillAgentDeps) -> str:
            seen.append([group["tag"] for group in deps.groups])
            deps.pages["not-requested"] = DistillPage(title="Junk", content="junk")
            return ""

        monkeypatch.setattr(agent, "run", _run)

        _distill_agentic(_groups("alpha"), [], agent, TIMESTAMP, tmp_path)

        assert len(seen) == 1

    def test_kept_pages_survive_into_the_written_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        calls = {"n": 0}

        def _run(deps: DistillAgentDeps) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                deps.pages["alpha"] = DistillPage(title="Alpha", content="LLM WROTE THIS", tags=["alpha"])
                raise UsageLimitExceeded("The next request would exceed the request_limit of 12.")
            return ""

        monkeypatch.setattr(agent, "run", _run)

        written = _distill_agentic(_groups("alpha", "beta"), [], agent, TIMESTAMP, tmp_path)

        bodies = [p.read_text(encoding="utf-8") for p in written]
        assert sum("LLM WROTE THIS" in body for body in bodies) == 1

    def test_group_order_is_preserved_across_rounds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        seen: list[list[str]] = []

        def _run(deps: DistillAgentDeps) -> str:
            tags = [group["tag"] for group in deps.groups]
            seen.append(tags)
            deps.pages[tags[0]] = DistillPage(title=tags[0].capitalize(), content="x", tags=[tags[0]])
            return ""

        monkeypatch.setattr(agent, "run", _run)

        _distill_agentic(_groups("a", "b", "c"), [], agent, TIMESTAMP, tmp_path)

        assert seen == [["a", "b", "c"], ["b", "c"], ["c"]]


def _groups(*tags: str) -> dict[str, list[CapturedNote]]:
    return {tag: [_note(tag)] for tag in tags}


def _note(tag: str) -> CapturedNote:
    return CapturedNote(title=f"{tag} note", content=f"body for {tag}", tags=[tag], source_file=Path(f"{tag}.md"))


def test_no_pending_groups_skips_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty corpus must not spin up a session."""
    agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")

    def _run(deps: DistillAgentDeps) -> str:  # pragma: no cover - must never run
        raise AssertionError("no session expected")

    monkeypatch.setattr(agent, "run", _run)

    assert _distill_agentic({}, [], agent, TIMESTAMP, tmp_path) == []


def test_fallback_helper_is_no_longer_referenced() -> None:
    """The deterministic per-group fallback is inlined; the helper is gone."""
    assert not hasattr(distill_mod, "_distill_fallback_per_group")
