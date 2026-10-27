"""Remediation must use a page-writing agent and never abort the lint run."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from kompend.core.models import CompiledWiki
from kompend.core.utils.exceptions import LLMOutputError
from kompend.lint import _remediation_gaps, flow


class TestRemediationAgent:
    def test_uses_the_agentic_remediation_agent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        built: dict[str, Any] = {}
        passed: dict[str, Any] = {}

        def _factory(**kwargs: Any) -> object:
            built.update(kwargs)
            return object()

        def _remediate(wiki: object, report: object, **kwargs: Any) -> list[Any]:
            passed.update(kwargs)
            return []

        monkeypatch.setattr(flow, "make_agentic_remediation_agent", _factory)
        monkeypatch.setattr(flow, "remediate", _remediate)

        flow._remediate_phase(CompiledWiki(nodes=[], edges=[], toc=[]), _vault(tmp_path), _config(), [], True)  # type: ignore[arg-type]

        assert built["model_string"] == "ollama/test-model"
        assert built["timeout"] == 300.0
        assert passed["agent"] is not None
        assert passed["dry_run"] is False

    def test_no_model_skips_remediation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _factory(**_kwargs: Any) -> object:
            raise AssertionError("no agent should be built without a model")

        monkeypatch.setattr(flow, "make_agentic_remediation_agent", _factory)

        actions = flow._remediate_phase(CompiledWiki(nodes=[], edges=[], toc=[]), _vault(tmp_path), _config(None), [], True)  # type: ignore[arg-type]

        assert actions == []


class TestFailedPageGeneration:
    def test_a_failed_page_is_skipped_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Failing:
            def run(self, _deps: object) -> str:
                raise LLMOutputError("Exceeded maximum output retries (2)")

        content = _remediation_gaps._generate_concept_page(_Failing(), {"name": "ABLE model"})  # type: ignore[arg-type]

        assert content is None


def _vault(tmp_path: Path) -> Any:
    return types.SimpleNamespace(
        base_dir=tmp_path,
        output_dir=tmp_path / "wiki",
        agent_memory_dir=tmp_path / ".kompend" / "agent-memory",
    )


def _config(model: str | None = "ollama/test-model") -> Any:
    return types.SimpleNamespace(
        kompend=types.SimpleNamespace(
            lint=types.SimpleNamespace(model=None, llm_timeout=300.0),
            compile=types.SimpleNamespace(
                llm=types.SimpleNamespace(default_model=model),
            ),
        )
    )
