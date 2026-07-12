"""Tests for the WiCER agents (knox-pyda Phase 5).

Three agents cover the WiCER compile-evaluate-refine loop:
- WicerProbeAgent (generate Q&A probes)
- WicerEvaluateAgent (judge if wiki answers probes)
- WicerDiagnoseAgent (extract critical facts from failures)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from knox.agents.deps import WicerDiagnoseDeps, WicerEvaluateDeps, WicerProbeDeps
from knox.agents.wicer import WicerDiagnoseAgent, WicerEvaluateAgent, WicerProbeAgent
from knox.compilation.schema import (
    WicerDiagnoseOutput,
    WicerEvalResult,
    WicerEvaluateOutput,
    WicerProbe,
    WicerProbeOutput,
)

# ── WicerProbeAgent ──────────────────────────────────────────────────────


class TestWicerProbeAgent:
    def test_class_metadata(self) -> None:
        assert WicerProbeAgent.template_name == "wicer_probe.md.j2"
        assert WicerProbeAgent.result_type is WicerProbeOutput

    def test_run_returns_probe_output(self) -> None:
        agent, _ = self._make_agent_and_mock()
        deps = WicerProbeDeps(text="Source text.", source_id="batch-1")
        result = agent.run(deps)
        assert isinstance(result, WicerProbeOutput)
        assert len(result.probes) == 1
        assert result.probes[0].question == "What is X?"

    def test_run_passes_text_in_user_prompt(self) -> None:
        agent, mock_sync = self._make_agent_and_mock()
        deps = WicerProbeDeps(text="PAYLOAD-TEXT-XYZ", source_id="batch-1")
        agent.run(deps)
        assert "PAYLOAD-TEXT-XYZ" in mock_sync.call_args.args[0]

    def test_run_wraps_exception(self) -> None:
        agent = WicerProbeAgent(litellm_model="ollama/gpt-oss:20b")
        agent._agent = MagicMock()
        agent._agent.run_sync = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(Exception, match="Agent failed for model") as exc_info:
            agent.run(WicerProbeDeps(text="x", source_id="s"))
        assert "ollama/gpt-oss:20b" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def _make_agent_and_mock(self) -> tuple[WicerProbeAgent, MagicMock]:
        agent = WicerProbeAgent(litellm_model="ollama/gpt-oss:20b")
        mock_output = WicerProbeOutput(
            probes=[WicerProbe(question="What is X?", answer="X is Y.")],
        )
        mock_sync = MagicMock(return_value=MagicMock(output=mock_output))
        agent._agent = MagicMock()
        agent._agent.run_sync = mock_sync
        return agent, mock_sync


# ── WicerEvaluateAgent ───────────────────────────────────────────────────


class TestWicerEvaluateAgent:
    def test_class_metadata(self) -> None:
        assert WicerEvaluateAgent.template_name == "wicer_evaluate.md.j2"
        assert WicerEvaluateAgent.result_type is WicerEvaluateOutput

    def test_run_returns_evaluate_output(self) -> None:
        agent, _ = self._make_agent_and_mock()
        deps = WicerEvaluateDeps(wiki_text="wiki content", probes_json='[{"question":"Q?"}]', source_id="eval-1")
        result = agent.run(deps)
        assert isinstance(result, WicerEvaluateOutput)
        assert len(result.results) == 1
        assert result.results[0].correct is True

    def test_run_passes_wiki_and_probes_in_prompt(self) -> None:
        agent, mock_sync = self._make_agent_and_mock()
        deps = WicerEvaluateDeps(
            wiki_text="WIKI-PAYLOAD",
            probes_json="PROBES-PAYLOAD",
            source_id="eval-1",
        )
        agent.run(deps)
        prompt = mock_sync.call_args.args[0]
        assert "WIKI-PAYLOAD" in prompt
        assert "PROBES-PAYLOAD" in prompt

    def _make_agent_and_mock(self) -> tuple[WicerEvaluateAgent, MagicMock]:
        agent = WicerEvaluateAgent(litellm_model="ollama/gpt-oss:20b")
        mock_output = WicerEvaluateOutput(
            results=[WicerEvalResult(question="Q?", expected_answer="A", found_answer="A", correct=True)],
        )
        mock_sync = MagicMock(return_value=MagicMock(output=mock_output))
        agent._agent = MagicMock()
        agent._agent.run_sync = mock_sync
        return agent, mock_sync


# ── WicerDiagnoseAgent ───────────────────────────────────────────────────


class TestWicerDiagnoseAgent:
    def test_class_metadata(self) -> None:
        assert WicerDiagnoseAgent.template_name == "wicer_diagnose.md.j2"
        assert WicerDiagnoseAgent.result_type is WicerDiagnoseOutput

    def test_run_returns_diagnose_output(self) -> None:
        agent, _ = self._make_agent_and_mock()
        deps = WicerDiagnoseDeps(failures_json='[{"question":"Q"}]', source_id="diag-1")
        result = agent.run(deps)
        assert isinstance(result, WicerDiagnoseOutput)
        assert result.critical_facts == ["Fact A", "Fact B"]

    def test_run_passes_failures_in_prompt(self) -> None:
        agent, mock_sync = self._make_agent_and_mock()
        deps = WicerDiagnoseDeps(failures_json="FAILURES-PAYLOAD", source_id="diag-1")
        agent.run(deps)
        assert "FAILURES-PAYLOAD" in mock_sync.call_args.args[0]

    def _make_agent_and_mock(self) -> tuple[WicerDiagnoseAgent, MagicMock]:
        agent = WicerDiagnoseAgent(litellm_model="ollama/gpt-oss:20b")
        mock_output = WicerDiagnoseOutput(critical_facts=["Fact A", "Fact B"])
        mock_sync = MagicMock(return_value=MagicMock(output=mock_output))
        agent._agent = MagicMock()
        agent._agent.run_sync = mock_sync
        return agent, mock_sync


# ── Deps defaults ────────────────────────────────────────────────────────


class TestWicerDepsDefaults:
    def test_probe_deps_defaults(self) -> None:
        deps = WicerProbeDeps(text="x", source_id="s")
        assert deps.timeout == 120.0

    def test_evaluate_deps_defaults(self) -> None:
        deps = WicerEvaluateDeps(wiki_text="x", probes_json="[]", source_id="s")
        assert deps.timeout == 120.0

    def test_diagnose_deps_defaults(self) -> None:
        deps = WicerDiagnoseDeps(failures_json="[]", source_id="s")
        assert deps.timeout == 120.0
