"""Tests for the shared agentic capability bundle."""

from __future__ import annotations

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.summarizer_agentic import make_agentic_summarizer_agent
from kompend.core.config._configs import LLMConfig


class TestAgenticHarnessCapabilities:
    def test_memory_is_off_by_default(self) -> None:
        """A notebook the prompt never mentions only invites hallucinated reads.

        ``read_memory`` answers a missing file with ``ModelRetry`` — once the
        tool-retry budget is spent the run dies and the node is left
        unsummarized.
        """
        capabilities = AgenticHarness(context_window=8192).build_capabilities()
        assert not any(cap.__class__.__name__ == "Memory" for cap in capabilities)

    def test_memory_can_be_enabled_explicitly(self) -> None:
        capabilities = AgenticHarness(context_window=8192, memory=True).build_capabilities()
        assert any(cap.__class__.__name__ == "Memory" for cap in capabilities)

    def test_compaction_and_clamp_stay_on(self) -> None:
        capabilities = AgenticHarness(context_window=8192).build_capabilities()
        names = [cap.__class__.__name__ for cap in capabilities]
        assert "SlidingWindowCompaction" in names
        assert "ClampOversizedMessages" in names


class TestSummarizerAgentHasNoMemory:
    def test_summarizer_agent_has_no_memory_capability(self) -> None:
        agent = make_agentic_summarizer_agent(
            "ollama/noop",
            context_window=8192,
        )
        assert not any(cap.__class__.__name__ == "Memory" for cap in agent.extra_capabilities)


class TestAgentMemoryConfig:
    """``compile.llm.agent_memory`` turns the notebook on for multi-output stages.

    The notebook is an ``InMemoryStore`` owned by one agent instance, so it only
    helps a stage where a single instance handles many outputs (summarisation,
    verification, remediation, distillation).
    """

    def test_config_off_keeps_the_notebook_off(self) -> None:
        agent = make_agentic_summarizer_agent("ollama/noop", context_window=8192, llm=LLMConfig())
        assert not any(cap.__class__.__name__ == "Memory" for cap in agent.extra_capabilities)

    def test_config_on_offers_the_notebook(self) -> None:
        agent = make_agentic_summarizer_agent(
            "ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=True),
        )
        assert any(cap.__class__.__name__ == "Memory" for cap in agent.extra_capabilities)
