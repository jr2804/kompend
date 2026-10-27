"""Tests for the shared agentic capability bundle."""

from __future__ import annotations

from typing import Any

from pydantic_ai_harness.memory import InMemoryStore

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.distill_agentic import make_agentic_distill_agent
from kompend.agents.remediate import make_agentic_remediation_agent
from kompend.agents.summarizer_agentic import make_agentic_summarizer_agent
from kompend.agents.vault_memory import DirectoryMemoryStore
from kompend.agents.verification import make_agentic_verification_agent
from kompend.core.config._configs import LLMConfig, LlmMemoryStore


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

    The notebook is owned by one agent instance, so it only helps a stage where
    a single instance handles many outputs (summarisation, verification,
    remediation, distillation).
    """

    def test_config_default_offers_the_notebook(self) -> None:
        agent = make_agentic_summarizer_agent("ollama/noop", context_window=8192, llm=LLMConfig())
        assert any(cap.__class__.__name__ == "Memory" for cap in agent.extra_capabilities)

    def test_config_off_keeps_the_notebook_off(self) -> None:
        agent = make_agentic_summarizer_agent(
            "ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=False),
        )
        assert not any(cap.__class__.__name__ == "Memory" for cap in agent.extra_capabilities)


class TestVaultMemoryStore:
    """``compile.llm.memory_store = "vault"`` backs the notebook with the
    vault-rooted :class:`DirectoryMemoryStore`.

    The store is injected by the caller (the pipeline factory passes the
    vault's agent-memory directory); the harness only chooses the default.
    """

    def test_session_store_is_the_default(self) -> None:
        capabilities = AgenticHarness(context_window=8192, memory=True).build_capabilities()
        memory = next(cap for cap in capabilities if cap.__class__.__name__ == "Memory")
        assert isinstance(memory.store, InMemoryStore)  # type: ignore[attr-defined]

    def test_memory_store_is_passed_through(self, tmp_path: Any) -> None:
        store = DirectoryMemoryStore(tmp_path)
        capabilities = AgenticHarness(context_window=8192, memory=True, memory_store=store).build_capabilities()
        memory = next(cap for cap in capabilities if cap.__class__.__name__ == "Memory")
        assert memory.store is store  # type: ignore[attr-defined]

    def test_vault_store_only_with_vault_config(self, tmp_path: Any) -> None:
        agent = make_agentic_summarizer_agent(
            "ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=True, memory_store=LlmMemoryStore.SESSION),
            memory_dir=tmp_path,
        )
        memory = next(cap for cap in agent.extra_capabilities if cap.__class__.__name__ == "Memory")
        assert isinstance(memory.store, InMemoryStore)  # type: ignore[attr-defined]

    def test_vault_config_builds_the_vault_store(self, tmp_path: Any) -> None:
        agent = make_agentic_summarizer_agent(
            "ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=True, memory_store=LlmMemoryStore.VAULT),
            memory_dir=tmp_path,
        )
        memory = next(cap for cap in agent.extra_capabilities if cap.__class__.__name__ == "Memory")
        assert isinstance(memory.store, DirectoryMemoryStore)  # type: ignore[attr-defined]
        assert memory.store.directory == tmp_path  # type: ignore[attr-defined]

    def test_store_requires_memory_enabled(self, tmp_path: Any) -> None:
        agent = make_agentic_summarizer_agent(
            "ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=False, memory_store=LlmMemoryStore.VAULT),
            memory_dir=tmp_path,
        )
        assert not any(cap.__class__.__name__ == "Memory" for cap in agent.extra_capabilities)


class TestNotebookStagesShareTheVault:
    """Every notebook stage keeps its notebook in the vault, scoped by agent name.

    The harness namespaces notebook paths by the agent name, so the stages
    share one directory and never one file.
    """

    def test_verification_uses_the_vault_store(self, tmp_path: Any) -> None:
        agent = make_agentic_verification_agent(
            model_string="ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=True, memory_store=LlmMemoryStore.VAULT),
            memory_dir=tmp_path,
        )
        memory = self._memory_capability(agent)
        assert isinstance(memory.store, DirectoryMemoryStore)  # type: ignore[attr-defined]
        assert memory.store.directory == tmp_path  # type: ignore[attr-defined]
        assert memory.agent_name == "verification"

    def test_remediation_uses_the_vault_store(self, tmp_path: Any) -> None:
        agent = make_agentic_remediation_agent(
            model_string="ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=True, memory_store=LlmMemoryStore.VAULT),
            memory_dir=tmp_path,
        )
        memory = self._memory_capability(agent)
        assert isinstance(memory.store, DirectoryMemoryStore)  # type: ignore[attr-defined]
        assert memory.agent_name == "remediation"

    def test_distillation_uses_the_vault_store(self, tmp_path: Any) -> None:
        agent = make_agentic_distill_agent(
            "ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=True, memory_store=LlmMemoryStore.VAULT),
            memory_dir=tmp_path,
        )
        memory = self._memory_capability(agent)
        assert isinstance(memory.store, DirectoryMemoryStore)  # type: ignore[attr-defined]
        assert memory.agent_name == "distill"

    def test_session_store_leaves_the_notebook_in_process(self, tmp_path: Any) -> None:
        agent = make_agentic_verification_agent(
            model_string="ollama/noop",
            context_window=8192,
            llm=LLMConfig(agent_memory=True, memory_store=LlmMemoryStore.SESSION),
            memory_dir=tmp_path,
        )
        memory = self._memory_capability(agent)
        assert isinstance(memory.store, InMemoryStore)  # type: ignore[attr-defined]

    @staticmethod
    def _memory_capability(agent: Any) -> Any:
        return next(cap for cap in agent.extra_capabilities if cap.__class__.__name__ == "Memory")
