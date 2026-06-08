"""WiCER: Compile-Evaluate-Refine loop for improved compilation quality.

Implements the iterative compilation loop described in
arXiv:2605.07068v1: compile (blind) → generate probes → evaluate →
diagnose → refine → repeat.

LLM calls are routed through the shared :class:`knox.llm.LLMClient`
which provides typed errors, failover and observability. When the
client returns ``None`` (all models in the chain failed), the
compiler falls back to deterministic heuristics so the pipeline never
crashes on a single bad provider.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.core.models import CompiledWiki, ResolvedChunk, WikiNode
from knox.core.template_loader import TemplateLoader
from knox.llm.client import LLMClient

logger = logging.getLogger(__name__)

_loader = TemplateLoader()


class WiCERCompiler(BaseCompiler):
    """Compile-Evaluate-Refine loop for improved compilation quality.

    Runs up to *max_iterations* rounds of blind compilation, probe
    generation, LLM-as-judge evaluation, failure diagnosis, and
    refinement.

    Parameters
    ----------
    base_compiler:
        The underlying compiler used for blind and refined compilation.
        Defaults to a fresh :class:`GraphCompiler`.
    max_iterations:
        Maximum number of refinement rounds (default 3).
    llm_client:
        Shared LLM client. When ``None``, a default client is built
        from the global default model. Inject a custom client (or one
        with a fallback chain) to control model selection per stage.
    """

    def __init__(
        self,
        base_compiler: BaseCompiler | None = None,
        max_iterations: int = 3,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._compiler = base_compiler or GraphCompiler()
        self.max_iterations = max_iterations
        self._client = llm_client  # None is OK — caller handles graceful degradation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Run the WiCER iterative-compilation loop.

        Returns a :class:`CompiledWiki` — never raises.
        """
        if not chunks:
            return CompiledWiki(
                nodes=[],
                edges=[],
                toc=[],
                metadata={"chunk_count": 0, "wicer_iterations": 0},
            )

        wiki = self._blind_compile(chunks)
        iteration_info: dict[str, Any] = {
            "wicer_iterations": 0,
            "wicer_refinements": 0,
            "wicer_probes_generated": 0,
            "wicer_probes_passed": 0,
        }

        for attempt in range(self.max_iterations):
            probes = self._generate_probes(wiki, chunks)
            if not probes:
                logger.info("WiCER: no probes generated, ending loop")
                break
            iteration_info["wicer_probes_generated"] += len(probes)

            eval_results = self._evaluate(wiki, probes)
            failures = [r for r in eval_results if not r.get("correct", False)]
            passed = len(eval_results) - len(failures)
            iteration_info["wicer_probes_passed"] += passed

            if not failures:
                logger.info("WiCER: all probes passed at iteration %d", attempt + 1)
                iteration_info["wicer_iterations"] = attempt + 1
                break

            pinned_facts = self._diagnose(failures)
            if not pinned_facts:
                logger.info("WiCER: no critical facts extracted, ending loop")
                iteration_info["wicer_iterations"] = attempt + 1
                break

            wiki = self._refine(wiki, pinned_facts)
            iteration_info["wicer_refinements"] += 1
            iteration_info["wicer_iterations"] = attempt + 1
        else:
            iteration_info["wicer_iterations"] = self.max_iterations

        wiki.metadata.update(iteration_info)
        return wiki

    # ------------------------------------------------------------------
    # WiCER steps  (each is overridable for testing)
    # ------------------------------------------------------------------

    def _blind_compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Step A: initial blind compilation."""
        return self._compiler.compile(chunks)

    def _generate_probes(self, wiki: CompiledWiki, chunks: list[ResolvedChunk]) -> list[dict]:
        """Step B: generate probe Q&A pairs from source chunks."""
        sample_texts = [rc.chunk.text for rc in chunks[:5] if rc.chunk.text]
        combined = "\n\n".join(sample_texts)
        if not combined:
            return []

        if self._client is None:
            logger.debug("WiCER has no LLM client — using fallback probes")
            return self._fallback_probes(combined)

        result = self._client.try_complete(
            messages=[
                {"role": "system", "content": "You are a test-probe generator. Return only valid JSON."},
                {"role": "user", "content": _loader.render("compilation/wicer_probe.md.j2", text=combined[:3000])},
            ],
            role="wicer.probe",
            response_format={"type": "json_object"},
        )

        probes = self._extract_probe_list(result)
        if probes is not None:
            return probes

        # Fallback: simple text-based probes
        return self._fallback_probes(combined)

    def _evaluate(self, wiki: CompiledWiki, probes: list[dict]) -> list[dict]:
        """Step B: evaluate wiki against probes using LLM-as-judge."""
        wiki_text = "\n\n".join(n.content for n in wiki.nodes if n.content)
        if not wiki_text:
            return [{"question": p.get("question", ""), "expected_answer": p.get("answer", ""), "found_answer": "", "correct": False} for p in probes]

        if self._client is None:
            logger.debug("WiCER has no LLM client — using fallback evaluation")
            return self._fallback_evaluate(wiki_text, probes)

        result = self._client.try_complete(
            messages=[
                {"role": "system", "content": "You are an impartial judge of factual correctness. Return only valid JSON."},
                {
                    "role": "user",
                    "content": _loader.render(
                        "compilation/wicer_evaluate.md.j2",
                        wiki_text=wiki_text[:4000],
                        probes_json=json.dumps(probes),
                    ),
                },
            ],
            role="wicer.judge",
            response_format={"type": "json_object"},
        )

        if result is not None and result.json_data is not None:
            results = result.json_data.get("results", [])
            if isinstance(results, list):
                return results

        return self._fallback_evaluate(wiki_text, probes)

    def _diagnose(self, failures: list[dict]) -> list[str]:
        """Step C: extract critical facts from failures using LLM."""
        if not failures:
            return []

        if self._client is None:
            logger.debug("WiCER has no LLM client — skipping diagnosis")
            return [f.get("expected_answer", "") for f in failures if f.get("expected_answer")]

        result = self._client.try_complete(
            messages=[
                {"role": "system", "content": "You are a diagnostic assistant. Return only valid JSON."},
                {"role": "user", "content": _loader.render("compilation/wicer_diagnose.md.j2", failures_json=json.dumps(failures))},
            ],
            role="wicer.refine",
            response_format={"type": "json_object"},
        )

        if result is not None and result.json_data is not None:
            facts = result.json_data.get("critical_facts", [])
            if isinstance(facts, list):
                return [str(f) for f in facts if f]

        return [f.get("expected_answer", "") for f in failures if f.get("expected_answer")]

    @staticmethod
    def _refine(wiki: CompiledWiki, pinned_facts: list[str]) -> CompiledWiki:
        """Step D: re-compile with pinned facts appended."""
        if not pinned_facts:
            return wiki

        fact_content = "\n".join(f"- {f}" for f in pinned_facts)
        fact_node_id = "node_refinement_pinned_facts"

        existing_nodes = list(wiki.nodes)
        refinement_idx = None
        for i, n in enumerate(existing_nodes):
            if n.id == fact_node_id:
                refinement_idx = i
                break

        refinement_node = WikiNode(
            id=fact_node_id,
            title="Pinned Refinement Facts",
            content=fact_content,
            node_type="reference",
            metadata={"pinned_facts": True},
        )

        if refinement_idx is not None:
            existing_nodes[refinement_idx] = refinement_node
        else:
            existing_nodes.append(refinement_node)

        return CompiledWiki(
            nodes=existing_nodes,
            edges=list(wiki.edges),
            toc=list(wiki.toc),
            metadata={**wiki.metadata, "refined_with_pinned_facts": len(pinned_facts)},
        )

    @staticmethod
    def _extract_probe_list(result: Any) -> list[dict] | None:
        """Extract a list of probes from various JSON shapes the LLM may return.

        Accepts ``[{...}, {...}]`` or ``{"probes": [...]}`` /
        ``{"results": [...]}`` / ``{"items": [...]}``. Falls back to
        the single list-valued key when the object has only one.
        """
        if result is None:
            return None
        # Try direct list shape first (older content path).
        if result.json_data is None:
            try:
                parsed = json.loads(result.content)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, list) else None

        data = result.json_data
        for key in ("probes", "results", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        list_values = [v for v in data.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        return None

    @staticmethod
    def _fallback_probes(text: str) -> list[dict]:
        """Generate simple fallback probes from text sentences."""
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
        probes: list[dict] = []
        for s in sentences[:5]:
            probes.append(
                {
                    "question": f"What does the source say about: {s[:60]}...?",
                    "answer": s[:200],
                }
            )
        return probes

    @staticmethod
    def _fallback_evaluate(wiki_text: str, probes: list[dict]) -> list[dict]:
        """Fallback evaluation using simple substring matching."""
        results: list[dict] = []
        wiki_lower = wiki_text.lower()
        for p in probes:
            answer = p.get("answer", "")
            found = answer.lower()[:50] in wiki_lower if answer else False
            results.append(
                {
                    "question": p.get("question", ""),
                    "expected_answer": answer,
                    "found_answer": answer if found else "",
                    "correct": found,
                }
            )
        return results
