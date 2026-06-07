"""WiCER: Compile-Evaluate-Refine loop for improved compilation quality.

Implements the iterative compilation loop described in
arXiv:2605.07068v1: compile (blind) → generate probes → evaluate →
diagnose → refine → repeat.

When the LLM call fails, the compiler falls back to blind one-shot
compilation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from knox.compilation.base import BaseCompiler
from knox.compilation.graph import GraphCompiler
from knox.core.models import CompiledWiki, ResolvedChunk, WikiNode

logger = logging.getLogger(__name__)

_DEFAULT_PROBE_TEMPLATE = (
    "Based on the following text, generate 2 Q&A pairs as a JSON list "
    'where each item has "question" and "answer" keys. '
    "Focus on factual content that can be verified. "
    "Text: {text}"
)

_EVALUATION_TEMPLATE = (
    "You are evaluating a compiled wiki for factual correctness and completeness. "
    "Given the following probe questions and the wiki content, determine whether "
    "each question is correctly answered. "
    'Return a JSON object with key "results" which is a list of dicts, each with '
    '"question", "expected_answer", "found_answer" (the text from the wiki that '
    "answers it, or empty string if missing), and \"correct\" (boolean).\\n\\n"
    "Wiki content:\\n{wiki_text}\\n\\n"
    "Probes:\\n{probes_json}"
)

_DIAGNOSE_TEMPLATE = (
    "The following probe questions were answered incorrectly by the compiled wiki. "
    "Extract the critical facts that are missing or wrong. "
    'Return a JSON object with key "critical_facts" which is a list of strings, '
    "each describing one fact that must be included for a correct answer.\\n\\n"
    "Failures:\\n{failures_json}"
)


def _try_litellm(prompt: str, system: str | None = None) -> str | None:
    """Call litellm completion, returning content string or None on failure."""
    try:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = litellm.completion(
            model="ollama/llama3.2",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return content if content else None
    except Exception:
        logger.exception("litellm call failed in WiCER loop")
        return None


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
    """

    def __init__(
        self,
        base_compiler: BaseCompiler | None = None,
        max_iterations: int = 3,
    ) -> None:
        self._compiler = base_compiler or GraphCompiler()
        self.max_iterations = max_iterations

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
        """Step B: generate probe Q&A pairs from source chunks.

        Uses litellm to generate probes; falls back to extracting
        simple truth statements from chunk text when LLM is unavailable.
        """
        # Pick a representative sample of chunk texts
        sample_texts = [rc.chunk.text for rc in chunks[:5] if rc.chunk.text]
        combined = "\n\n".join(sample_texts)
        if not combined:
            return []

        # Try LLM-based probe generation
        system = "You are a test-probe generator. Return only valid JSON."
        prompt = _DEFAULT_PROBE_TEMPLATE.format(text=combined[:3000])
        content = _try_litellm(prompt, system=system)

        if content:
            try:
                probes: list[dict] = json.loads(content)
                if isinstance(probes, list):
                    return probes
            except (json.JSONDecodeError, TypeError):
                logger.warning("WiCER: failed to parse probes from LLM output")

        # Fallback: simple text-based probes
        return self._fallback_probes(combined)

    @staticmethod
    def _fallback_probes(text: str) -> list[dict]:
        """Generate simple fallback probes from text sentences."""
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
        probes: list[dict] = []
        for s in sentences[:5]:
            probes.append({
                "question": f"What does the source say about: {s[:60]}...?",
                "answer": s[:200],
            })
        return probes

    def _evaluate(self, wiki: CompiledWiki, probes: list[dict]) -> list[dict]:
        """Step B: evaluate wiki against probes using LLM-as-judge.

        Falls back to simple substring matching when LLM is unavailable.
        """
        wiki_text = "\n\n".join(n.content for n in wiki.nodes if n.content)
        if not wiki_text:
            # Everything is a failure
            return [
                {"question": p.get("question", ""), "expected_answer": p.get("answer", ""),
                 "found_answer": "", "correct": False}
                for p in probes
            ]

        system = "You are an impartial judge of factual correctness. Return only valid JSON."
        prompt = _EVALUATION_TEMPLATE.format(
            wiki_text=wiki_text[:4000],
            probes_json=json.dumps(probes),
        )
        content = _try_litellm(prompt, system=system)

        if content:
            try:
                data: Any = json.loads(content)
                if isinstance(data, dict):
                    results = data.get("results", [])
                elif isinstance(data, list):
                    # Treat as direct list of results
                    results = data
                else:
                    results = []
                if isinstance(results, list):
                    return results
            except (json.JSONDecodeError, TypeError):
                logger.warning("WiCER: failed to parse evaluation results from LLM output")

        # Fallback: substring matching
        return self._fallback_evaluate(wiki_text, probes)

    @staticmethod
    def _fallback_evaluate(wiki_text: str, probes: list[dict]) -> list[dict]:
        """Fallback evaluation using simple substring matching."""
        results: list[dict] = []
        wiki_lower = wiki_text.lower()
        for p in probes:
            answer = p.get("answer", "")
            found = answer.lower()[:50] in wiki_lower if answer else False
            results.append({
                "question": p.get("question", ""),
                "expected_answer": answer,
                "found_answer": answer if found else "",
                "correct": found,
            })
        return results

    def _diagnose(self, failures: list[dict]) -> list[str]:
        """Step C: extract critical facts from failures using LLM.

        Falls back to extracting expected answers as plain facts.
        """
        if not failures:
            return []

        system = "You are a diagnostic assistant. Return only valid JSON."
        prompt = _DIAGNOSE_TEMPLATE.format(failures_json=json.dumps(failures))
        content = _try_litellm(prompt, system=system)

        if content:
            try:
                data: dict[str, Any] = json.loads(content)
                facts = data.get("critical_facts", [])
                if isinstance(facts, list):
                    return [str(f) for f in facts if f]
            except (json.JSONDecodeError, TypeError):
                logger.warning("WiCER: failed to parse diagnosis from LLM output")

        # Fallback: use expected answers as facts
        return [
            f.get("expected_answer", "") for f in failures
            if f.get("expected_answer")
        ]

    def _refine(self, wiki: CompiledWiki, pinned_facts: list[str]) -> CompiledWiki:
        """Step D: re-compile with pinned facts appended.

        Creates a new CompiledWiki that includes the pinned facts as
        an additional reference node, preserving all original nodes and edges.
        """
        if not pinned_facts:
            return wiki

        fact_content = "\n".join(f"- {f}" for f in pinned_facts)
        fact_node_id = "node_refinement_pinned_facts"

        # Check if a refinement node already exists — if so, update it
        existing_nodes = list(wiki.nodes)
        refinement_idx = None
        for i, n in enumerate(existing_nodes):
            if n.id == fact_node_id:
                refinement_idx = i
                break

        if refinement_idx is not None:
            existing_nodes[refinement_idx] = WikiNode(
                id=fact_node_id,
                title="Pinned Refinement Facts",
                content=fact_content,
                node_type="reference",
                metadata={"pinned_facts": True},
            )
        else:
            existing_nodes.append(
                WikiNode(
                    id=fact_node_id,
                    title="Pinned Refinement Facts",
                    content=fact_content,
                    node_type="reference",
                    metadata={"pinned_facts": True},
                )
            )

        return CompiledWiki(
            nodes=existing_nodes,
            edges=list(wiki.edges),
            toc=list(wiki.toc),
            metadata={**wiki.metadata, "refined_with_pinned_facts": len(pinned_facts)},
        )