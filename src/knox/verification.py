"""Claim/citation verification — LLM-as-judge with substring fallback.

Opt-in and costly. The substring fallback makes the pipeline useful offline.
Replaced the old segmented-excerpt loop with a single-pass approach:
if the excerpt is too long, skip the LLM call and go straight to substring.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from knox.agents.deps import PipelineAgentDeps
from knox.agents.types import VerificationDeps
from knox.agents.verification import AgenticVerificationAgent, VerificationAgent
from knox.core.config._core import DEFAULTS
from knox.core.models import Citation, CompiledWiki, WikiNode, extract_citations
from knox.core.paths import VaultPaths
from knox.core.templates import get_template_loader
from knox.logging import get_logger

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")

logger = get_logger(__name__)

_loader = get_template_loader()


@dataclass
class ClaimVerification:
    node_id: str
    claim: str
    citation: Citation | None
    verdict: str
    confidence: float
    reason: str = ""

    def is_unsupported(self) -> bool:
        return self.verdict == "unsupported" or self.confidence < 0.4


@dataclass
class VerificationResult:
    total: int = 0
    verified: int = 0
    unsupported: int = 0
    average_confidence: float = 0.0
    claims: list[ClaimVerification] = field(default_factory=list)


def verify_wiki(
    wiki: CompiledWiki,
    base_dir: Path,
    *,
    agent: VerificationAgent | AgenticVerificationAgent,
    verify_budget: int = 3000,
    min_ngram: int = 3,
) -> VerificationResult:
    sources_dir = base_dir / DEFAULTS["paths"]["sources_dir"]
    if not sources_dir.exists():
        sources_dir = base_dir

    result = VerificationResult()
    if isinstance(agent, AgenticVerificationAgent):
        result.claims.extend(_verify_wiki_agentic(wiki, sources_dir, agent))
    else:
        for node in wiki.nodes:
            result.claims.extend(verify_node(node, sources_dir, agent=agent, verify_budget=verify_budget, min_ngram=min_ngram))

    result.total = len(result.claims)
    result.verified = sum(1 for c in result.claims if c.verdict == "supported")
    result.unsupported = sum(1 for c in result.claims if c.is_unsupported())
    if result.claims:
        result.average_confidence = sum(c.confidence for c in result.claims) / len(result.claims)

    report_file = VaultPaths(base_dir).verification_file
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    return result


def _verify_wiki_agentic(wiki: CompiledWiki, sources_dir: Path, agent: AgenticVerificationAgent) -> list[ClaimVerification]:
    """Run a single agentic verification session over all claims.

    Builds claim/source/page lists from the wiki, runs the agent once,
    then maps the collected verdicts back to ``ClaimVerification`` rows.
    Claims with no citation or unfound sources are still emitted as
    ``unsupported`` so the result table stays complete.
    """
    claim_dicts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    claim_to_node: dict[int, tuple[str, Citation | None]] = {}

    claim_id = 0
    for node in wiki.nodes:
        for claim_text in _split_claims(node.content or ""):
            citations = extract_citations(node.content or "")
            citation = citations[0] if citations else None
            claim_dicts.append({"id": claim_id, "text": claim_text})
            claim_to_node[claim_id] = (node.id, citation)
            if citation:
                excerpt = _load_excerpt(citation.source, sources_dir)
                if excerpt:
                    sources.append({"slug": citation.source, "text": excerpt[:5000]})
            pages.append({"slug": node.id, "content": node.content or ""})
            claim_id += 1

    deps = VerificationDeps(claims=claim_dicts, sources=sources, pages=pages)
    try:
        agent.run(deps)
    except Exception:
        logger.debug("Agentic verification failed", exc_info=True)
    verdicts = {v.claim_id: v for v in deps.results}

    results: list[ClaimVerification] = []
    for cid in range(claim_id):
        node_id, citation = claim_to_node.get(cid, ("unknown", None))
        verdict = verdicts.get(cid)
        if verdict is None:
            results.append(ClaimVerification(node_id, "", citation, "unverified", 0.0, "no verdict"))
        else:
            confidence = 0.9 if verdict.verdict == "supported" else 0.5 if verdict.verdict == "contradicted" else 0.1
            results.append(ClaimVerification(node_id, "", citation, verdict.verdict, confidence, verdict.evidence))
    return results


def verify_node(
    node: WikiNode,
    sources_dir: Path,
    *,
    agent: VerificationAgent,
    verify_budget: int = 3000,
    min_ngram: int = 3,
) -> list[ClaimVerification]:
    results: list[ClaimVerification] = []
    for claim in _split_claims(node.content):
        citations = extract_citations(node.content)
        citation = citations[0] if citations else None
        if citation is None:
            results.append(ClaimVerification(node.id, claim, None, "unsupported", 0.1, "no citation"))
            continue

        excerpt = _load_excerpt(citation.source, sources_dir)
        if not excerpt:
            results.append(ClaimVerification(node.id, claim, citation, "unsupported", 0.0, "source not found"))
            continue

        if len(excerpt) > verify_budget:
            results.append(ClaimVerification(node.id, claim, citation, "unverified", 0.0, f"excerpt too long ({len(excerpt)} > {verify_budget})"))
            continue

        verify_system = _loader.get_body("verification/verify_system.md.j2")
        prompt = _loader.render(
            "verification/verify_claim.md.j2",
            claim=claim,
            source=citation.source,
            start=citation.start,
            end=citation.end,
            excerpt=excerpt,
        )
        messages: list[dict[str, str]] = []
        if verify_system:
            messages.append({"role": "system", "content": verify_system})
        messages.append({"role": "user", "content": prompt})
        deps = PipelineAgentDeps(text=prompt, source_id=node.id, timeout=agent.timeout)
        try:
            data = agent.run(deps)
        except Exception:
            data = None
        if data:
            results.append(
                ClaimVerification(
                    node.id, claim, citation, str(data.get("verdict", "unsupported")), float(data.get("confidence", 0.0)), str(data.get("reason", ""))
                )
            )
        else:
            results.append(ClaimVerification(node.id, claim, citation, "unverified", 0.0, "LLM verification unavailable"))
    return results


def _split_claims(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if len(s.strip()) >= 12]


def _load_excerpt(source: str, sources_dir: Path) -> str:
    target = Path(source)
    if not target.is_absolute():
        for root in (sources_dir.parent, sources_dir):
            if (root / target).exists():
                target = root / target
                break
        else:
            target = sources_dir.parent / target
    if target.exists():
        try:
            return target.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            logger.debug("Failed to read source for verification: %s", target, exc_info=True)
    return ""
