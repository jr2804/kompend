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

from knox.agents.deps import PipelineAgentDeps
from knox.agents.verification import VerificationAgent
from knox.core.config._core import DEFAULTS
from knox.core.models import Citation, CompiledWiki, WikiNode, extract_citations
from knox.core.paths import VaultPaths
from knox.core.templates import get_template_loader
from knox.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


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
    output_dir: Path,
    *,
    agent: VerificationAgent,
    verify_budget: int = 3000,
    min_ngram: int = 3,
) -> VerificationResult:
    sources_dir = output_dir.parent / DEFAULTS["paths"]["sources_dir"]
    if not sources_dir.exists():
        sources_dir = output_dir

    result = VerificationResult()
    for node in wiki.nodes:
        result.claims.extend(verify_node(node, sources_dir, agent=agent, verify_budget=verify_budget, min_ngram=min_ngram))

    result.total = len(result.claims)
    result.verified = sum(1 for c in result.claims if c.verdict == "supported")
    result.unsupported = sum(1 for c in result.claims if c.is_unsupported())
    if result.claims:
        result.average_confidence = sum(c.confidence for c in result.claims) / len(result.claims)

    report_path = VaultPaths(output_dir.parent).verification_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    return result


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
        deps = PipelineAgentDeps(text=prompt, source_id=node.id, timeout=agent._timeout)
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
