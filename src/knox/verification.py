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

from knox.core.config._core import DEFAULTS
from knox.core.models import Citation, CompiledWiki, WikiNode, extract_citations
from knox.core.paths import VaultPaths
from knox.core.template_loader import get_template_loader
from knox.llm.client import JSON_RESPONSE_FORMAT, LLMClient
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


def _split_claims(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if len(s.strip()) >= 12]


def _load_excerpt(source: str, sources_dir: Path) -> str:
    target = sources_dir / source
    if target.exists():
        try:
            return target.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    return ""


def _fallback_verify(claim: str, excerpt: str, min_ngram: int = 3) -> ClaimVerification:
    """Substring n-gram matching. Used when LLM is unavailable or excerpt is too long."""
    if not excerpt:
        return ClaimVerification("", claim, None, "unsupported", 0.0, "source not found")
    tokens = claim.split()
    excerpt_lower = excerpt.lower()
    best = 0
    for n in range(len(tokens), min_ngram - 1, -1):
        for start in range(len(tokens) - n + 1):
            if " ".join(tokens[start : start + n]).lower() in excerpt_lower and n > best:
                best = n
                break
        if best:
            break
    if best:
        coverage = best / len(tokens)
        return ClaimVerification("", claim, None, "supported", round(0.3 + 0.4 * coverage, 2), f"substring match ({best}/{len(tokens)} words)")
    return ClaimVerification("", claim, None, "unsupported", 0.2, "no substring match")


def verify_node(
    node: WikiNode,
    sources_dir: Path,
    *,
    client: LLMClient,
    model: str,
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

        # Try LLM, fall through to substring if excerpt is too long or LLM fails.
        if len(excerpt) <= verify_budget:
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
            result = client.try_complete(messages=messages, role="verify", response_format=JSON_RESPONSE_FORMAT)
            if result and result.json_data:
                data = result.json_data
                results.append(
                    ClaimVerification(
                        node.id, claim, citation, str(data.get("verdict", "unsupported")), float(data.get("confidence", 0.0)), str(data.get("reason", ""))
                    )
                )
                continue

        # Fallback: substring matching
        fv = _fallback_verify(claim, excerpt, min_ngram=min_ngram)
        fv.node_id = node.id
        fv.citation = citation
        results.append(fv)
    return results


def verify_wiki(
    wiki: CompiledWiki,
    output_dir: Path,
    *,
    client: LLMClient,
    model: str,
    verify_budget: int = 3000,
    min_ngram: int = 3,
) -> VerificationResult:
    sources_dir = output_dir.parent / DEFAULTS["paths"]["sources_dir"]
    if not sources_dir.exists():
        sources_dir = output_dir

    result = VerificationResult()
    for node in wiki.nodes:
        result.claims.extend(verify_node(node, sources_dir, client=client, model=model, verify_budget=verify_budget, min_ngram=min_ngram))

    result.total = len(result.claims)
    result.verified = sum(1 for c in result.claims if c.verdict == "supported")
    result.unsupported = sum(1 for c in result.claims if c.is_unsupported())
    if result.claims:
        result.average_confidence = sum(c.confidence for c in result.claims) / len(result.claims)

    report_path = VaultPaths(output_dir.parent).verification_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    return result
