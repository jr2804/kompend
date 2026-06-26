"""Q21 grounding verification — LLM-as-judge on claim/citation pairs.

The verifier compares each claim with the cited source excerpt and
returns a confidence score in ``[0.0, 1.0]``. This is opt-in and
costly — never enabled by default.

When the LLM is unavailable the verifier falls back to a deterministic
substring heuristic so the pipeline remains useful offline.

ponytail: opt-in, costly, never enabled by default. The substring fallback
  IS the practical path. Add LLM path only when a user profile requests it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Verification result for a single claim."""

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
    """Aggregate verification outcome for an entire wiki."""

    total: int = 0
    verified: int = 0
    unsupported: int = 0
    average_confidence: float = 0.0
    claims: list[ClaimVerification] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "verified": self.verified,
            "unsupported": self.unsupported,
            "average_confidence": self.average_confidence,
            "claims": [
                {
                    "node_id": c.node_id,
                    "claim": c.claim,
                    "citation": c.citation.to_marker() if c.citation else "",
                    "verdict": c.verdict,
                    "confidence": c.confidence,
                    "reason": c.reason,
                }
                for c in self.claims
            ],
        }


def _split_claims(text: str) -> list[str]:
    """Split text into claim-sized sentences (>= 12 chars)."""
    return [s.strip() for s in _SENTENCE_RE.findall(text) if len(s.strip()) >= 12]


def _verify_via_client(
    client: LLMClient,
    prompt: str,
    system: str | None = None,
) -> dict[str, Any] | None:
    """Call LLM via the client (with retry + failure tracking) and return parsed JSON."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    result = client.try_complete(
        messages=messages,
        role="verify",
        response_format=JSON_RESPONSE_FORMAT,
    )
    return result.json_data if result else None


def _load_excerpt(source: str, sources_dir: Path) -> str:
    """Load the full cited source text from the sources dir."""
    if not sources_dir.exists():
        return ""
    target = sources_dir / source
    if not target.exists():
        return ""
    try:
        return target.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _fallback_verify(claim: str, excerpt: str, min_ngram: int = 3) -> ClaimVerification:
    """Substring-based verification used when the LLM is unavailable."""
    if not excerpt:
        return ClaimVerification(
            node_id="",
            claim=claim,
            citation=None,
            verdict="unsupported",
            confidence=0.0,
            reason="source not found",
        )
    # Try n-grams from the full claim down to min_ngram words.
    # Longer n-grams are more distinctive — return on the first hit.
    tokens = claim.split()
    excerpt_lower = excerpt.lower()
    best_match_len = 0

    for n in range(len(tokens), min_ngram - 1, -1):
        for start in range(len(tokens) - n + 1):
            probe = " ".join(tokens[start : start + n]).lower()
            if probe in excerpt_lower and n > best_match_len:
                best_match_len = n
                break  # found a match at this n-gram size, no need to check more starts
        if best_match_len > 0:
            break  # found the longest matching n-gram

    if best_match_len > 0:
        # Confidence proportional to how much of the claim matched.
        coverage = best_match_len / len(tokens)
        return ClaimVerification(
            node_id="",
            claim=claim,
            citation=None,
            verdict="supported",
            confidence=round(0.3 + 0.4 * coverage, 2),
            reason=f"substring match ({best_match_len}/{len(tokens)} words, fallback)",
        )

    return ClaimVerification(
        node_id="",
        claim=claim,
        citation=None,
        verdict="unsupported",
        confidence=0.2,
        reason="no substring match (fallback)",
    )


def _verify_claim_with_excerpt(
    claim: str,
    excerpt: str,
    citation: Citation,
    node_id: str,
    *,
    client: LLMClient,
    model: str,
    verify_budget: int = 3000,
    min_ngram: int = 3,
) -> ClaimVerification:
    """Verify a single claim against an excerpt, iterating if needed."""
    verify_system = _loader.get_body("verification/verify_system.md.j2")

    # If the excerpt fits in one call, verify directly.
    if len(excerpt) <= verify_budget:
        prompt = _loader.render(
            "verification/verify_claim.md.j2",
            claim=claim,
            source=citation.source,
            start=citation.start,
            end=citation.end,
            excerpt=excerpt,
        )
        data = _verify_via_client(client, prompt, system=verify_system)
        if isinstance(data, dict):
            return ClaimVerification(
                node_id=node_id,
                claim=claim,
                citation=citation,
                verdict=str(data.get("verdict", "unsupported")),
                confidence=float(data.get("confidence", 0.0)),
                reason=str(data.get("reason", "")),
            )
        fallback = _fallback_verify(claim, excerpt, min_ngram=min_ngram)
        fallback.node_id = node_id
        fallback.citation = citation
        return fallback

    # Excerpt exceeds budget — iterate through segments.
    # A claim is supported if it passes in ANY segment.
    best: ClaimVerification | None = None
    for seg_start in range(0, len(excerpt), verify_budget):
        segment = excerpt[seg_start : seg_start + verify_budget]
        prompt = _loader.render(
            "verification/verify_claim.md.j2",
            claim=claim,
            source=citation.source,
            start=citation.start,
            end=citation.end,
            excerpt=segment,
        )
        data = _verify_via_client(client, prompt, system=verify_system)
        if isinstance(data, dict):
            confidence = float(data.get("confidence", 0.0))
            verdict = str(data.get("verdict", "unsupported"))
            if best is None or confidence > best.confidence:
                best = ClaimVerification(
                    node_id=node_id,
                    claim=claim,
                    citation=citation,
                    verdict=verdict,
                    confidence=confidence,
                    reason=str(data.get("reason", "")),
                )
            if verdict == "supported":
                return best  # early exit — claim confirmed

    if best is not None:
        return best

    # All segments failed LLM call — use substring fallback on full excerpt.
    fallback = _fallback_verify(claim, excerpt)
    fallback.node_id = node_id
    fallback.citation = citation
    return fallback


def verify_node(
    node: WikiNode,
    sources_dir: Path,
    *,
    client: LLMClient,
    model: str,
    min_confidence: float = 0.4,
    verify_budget: int = 3000,
    min_ngram: int = 3,
) -> list[ClaimVerification]:
    """Verify the claims in a single node."""
    claims = _split_claims(node.content)
    citations = extract_citations(node.content)
    out: list[ClaimVerification] = []

    for claim in claims:
        citation = citations[0] if citations else None
        excerpt = ""
        if citation is not None:
            excerpt = _load_excerpt(citation.source, sources_dir)

        if citation is None:
            # No provenance → unsourced claim, low confidence.
            out.append(
                ClaimVerification(
                    node_id=node.id,
                    claim=claim,
                    citation=None,
                    verdict="unsupported",
                    confidence=0.1,
                    reason="no citation",
                )
            )
            continue

        out.append(_verify_claim_with_excerpt(claim, excerpt, citation, node.id, client=client, model=model, verify_budget=verify_budget, min_ngram=min_ngram))
    return out


def verify_wiki(wiki: CompiledWiki, output_dir: Path, *, client: LLMClient, model: str, verify_budget: int = 3000, min_ngram: int = 3) -> VerificationResult:
    """Verify every node in a compiled wiki.

    The ``output_dir`` is expected to be a sibling of the ``sources/``
    directory; the function uses the conventional ``sources/`` lookup
    to resolve cited filenames.
    """
    sources_dir = output_dir.parent / DEFAULTS["paths"]["sources_dir"]
    if not sources_dir.exists():
        sources_dir = output_dir

    result = VerificationResult()
    for node in wiki.nodes:
        claims = verify_node(node, sources_dir, client=client, model=model, verify_budget=verify_budget, min_ngram=min_ngram)
        result.claims.extend(claims)

    result.total = len(result.claims)
    result.verified = sum(1 for c in result.claims if c.verdict == "supported")
    result.unsupported = sum(1 for c in result.claims if c.is_unsupported())
    if result.claims:
        result.average_confidence = sum(c.confidence for c in result.claims) / len(result.claims)

    # Persist report next to the wiki output.
    report_path = VaultPaths(output_dir.parent).verification_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return result
