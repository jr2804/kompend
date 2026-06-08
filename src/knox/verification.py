"""Q21 grounding verification — LLM-as-judge on claim/citation pairs.

The verifier compares each claim with the cited source excerpt and
returns a confidence score in ``[0.0, 1.0]``. This is opt-in and
costly — never enabled by default.

When the LLM is unavailable the verifier falls back to a deterministic
substring heuristic so the pipeline remains useful offline.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm

from knox.core.models import CompiledWiki, WikiNode
from knox.provenance import Citation, extract_citations

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")

_VERIFY_SYSTEM = (
    "You verify whether a claim is supported by its cited source. "
    'Return only valid JSON: {"verdict": "supported|partial|unsupported", '
    '"confidence": 0.0-1.0, "reason": "..."}'
)
_VERIFY_TEMPLATE = "Claim: {claim}\n\nCited source ({source}, lines {start}-{end}): {excerpt}\n\nReturn a JSON object with verdict, confidence, and reason."


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


def _try_litellm_json(prompt: str, system: str | None = None, *, model: str = "ollama/llama3.2") -> dict[str, Any] | None:
    """Call litellm completion and return parsed JSON, or None on failure."""
    try:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = litellm.completion(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.exception("litellm call failed in verification")
        return None


def _load_excerpt(source: str, sources_dir: Path) -> str:
    """Best-effort load of the cited source excerpt from the sources dir."""
    if not sources_dir.exists():
        return ""
    target = sources_dir / source
    if not target.exists():
        return ""
    try:
        return target.read_text(encoding="utf-8", errors="ignore")[:50_000]
    except OSError:
        return ""


def _fallback_verify(claim: str, excerpt: str) -> ClaimVerification:
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
    # Pick the most distinctive 5-gram from the claim and look it up.
    tokens = claim.split()
    if len(tokens) < 5:
        return ClaimVerification(
            node_id="",
            claim=claim,
            citation=None,
            verdict="unsupported",
            confidence=0.0,
            reason="claim too short",
        )
    probe = " ".join(tokens[:5]).lower()
    if probe in excerpt.lower():
        return ClaimVerification(
            node_id="",
            claim=claim,
            citation=None,
            verdict="supported",
            confidence=0.6,
            reason="substring match (fallback)",
        )
    return ClaimVerification(
        node_id="",
        claim=claim,
        citation=None,
        verdict="unsupported",
        confidence=0.2,
        reason="no substring match (fallback)",
    )


def verify_node(
    node: WikiNode,
    sources_dir: Path,
    *,
    min_confidence: float = 0.4,
    model: str = "ollama/llama3.2",
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

        prompt = _VERIFY_TEMPLATE.format(
            claim=claim,
            source=citation.source,
            start=citation.start,
            end=citation.end,
            excerpt=excerpt[:3000],
        )
        data = _try_litellm_json(prompt, system=_VERIFY_SYSTEM, model=model)
        if isinstance(data, dict):
            confidence = float(data.get("confidence", 0.0))
            verdict = str(data.get("verdict", "unsupported"))
            out.append(
                ClaimVerification(
                    node_id=node.id,
                    claim=claim,
                    citation=citation,
                    verdict=verdict,
                    confidence=confidence,
                    reason=str(data.get("reason", "")),
                )
            )
        else:
            fallback = _fallback_verify(claim, excerpt)
            fallback.node_id = node.id
            fallback.citation = citation
            out.append(fallback)
    return out


def verify_wiki(wiki: CompiledWiki, output_dir: Path, *, model: str = "ollama/llama3.2") -> VerificationResult:
    """Verify every node in a compiled wiki.

    The ``output_dir`` is expected to be a sibling of the ``sources/``
    directory; the function uses the conventional ``sources/`` lookup
    to resolve cited filenames.
    """
    sources_dir = output_dir.parent / "sources"
    if not sources_dir.exists():
        sources_dir = output_dir

    result = VerificationResult()
    for node in wiki.nodes:
        claims = verify_node(node, sources_dir, model=model)
        result.claims.extend(claims)

    result.total = len(result.claims)
    result.verified = sum(1 for c in result.claims if c.verdict == "supported")
    result.unsupported = sum(1 for c in result.claims if c.is_unsupported())
    if result.claims:
        result.average_confidence = sum(c.confidence for c in result.claims) / len(result.claims)

    # Persist report next to the wiki output.
    report_path = output_dir / ".knox" / "verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return result
