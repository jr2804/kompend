"""Claim/citation verification — LLM-as-judge with substring fallback.

Opt-in and costly. The substring fallback makes the pipeline useful offline.
Replaced the old segmented-excerpt loop with a single-pass approach:
if the excerpt is too long, skip the LLM call and go straight to substring.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import VerificationClaim, VerificationDeps, VerificationPage, VerificationSource
from kompend.agents.verification import AgenticVerificationAgent, VerificationAgent
from kompend.core.config._core import DEFAULTS
from kompend.core.models import Citation, CompiledWiki, WikiNode, extract_citations
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import VERIFICATION_TABLE
from kompend.core.templates import get_template_loader
from kompend.core.utils.time import iso_local
from kompend.logging import get_logger

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")

#: Cited source excerpt passed to the verification agent per citation. The
#: agent judges a claim against this excerpt, so it is a context budget rather
#: than a display crop — raise it if verdicts start missing support that the
#: source does contain.
_EXCERPT_CHARS: int = 5000

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

    _persist_verification(base_dir, result)
    return result


def _persist_verification(base_dir: Path, result: VerificationResult) -> None:
    """Append the verification summary + per-claim records to ``verification``.

    Each run appends a row; readers take the latest (``ORDER BY id DESC``).
    """
    claims_json = json.dumps([asdict(c) for c in result.claims], default=str)
    with open_store(vault_db_path(base_dir)) as conn:
        conn.execute(
            f"INSERT INTO {VERIFICATION_TABLE} (written_at, total, verified, unsupported, average_confidence, claims) VALUES (?, ?, ?, ?, ?, ?)",
            (
                iso_local(),
                result.total,
                result.verified,
                result.unsupported,
                result.average_confidence,
                claims_json,
            ),
        )


def _verify_wiki_agentic(wiki: CompiledWiki, sources_dir: Path, agent: AgenticVerificationAgent) -> list[ClaimVerification]:
    """Run a single agentic verification session over all claims.

    Builds claim/source/page lists from the wiki, runs the agent once,
    then maps the collected verdicts back to ``ClaimVerification`` rows.
    Claims with no citation or unfound sources are still emitted as
    ``unsupported`` so the result table stays complete.
    """
    claims: list[VerificationClaim] = []
    sources: list[VerificationSource] = []
    pages: list[VerificationPage] = []
    claim_to_node: dict[int, tuple[str, Citation | None]] = {}

    claim_id = 0
    for node in wiki.nodes:
        for claim_text in _split_claims(node.content or ""):
            citations = extract_citations(node.content or "")
            citation = citations[0] if citations else None
            claims.append(VerificationClaim(id=claim_id, text=claim_text))
            claim_to_node[claim_id] = (node.id, citation)
            if citation:
                excerpt = _load_excerpt(citation.source, sources_dir)
                if excerpt:
                    sources.append(VerificationSource(slug=citation.source, text=excerpt[:_EXCERPT_CHARS]))
            pages.append(VerificationPage(slug=node.id, content=node.content or ""))
            claim_id += 1

    deps = VerificationDeps(claims=claims, sources=sources, pages=pages)
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
