"""Claim verification — one LLM session over every claim in the wiki.

Opt-in and costly. Claims are split per sentence, their citations' sources
are loaded, and :class:`~kompend.agents.verification.VerificationAgent`
works through them all in one multi-turn session (get_claim() →
search_sources()/read_page() → mark_*()); claims left without a verdict
come out as ``unverified``. There is no substring fallback — an LLM-less
run records the gap instead of guessing it.

The report row goes through SQLAlchemy Core against the shared
``verification`` table, so the table and its columns are named once.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from kompend.agents.types import VerificationClaim, VerificationDeps, VerificationPage, VerificationSource, VerificationVerdict
from kompend.agents.verification import VerificationAgent
from kompend.core.config._core import DEFAULTS
from kompend.core.models import Citation, CompiledWiki, extract_citations
from kompend.core.store import core_connection, open_store, vault_db_path
from kompend.core.store.tables import verification
from kompend.core.utils.time import iso_local
from kompend.logging import get_logger

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")

logger = get_logger(__name__)


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
    agent: VerificationAgent,
) -> VerificationResult:
    sources_dir = base_dir / DEFAULTS["paths"]["sources_dir"]
    if not sources_dir.exists():
        sources_dir = base_dir

    result = VerificationResult()
    result.claims.extend(_verify_wiki_session(wiki, sources_dir, agent))

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
    with open_store(vault_db_path(base_dir)) as conn, core_connection(conn) as core:
        core.execute(
            sqlite_insert(verification).values(
                written_at=iso_local(),
                total=result.total,
                verified=result.verified,
                unsupported=result.unsupported,
                average_confidence=result.average_confidence,
                claims=claims_json,
            )
        )


def _verify_wiki_session(wiki: CompiledWiki, sources_dir: Path, agent: VerificationAgent) -> list[ClaimVerification]:
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
                    sources.append(VerificationSource(slug=citation.source, text=excerpt))
            pages.append(VerificationPage(slug=node.id, content=node.content or ""))
            claim_id += 1

    deps = VerificationDeps(claims=claims, sources=sources, pages=pages)
    agent.run(deps)
    verdicts: dict[int, VerificationVerdict] = {v.claim_id: v for v in deps.results}

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
