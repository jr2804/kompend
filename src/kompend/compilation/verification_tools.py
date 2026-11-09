"""PydanticAI tools for the agentic claim-verification agent.

Provides ``build_verification_tools()`` returning the toolset an
agentic verification agent uses to verify claims:

- ``get_claim(claim_id: int)`` — read a claim text by ID.
- ``search_sources(query: str, k: int = 5)`` — substring-search the
  source excerpts loaded into this run and return the first ``k`` hits.
  It is not an embedding search: the excerpts come from the sources the
  wiki cites (``kompend.verification._load_excerpt``) and are matched
  literally, case-insensitively.
- ``read_page(slug: str)`` — read a full wiki page.
- ``mark_verified(claim_id: int, evidence: str, sources: list[str])``
- ``mark_contradicted(claim_id: int, evidence: str, sources: list[str])``
- ``mark_unverifiable(claim_id: int, reason: str)``

All tools access their data through ``RunContext[VerificationDeps]``.
The verdicts are collected on ``deps.results`` and applied by the
caller after the run finishes.
"""

from __future__ import annotations

from pydantic_ai import RunContext, Tool

from kompend.agents.types import VerificationDeps, VerificationVerdict

#: Snippet length per source hit returned to the verification agent.
_HIT_TEXT_CHARS: int = 200


def build_verification_tools() -> list[Tool[VerificationDeps]]:
    """Return the PydanticAI toolset for the agentic verification agent."""
    return [
        Tool(get_claim),
        Tool(search_sources),
        Tool(read_page),
        Tool(mark_verified),
        Tool(mark_contradicted),
        Tool(mark_unverifiable),
    ]


async def get_claim(ctx: RunContext[VerificationDeps], claim_id: int) -> str:
    """Return the claim text for *claim_id*."""
    for claim in ctx.deps.claims:
        if claim["id"] == claim_id:
            return claim["text"]
    return f"Unknown claim: {claim_id}"


async def search_sources(ctx: RunContext[VerificationDeps], query: str, k: int = 5) -> str:
    """Return up to *k* source excerpts that contain *query* verbatim.

    Args:
        ctx: Runtime context providing the cited source excerpts via deps.
        query: Text to look for — matched case-insensitively as a substring.
        k: Maximum number of excerpts to return (default 5).

    The match is literal, not semantic: a paraphrase or a synonym finds
    nothing, so search for a distinctive term from the claim. Each hit is
    shown as a ``_HIT_TEXT_CHARS`` preview of its excerpt — read the page
    that cites it with ``read_page()`` when the evidence sits further in.
    """
    hits = [s for s in ctx.deps.sources if query.lower() in s["text"].lower()][:k]
    if not hits:
        return "(no matching sources found)"
    return "\n".join(f"- [{s['slug']}] {s['text'][:_HIT_TEXT_CHARS]}" for s in hits)


async def read_page(ctx: RunContext[VerificationDeps], slug: str) -> str:
    """Read the full content of a wiki page."""
    for page in ctx.deps.pages:
        if page["slug"] == slug:
            return page["content"]
    return f"Unknown page: {slug}"


async def mark_verified(ctx: RunContext[VerificationDeps], claim_id: int, evidence: str, sources: list[str]) -> str:
    """Record a verified verdict for *claim_id*."""
    ctx.deps.results.append(VerificationVerdict(claim_id=claim_id, verdict="supported", evidence=evidence, sources=sources))
    return f"Claim {claim_id} marked supported."


async def mark_contradicted(ctx: RunContext[VerificationDeps], claim_id: int, evidence: str, sources: list[str]) -> str:
    """Record a contradicted verdict for *claim_id*."""
    ctx.deps.results.append(VerificationVerdict(claim_id=claim_id, verdict="contradicted", evidence=evidence, sources=sources))
    return f"Claim {claim_id} marked contradicted."


async def mark_unverifiable(ctx: RunContext[VerificationDeps], claim_id: int, reason: str) -> str:
    """Record an unverifiable verdict for *claim_id*."""
    ctx.deps.results.append(VerificationVerdict(claim_id=claim_id, verdict="unverifiable", evidence=reason, sources=[]))
    return f"Claim {claim_id} marked unverifiable: {reason}"
