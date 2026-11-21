"""The synthetic corpus the traversal spike runs on.

The bead allows a synthetic cut when the pyvas document is out of reach. What
the spike needs from the corpus is a tree of the same *shape* (deep, wide, one
eligible internal node per section) with content a model can judge:

- dense technical prose in the branches that deserve a summary;
- boilerplate branches (legal notices, revision histories, a table of
  contents) that deserve a skip;
- the same safety notice repeated across chapters — repeated material;
- a few facts a reader would ask for, planted in named branches.

Everything is seeded, so both variants walk the identical tree.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from common import PlantedFact, SyntheticDoc

from kompend.core.models.chunks import Chunk
from kompend.core.models.enums import ChunkStrategy, ElementType
from kompend.core.models.trees import ChunkNode, ChunkTree

#: Sections whose content is boilerplate — a skip is the right call here.
BOILERPLATE_HEADINGS: tuple[str, ...] = (
    "Legal Notice",
    "Copyright and Trademarks",
    "Table of Contents",
    "Revision History",
    "Safety Notice",
    "Glossary of Terms",
)

#: The boilerplate text a boilerplate branch carries. The safety notice is
#: repeated verbatim across chapters on purpose: repeated material is a
#: different skip reason than boilerplate.
_BOILERPLATE_TEXT: dict[str, str] = {  # nocheck: str-registry — the keys are the document's own section headings (data), not a code vocabulary
    "Legal Notice": (
        "This document is provided for informational purposes only. No warranties of any kind are "
        "expressed or implied. In no event shall the publisher be liable for any loss or damage "
        "arising from the use of this material. The information herein is subject to change without "
        "notice, and the publisher assumes no obligation to update it."
    ),
    "Copyright and Trademarks": (
        "All rights reserved. Reproduction or transmission of this document, in whole or in part, "
        "without the prior written permission of the publisher is prohibited. Product names and "
        "logos referenced herein are the property of their respective owners and are used for "
        "identification only."
    ),
    "Table of Contents": (
        "This table of contents is generated automatically from the document headings. Page "
        "references point at the rendered wiki rather than at a fixed pagination, and the list is "
        "rebuilt whenever the document is recompiled."
    ),
    "Revision History": (
        "Version 1.0 - initial release. Version 1.1 - clarified the configuration examples. "
        "Version 1.2 - added the troubleshooting chapter. Version 2.0 - restructured the "
        "deployment sections. Version 2.1 - editorial corrections throughout."
    ),
    "Safety Notice": (
        "Before performing any maintenance on the appliance, disconnect the power supply and "
        "verify that the grounding conductor is attached. The unit contains no user-serviceable "
        "components. Opening the chassis voids the warranty and exposes the operator to "
        "electric shock."
    ),
    "Glossary of Terms": (
        "Terminology used throughout this manual is defined here. Where a term has a meaning "
        "specific to the platform, that meaning is stated explicitly; otherwise the general "
        "industry meaning applies."
    ),
}


@dataclass(frozen=True)
class _Chapter:
    """One chapter of the synthetic manual."""

    title: str
    sections: tuple[str, ...]
    nouns: tuple[str, ...]
    adjectives: tuple[str, ...]


_CHAPTERS: tuple[_Chapter, ...] = (
    _Chapter(
        title="Installation and Setup",
        sections=("Unpacking the Unit", "Rack Preparation", "First Power-On", "Initial Configuration", "Firmware Update", "Legal Notice"),
        nouns=("chassis", "power supply", "mounting rail", "cable harness", "front panel", "kickplate"),
        adjectives=("rear", "front", "redundant", "removable"),
    ),
    _Chapter(
        title="System Requirements",
        sections=("Supported Hardware", "Operating Systems", "Memory and Storage", "Network Interfaces", "Environmental Limits", "Safety Notice"),
        nouns=("node", "volume", "interface", "kernel", "scheduler", "memory pool"),
        adjectives=("supported", "minimal", "recommended", "physical"),
    ),
    _Chapter(
        title="Network Configuration",
        sections=("Addressing", "Routing", "Firewalls and Ports", "TLS Termination", "Service Discovery", "Table of Contents"),
        nouns=("subnet", "gateway", "certificate", "listener", "endpoint", "resolver"),
        adjectives=("private", "public", "static", "dynamic"),
    ),
    _Chapter(
        title="Storage and Backup",
        sections=("Volume Layout", "Snapshot Policy", "Retention Rules", "Restore Procedure", "Capacity Planning", "Revision History"),
        nouns=("volume", "snapshot", "journal", "tier", "bucket", "manifest"),
        adjectives=("warm", "cold", "immutable", "compressed"),
    ),
    _Chapter(
        title="Security and Access Control",
        sections=("Identity Providers", "Roles and Permissions", "Secret Handling", "Audit Logging", "Session Policy", "Copyright and Trademarks"),
        nouns=("principal", "role binding", "credential", "audit trail", "policy", "token"),
        adjectives=("least-privilege", "rotated", "scoped", "immutable"),
    ),
    _Chapter(
        title="Monitoring and Alerting",
        sections=("Metrics and Counters", "Health Checks", "Alert Routing", "Dashboards", "Log Shipping", "Glossary of Terms"),
        nouns=("counter", "probe", "sink", "dashboard", "gauge", "window"),
        adjectives=("cumulative", "per-tenant", "sampled", "streamed"),
    ),
    _Chapter(
        title="Scaling and Load Balancing",
        sections=("Horizontal Scaling", "Shard Placement", "Queue Depth", "Rate Limiting", "Autoscaling Signals", "Legal Notice"),
        nouns=("replica", "shard", "queue", "limiter", "pool", "scheduler"),
        adjectives=("saturated", "under-provisioned", "elastic", "pinned"),
    ),
    _Chapter(
        title="Data Import and Export",
        sections=("Supported Formats", "Chunked Ingestion", "Schema Mapping", "Export Pipelines", "Validation Runs", "Safety Notice"),
        nouns=("parser", "record", "schema", "pipeline", "validator", "batch"),
        adjectives=("nested", "incremental", "idempotent", "verified"),
    ),
    _Chapter(
        title="API Reference",
        sections=("Authentication", "Core Endpoints", "Pagination", "Error Semantics", "Rate Limits", "Table of Contents"),
        nouns=("request", "response", "cursor", "envelope", "status code", "header"),
        adjectives=("idempotent", "authorised", "paginated", "versioned"),
    ),
    _Chapter(
        title="Troubleshooting",
        sections=("Diagnostic Commands", "Common Failures", "Log Analysis", "Escalation Path", "Known Issues", "Revision History"),
        nouns=("diagnostic", "stack trace", "probe", "runbook", "ticket", "trace"),
        adjectives=("transient", "persistent", "misleading", "correlated"),
    ),
    _Chapter(
        title="Upgrade Procedures",
        sections=("Version Matrix", "Rolling Upgrades", "Rollback Plan", "Compatibility Notes", "Post-Upgrade Checks", "Copyright and Trademarks"),
        nouns=("release", "canary", "replica set", "manifest", "migration", "health check"),
        adjectives=("rolling", "reversible", "staged", "deprecated"),
    ),
    _Chapter(
        title="Compliance and Audit",
        sections=("Retention Policy", "Data Residency", "Evidence Collection", "Control Mapping", "Certification Status", "Glossary of Terms"),
        nouns=("control", "evidence bundle", "attestation", "residency region", "ledger", "scope"),
        adjectives=("attested", "region-scoped", "immutable", "documented"),
    ),
)

#: The facts a reader would ask the wiki for. Each is planted verbatim in one
#: leaf of a named chapter, so a coverage check can look for it.
_FACTS: tuple[tuple[str, str, str], ...] = (
    (
        "System Requirements",
        "What is the minimum memory the platform supports per node?",
        "The platform requires at least 16 GiB of memory per node.",
    ),
    (
        "Network Configuration",
        "Which ports must be open between nodes?",
        "Ports 7946 for the cluster protocol, 2380 for the peer channel and 9090 for metrics must be open between nodes.",
    ),
    (
        "Storage and Backup",
        "How long are snapshots retained by default?",
        "Snapshots are retained for 35 days by default.",
    ),
    (
        "Security and Access Control",
        "How long does a session stay valid without a refresh?",
        "A session stays valid for 30 minutes without a refresh.",
    ),
    (
        "API Reference",
        "What is the maximum number of items a page may carry?",
        "A page may carry at most 200 items.",
    ),
)

#: Subsection titles, generic enough to sit under any section.
_SUBSECTIONS: tuple[str, ...] = (
    "Overview",
    "Requirements",
    "Procedure",
    "Configuration",
    "Verification",
    "Notes",
    "Failure Modes",
    "Limits",
)


def _sentences(rng: random.Random, chapter: _Chapter, count: int) -> list[str]:
    """Plausible technical sentences for one chapter."""
    nouns = chapter.nouns
    adjectives = chapter.adjectives
    out: list[str] = []
    for _ in range(count):
        noun = rng.choice(nouns)
        adj = rng.choice(adjectives)
        other = rng.choice(nouns)
        template = rng.randint(0, 5)
        if template == 0:
            out.append(f"The {adj} {noun} is reconciled every {rng.choice((5, 15, 30, 60))} seconds against the {other} registry.")
        elif template == 1:
            out.append(f"When the {noun} fails over, the {adj} {other} takes over the outstanding requests within one heartbeat interval.")
        elif template == 2:
            out.append(f"Operators should size the {noun} so that a {adj} {other} does not saturate it during peak traffic.")
        elif template == 3:
            out.append(f"The {noun} writes its state to the {adj} {other} before it acknowledges the write to the caller.")
        elif template == 4:
            out.append(f"Each {adj} {noun} carries a bounded queue; once the queue fills, requests are rejected with the documented back-pressure status.")
        else:
            out.append(f"Tuning the {noun} is a matter of balancing {adj} {other} throughput against the memory the {noun} reserves for in-flight work.")
    return out


def _leaf_text(rng: random.Random, chapter: _Chapter, *, planted: str | None) -> str:
    """The text of one leaf."""
    parts = _sentences(rng, chapter, rng.randint(3, 6))
    if planted is not None:
        parts.insert(rng.randint(0, len(parts)), planted)
    return " ".join(parts)


def build_synthetic_doc(*, seed: int = 20261121) -> SyntheticDoc:
    """Build the corpus both variants walk.

    The tree is four levels deep: root, chapters, sections, subsections, and
    a leaf chunk per subsection (two or three per subsection). Roughly 1,100
    nodes, with one eligible internal node per section — the shape the pyvas
    document has.
    """
    rng = random.Random(seed)  # noqa: S311 — a seeded generator makes the corpus reproducible, not secret
    nodes: dict[str, ChunkNode] = {}
    facts: list[PlantedFact] = []
    boilerplate_ids: set[str] = set()
    answer_bearing_ids: set[str] = set()
    counter = 0

    def next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:05d}"

    root = ChunkNode(id="root", parent_id=None, children_ids=[], depth=0, heading_chain=())
    nodes[root.id] = root

    for _chapter_index, chapter in enumerate(_CHAPTERS):
        chapter_id = next_id("chap")
        chapter_node = ChunkNode(
            id=chapter_id,
            parent_id=root.id,
            children_ids=[],
            depth=1,
            heading_chain=(chapter.title,),
        )
        nodes[chapter_id] = chapter_node
        root.children_ids.append(chapter_id)

        for section_index, section_title in enumerate(chapter.sections):
            section_id = next_id("sec")
            is_boilerplate = section_title in BOILERPLATE_HEADINGS
            section_node = ChunkNode(
                id=section_id,
                parent_id=chapter_id,
                children_ids=[],
                depth=2,
                heading_chain=(chapter.title, section_title),
            )
            nodes[section_id] = section_node
            chapter_node.children_ids.append(section_id)
            if is_boilerplate:
                boilerplate_ids.add(section_id)

            for subsection_index, subsection_title in enumerate(_SUBSECTIONS[: rng.randint(3, 5)]):
                subsection_id = next_id("sub")
                subsection_node = ChunkNode(
                    id=subsection_id,
                    parent_id=section_id,
                    children_ids=[],
                    depth=3,
                    heading_chain=(chapter.title, section_title, subsection_title),
                )
                nodes[subsection_id] = subsection_node
                section_node.children_ids.append(subsection_id)

                # The planted fact of this chapter goes into the first
                # subsection of the chapter's second section, so it is buried
                # rather than sitting next to the chapter heading.
                planted: str | None = None
                for fact_chapter, question, answer in _FACTS:
                    if fact_chapter != chapter.title:
                        continue
                    if section_index == 1 and subsection_index == 0:
                        planted = answer
                        answer_bearing_ids.add(section_id)
                        facts.append(
                            PlantedFact(
                                question=question,
                                answer=answer,
                                node_id=subsection_id,
                                chapter=chapter.title,
                            )
                        )

                for leaf_index in range(rng.randint(2, 4)):
                    leaf_id = next_id("leaf")
                    text = _BOILERPLATE_TEXT[section_title] if is_boilerplate else _leaf_text(rng, chapter, planted=planted if leaf_index == 0 else None)
                    leaf = Chunk(
                        id=leaf_id,
                        text=text,
                        element_types=[ElementType.TEXT],
                        source_elements=[counter],
                        heading_context=" / ".join((chapter.title, section_title, subsection_title)),
                        token_count=max(1, len(text) // 4),
                    )
                    nodes[leaf_id] = ChunkNode(
                        id=leaf_id,
                        parent_id=subsection_id,
                        children_ids=[],
                        depth=4,
                        heading_chain=(chapter.title, section_title, subsection_title),
                        chunk=leaf,
                    )
                    subsection_node.children_ids.append(leaf_id)

    tree = ChunkTree(
        root_id=root.id,
        nodes=nodes,
        strategy=ChunkStrategy.MULTI_RESOLUTION,
    )
    return SyntheticDoc(
        tree=tree,
        facts=tuple(facts),
        boilerplate_ids=frozenset(boilerplate_ids),
        answer_bearing_ids=frozenset(answer_bearing_ids),
        chapter_headings=tuple(chapter.title for chapter in _CHAPTERS),
    )
