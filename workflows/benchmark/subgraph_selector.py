"""Subgraph Selection & Extraction for Benchmark v4.0 (Task 10).

From candidate sections (T9), selects ~80-100 subgraphs, runs
SEED+ENRICH+BOUND extraction (T3), computes adaptive allocation (T4).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from workflows.benchmark.models import Subgraph
from workflows.benchmark.query_allocator import balance_allocations
from workflows.benchmark.section_lister import CandidateSection
from workflows.benchmark.subgraph_extractor import SubgraphExtractor

logger = logging.getLogger(__name__)

MIN_CONTENT_NODES = 1


@dataclass
class SelectionReport:
    """Summary statistics for a subgraph selection run."""

    total_candidates: int
    selected_count: int
    filtered_out: int
    tier_distribution: dict[str, int]
    source_count: int
    sources: list[str]
    total_allocation: int
    avg_relation_coverage: float


def filter_candidates(
    candidates: list[CandidateSection],
    min_content: int = MIN_CONTENT_NODES,
    require_summary: bool = False,
) -> list[CandidateSection]:
    """Filter candidate sections that are too small or lack summary.

    Args:
        candidates: All L5 candidates from section_lister.
        min_content: Minimum number of content nodes.
        require_summary: If True, reject sections without summary.
    """
    filtered = []
    for c in candidates:
        if c.content_count < min_content:
            continue
        if require_summary and not c.has_summary:
            continue
        filtered.append(c)
    return filtered


def select_and_extract(
    client,
    candidates: list[CandidateSection],
    max_subgraphs: int = 100,
) -> tuple[list[Subgraph], SelectionReport]:
    """Select subgraphs from candidates, extract, and compute allocations.

    Pipeline:
        1. Filter candidates (min content nodes)
        2. Extract subgraph for each via SEED+ENRICH+BOUND
        3. Skip empty subgraphs (0 nodes)
        4. Compute adaptive query allocation
        5. Generate report

    Args:
        client: ArangoDB client.
        candidates: Candidate sections from section_lister.
        max_subgraphs: Maximum number of subgraphs to extract.

    Returns:
        (subgraphs, report)
    """
    extractor = SubgraphExtractor(client)

    # Step 1: Filter
    filtered = filter_candidates(candidates)
    logger.info(
        "Filtered %d → %d candidates (min_content=%d)",
        len(candidates),
        len(filtered),
        MIN_CONTENT_NODES,
    )

    # Step 2: Cap at max_subgraphs with diversity (≥3 sources when possible)
    selected = _select_diverse(filtered, max_subgraphs)

    # Step 3: Extract subgraphs
    subgraphs: list[Subgraph] = []
    coverage_values: list[float] = []

    for i, candidate in enumerate(selected):
        breadcrumb = _build_candidate_breadcrumb(candidate)
        sg = extractor.extract(
            anchor_key=candidate.section_key,
            anchor_title=candidate.title,
            anchor_summary=candidate.summary,
            breadcrumb=breadcrumb,
        )

        # Skip empty subgraphs (extract returns None when no content nodes)
        if sg is None:
            logger.debug("Skipping empty subgraph for %s", candidate.section_id)
            continue

        # Compute relation coverage
        coverage = extractor.compute_relation_coverage(sg)
        coverage_values.append(coverage)

        subgraphs.append(sg)

        if (i + 1) % 20 == 0:
            logger.info("Extracted %d/%d subgraphs", i + 1, len(selected))

    # Step 4: Compute balanced allocations
    subgraphs = balance_allocations(subgraphs)

    # Step 5: Report
    tier_dist = dict(Counter(sg.tier for sg in subgraphs))
    sources = list({c.source for c in selected if c.source})
    avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0

    report = SelectionReport(
        total_candidates=len(candidates),
        selected_count=len(subgraphs),
        filtered_out=len(candidates) - len(filtered),
        tier_distribution=tier_dist,
        source_count=len(sources),
        sources=sources,
        total_allocation=sum(sg.query_allocation for sg in subgraphs),
        avg_relation_coverage=avg_coverage,
    )

    logger.info(
        "Selection complete: %d subgraphs, %d total allocation, %.1f%% avg relation coverage",
        report.selected_count,
        report.total_allocation,
        report.avg_relation_coverage * 100,
    )

    return subgraphs, report


MIN_DIVERSE_SOURCES = 3


def _select_diverse(
    candidates: list[CandidateSection],
    max_subgraphs: int,
) -> list[CandidateSection]:
    """Select up to max_subgraphs candidates with source diversity.

    Takes the top candidates (already sorted by relation_count desc).
    If unique sources in the top-N < MIN_DIVERSE_SOURCES, pads from
    underrepresented sources by swapping in candidates from missing sources.

    Args:
        candidates: Filtered and sorted candidates.
        max_subgraphs: Maximum number to return.

    Returns:
        List of selected candidates with ≥3 sources when possible.
    """
    if len(candidates) <= max_subgraphs:
        return candidates

    top = candidates[:max_subgraphs]
    top_sources = {c.source for c in top if c.source}

    if len(top_sources) >= MIN_DIVERSE_SOURCES:
        return top

    # Find candidates from sources not in top selection
    remaining = candidates[max_subgraphs:]
    missing_source_candidates: list[CandidateSection] = []
    for c in remaining:
        if c.source and c.source not in top_sources:
            missing_source_candidates.append(c)
            top_sources.add(c.source)
            if len(top_sources) >= MIN_DIVERSE_SOURCES:
                break

    if not missing_source_candidates:
        return top

    # Replace lowest-ranked entries in top with diverse candidates
    replacements_needed = len(missing_source_candidates)
    selected = top[: max_subgraphs - replacements_needed] + missing_source_candidates
    return selected[:max_subgraphs]


def _build_candidate_breadcrumb(candidate: CandidateSection) -> str:
    """Build a breadcrumb string from candidate metadata."""
    parts = []
    if candidate.parent_title:
        parts.append(candidate.parent_title)
    parts.append(candidate.title)
    return " > ".join(parts)
