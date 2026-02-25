"""Direction Balancing for Benchmark v4.0 (Task 18).

Analyzes direction distribution after classification, identifies
underrepresented directions, and provides utilities for rebalancing.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from workflows.benchmark.models import BenchmarkQuery, Subgraph

logger = logging.getLogger(__name__)

# Target distribution percentages
DIRECTION_TARGETS: dict[str, float] = {
    "TD": 0.25,
    "HZ": 0.30,
    "BU": 0.25,
    "NT": 0.10,
    "CS": 0.10,
}

# Minimum acceptable fraction of target (80%)
MIN_TARGET_FRACTION = 0.80


@dataclass
class BalanceReport:
    """Summary of direction distribution analysis."""

    total_queries: int
    distribution: dict[str, int]
    targets: dict[str, int]
    deficits: dict[str, int]
    is_balanced: bool
    ambiguous_count: int
    ambiguous_pct: float


def compute_targets(total: int) -> dict[str, int]:
    """Compute absolute target counts from total query count."""
    return {direction: max(1, int(total * pct)) for direction, pct in DIRECTION_TARGETS.items()}


def compute_subgraph_allocation(subgraph: Subgraph, target_count: int) -> dict[str, int]:
    """Exact-sum direction allocation for one subgraph, skipping infeasible directions.

    Feasibility rules:
    - HZ requires ≥1 relation in subgraph.relations
    - BU requires ≥3 distinct parent_section_titles across nodes
    - CS requires ≥2 distinct parent_section_titles
    - NT/TD always feasible (need only ≥1 node)

    Uses largest-remainder method so sum(allocation.values()) == target_count.

    Args:
        subgraph: The subgraph to compute allocation for.
        target_count: Exact number of queries to allocate across directions.

    Returns:
        Dict mapping direction → count with guaranteed exact sum == target_count.
    """
    unique_sections = len(
        {n.parent_section_title for n in subgraph.nodes if n.parent_section_title}
    )
    has_relations = len(subgraph.relations) > 0

    # Build feasible subset using global ratios
    feasible: dict[str, float] = {
        "TD": DIRECTION_TARGETS["TD"],
        "NT": DIRECTION_TARGETS["NT"],
    }
    if has_relations:
        feasible["HZ"] = DIRECTION_TARGETS["HZ"]
    if unique_sections >= 3:
        feasible["BU"] = DIRECTION_TARGETS["BU"]
    if unique_sections >= 2:
        feasible["CS"] = DIRECTION_TARGETS["CS"]

    total_ratio = sum(feasible.values())
    raw = {d: target_count * r / total_ratio for d, r in feasible.items()}
    allocation = {d: int(v) for d, v in raw.items()}
    remainder = target_count - sum(allocation.values())
    # Assign leftover counts to directions with largest fractional parts
    for d, _ in sorted(raw.items(), key=lambda x: -(x[1] - int(x[1])))[:remainder]:
        allocation[d] += 1

    return allocation  # guaranteed: sum(allocation.values()) == target_count


def analyze_distribution(
    queries: list[BenchmarkQuery],
) -> BalanceReport:
    """Analyze direction distribution and identify deficits.

    Args:
        queries: Classified queries (must have direction field set).

    Returns:
        BalanceReport with distribution stats and deficit analysis.
    """
    total = len(queries)
    if total == 0:
        return BalanceReport(
            total_queries=0,
            distribution={},
            targets={},
            deficits={},
            is_balanced=True,
            ambiguous_count=0,
            ambiguous_pct=0.0,
        )

    dist = Counter(q.direction for q in queries)
    targets = compute_targets(total)

    deficits: dict[str, int] = {}
    for direction, target in targets.items():
        actual = dist.get(direction, 0)
        min_acceptable = int(target * MIN_TARGET_FRACTION)
        if actual < min_acceptable:
            deficits[direction] = target - actual

    ambiguous = dist.get("AMBIGUOUS", 0)

    return BalanceReport(
        total_queries=total,
        distribution=dict(dist),
        targets=targets,
        deficits=deficits,
        is_balanced=len(deficits) == 0,
        ambiguous_count=ambiguous,
        ambiguous_pct=(ambiguous / total * 100) if total > 0 else 0.0,
    )


def identify_rebalance_candidates(
    queries: list[BenchmarkQuery],
    deficit_direction: str,
) -> list[str]:
    """Identify subgraph IDs that could produce more queries of a given direction.

    Heuristics per direction:
        BU: Subgraphs with queries having ≥3 distinct parent sections
        CS: Subgraphs with queries having exactly 2 parent sections
        NT: Subgraphs with queries having ≤2 fact nodes, 1 section
        TD: Subgraphs with queries having ≥3 fact nodes, 1 section
        HZ: Subgraphs with queries having relations
    """
    subgraph_ids: set[str] = set()

    for q in queries:
        sections = {fn.parent_section for fn in q.fact_nodes}
        n_facts = len(q.fact_nodes)
        has_relations = len(q.relations_used) > 0

        match deficit_direction:
            case "BU":
                if len(sections) >= 3:
                    subgraph_ids.add(q.subgraph_id)
            case "CS":
                if len(sections) == 2 and not has_relations:
                    subgraph_ids.add(q.subgraph_id)
            case "NT":
                if n_facts <= 2 and len(sections) == 1:
                    subgraph_ids.add(q.subgraph_id)
            case "TD":
                if n_facts >= 3 and len(sections) == 1:
                    subgraph_ids.add(q.subgraph_id)
            case "HZ":
                if has_relations:
                    subgraph_ids.add(q.subgraph_id)

    return sorted(subgraph_ids)
