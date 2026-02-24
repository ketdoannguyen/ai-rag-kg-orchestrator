"""Rule-Based Direction Classifier for Benchmark v4.0.

Classifies benchmark queries into 5 directions (TD/HZ/BU/NT/CS) based on
the structure of fact_nodes — deterministic, not content-based.

Rule order matters (NT is a strict subset of TD conditions):
  NT → TD → HZ → BU → CS → AMBIGUOUS
"""

from __future__ import annotations

from collections import Counter

from workflows.benchmark.models import BenchmarkQuery

VALID_DIRECTIONS = ("TD", "HZ", "BU", "NT", "CS", "AMBIGUOUS")


def classify_direction(query: BenchmarkQuery) -> str:
    """Classify a single query's retrieval direction.

    Rules (applied in order):
        NT: ≤2 fact nodes, single section, no relations
        TD: single section, ≥3 fact nodes
        HZ: has any relation
        BU: ≥3 distinct sections, no relations
        CS: exactly 2 sections, no relations
        AMBIGUOUS: fallback
    """
    fact_nodes = query.fact_nodes
    relations = query.relations_used
    parent_sections = list({fn.parent_section for fn in fact_nodes})

    # Rule 1: Neutral — minimal, single-section, no relations
    if len(fact_nodes) <= 2 and len(parent_sections) == 1 and len(relations) == 0:
        return "NT"

    # Rule 2: Top-Down — many nodes in the same section
    if len(parent_sections) == 1 and len(fact_nodes) >= 3:
        return "TD"

    # Rule 3: Horizontal — involves cross-section relations
    if len(relations) >= 1:
        return "HZ"

    # Rule 4: Bottom-Up — many sections, no relations
    if len(parent_sections) >= 3 and len(relations) == 0:
        return "BU"

    # Rule 5: Cross-Section — exactly 2 sections, no relations
    if len(parent_sections) == 2 and len(relations) == 0:
        return "CS"

    return "AMBIGUOUS"


def classify_batch(
    queries: list[BenchmarkQuery],
) -> tuple[list[BenchmarkQuery], dict[str, int]]:
    """Classify all queries and return updated queries with distribution stats.

    Returns:
        (updated_queries, distribution) where distribution maps direction → count.
    """
    updated = []
    directions: list[str] = []
    for q in queries:
        direction = classify_direction(q)
        updated.append(q.model_copy(update={"direction": direction}))
        directions.append(direction)

    distribution = dict(Counter(directions))
    return updated, distribution
