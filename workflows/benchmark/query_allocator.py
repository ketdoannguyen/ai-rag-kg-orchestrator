"""Adaptive Query Allocator for Benchmark v4.0.

Allocates benchmark queries per subgraph proportional to core content node
count. v4.0: enriched nodes are excluded from allocation count.
"""

from __future__ import annotations

from workflows.benchmark.models import Subgraph

TIER_MULTIPLIERS: dict[str, float] = {
    "large": 3.0,
    "medium": 3.5,
    "small": 4.0,
    "tiny": 5.0,
}

TIER_MIN_QUERIES: dict[str, int] = {
    "large": 21,
    "medium": 14,
    "small": 8,
    "tiny": 5,
}

TARGET_TOTAL = 1500


def classify_tier(core_content_count: int) -> str:
    """Classify subgraph tier based on core content node count.

    Thresholds:
        large  : >= 7 core nodes
        medium : 4-6 core nodes
        small  : 2-3 core nodes
        tiny   : <= 1 core node
    """
    if core_content_count >= 7:
        return "large"
    if core_content_count >= 4:
        return "medium"
    if core_content_count >= 2:
        return "small"
    return "tiny"


def allocate_queries(core_content_count: int) -> int:
    """Compute raw query allocation for a subgraph.

    Formula: max(MIN_QUERIES[tier], floor(core_count * MULTIPLIER[tier]))
    """
    tier = classify_tier(core_content_count)
    raw = int(core_content_count * TIER_MULTIPLIERS[tier])
    return max(TIER_MIN_QUERIES[tier], raw)


def balance_allocations(
    subgraphs: list[Subgraph],
    target_total: int = TARGET_TOTAL,
) -> list[Subgraph]:
    """Adjust per-subgraph allocations so the sum approximates target_total.

    Applies a uniform scale factor. Returns new Subgraph instances with updated
    query_allocation fields (original list is not modified).
    """
    if not subgraphs:
        return []

    # Compute raw allocations
    raw_allocs = [allocate_queries(sg.content_count) for sg in subgraphs]
    total_raw = sum(raw_allocs)

    if total_raw == 0:
        # All subgraphs are empty; assign 0 to everyone
        return [sg.model_copy(update={"query_allocation": 0}) for sg in subgraphs]

    scale = target_total / total_raw

    updated = []
    for sg, raw in zip(subgraphs, raw_allocs, strict=True):
        adjusted = max(1, round(raw * scale))
        updated.append(sg.model_copy(update={"query_allocation": adjusted}))

    return updated


def assign_allocations(subgraphs: list[Subgraph]) -> list[Subgraph]:
    """Compute and assign raw (unbalanced) allocations to each subgraph."""
    return [
        sg.model_copy(update={"query_allocation": allocate_queries(sg.content_count)})
        for sg in subgraphs
    ]
