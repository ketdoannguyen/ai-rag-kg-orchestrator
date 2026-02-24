"""Unit tests for workflows/benchmark/query_allocator.py.

Covers: classify_tier, allocate_queries, balance_allocations, assign_allocations.
"""

from __future__ import annotations

import pytest

from workflows.benchmark.models import Subgraph, SubgraphNode
from workflows.benchmark.query_allocator import (
    TARGET_TOTAL,
    TIER_MIN_QUERIES,
    TIER_MULTIPLIERS,
    allocate_queries,
    assign_allocations,
    balance_allocations,
    classify_tier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_minimal_subgraph(
    subgraph_id: str = "sg-001",
    content_count: int = 1,
    tier: str = "tiny",
    query_allocation: int = 0,
) -> Subgraph:
    """Create a minimal valid Subgraph for allocation tests."""
    node = SubgraphNode(node_id="content_nodes/n1", text="text")
    return Subgraph(
        subgraph_id=subgraph_id,
        anchor_key="s1",
        anchor_title="Test",
        breadcrumb="A > B",
        nodes=[node],
        core_node_ids=["content_nodes/n1"],
        enriched_node_ids=[],
        relations=[],
        content_count=content_count,
        relation_count=0,
        tier=tier,
        query_allocation=query_allocation,
    )


# ---------------------------------------------------------------------------
# classify_tier
# ---------------------------------------------------------------------------


class TestClassifyTier:
    @pytest.mark.parametrize(
        "count,expected",
        [
            (7, "large"),
            (8, "large"),
            (10, "large"),
            (100, "large"),
        ],
    )
    def test_large_tier(self, count, expected):
        assert classify_tier(count) == expected

    @pytest.mark.parametrize(
        "count,expected",
        [
            (4, "medium"),
            (5, "medium"),
            (6, "medium"),
        ],
    )
    def test_medium_tier(self, count, expected):
        assert classify_tier(count) == expected

    @pytest.mark.parametrize(
        "count,expected",
        [
            (2, "small"),
            (3, "small"),
        ],
    )
    def test_small_tier(self, count, expected):
        assert classify_tier(count) == expected

    @pytest.mark.parametrize(
        "count,expected",
        [
            (0, "tiny"),
            (1, "tiny"),
        ],
    )
    def test_tiny_tier(self, count, expected):
        assert classify_tier(count) == expected

    def test_boundary_7_is_large_not_medium(self):
        assert classify_tier(7) == "large"
        assert classify_tier(6) == "medium"

    def test_boundary_4_is_medium_not_small(self):
        assert classify_tier(4) == "medium"
        assert classify_tier(3) == "small"

    def test_boundary_2_is_small_not_tiny(self):
        assert classify_tier(2) == "small"
        assert classify_tier(1) == "tiny"


# ---------------------------------------------------------------------------
# allocate_queries
# ---------------------------------------------------------------------------


class TestAllocateQueries:
    def test_uses_min_for_zero_core_nodes(self):
        # 0 nodes → tiny, raw = 0 * 5.0 = 0, min = 5 → result = 5
        result = allocate_queries(0)
        assert result == TIER_MIN_QUERIES["tiny"]

    def test_uses_min_for_single_node(self):
        # 1 node → tiny, raw = 1 * 5.0 = 5, min = 5 → result = 5
        result = allocate_queries(1)
        assert result == max(TIER_MIN_QUERIES["tiny"], int(1 * TIER_MULTIPLIERS["tiny"]))

    def test_formula_large(self):
        count = 10
        tier = "large"
        expected = max(TIER_MIN_QUERIES[tier], int(count * TIER_MULTIPLIERS[tier]))
        assert allocate_queries(count) == expected

    def test_formula_medium(self):
        count = 5
        tier = "medium"
        expected = max(TIER_MIN_QUERIES[tier], int(count * TIER_MULTIPLIERS[tier]))
        assert allocate_queries(count) == expected

    def test_formula_small(self):
        count = 2
        tier = "small"
        expected = max(TIER_MIN_QUERIES[tier], int(count * TIER_MULTIPLIERS[tier]))
        assert allocate_queries(count) == expected

    def test_formula_tiny(self):
        count = 1
        tier = "tiny"
        expected = max(TIER_MIN_QUERIES[tier], int(count * TIER_MULTIPLIERS[tier]))
        assert allocate_queries(count) == expected

    def test_result_always_at_least_min(self):
        # Very low counts should never go below the tier minimum
        for count in range(0, 8):
            tier = classify_tier(count)
            assert allocate_queries(count) >= TIER_MIN_QUERIES[tier]

    def test_large_count_exceeds_minimum(self):
        # 20 large nodes: 20 * 3.0 = 60, min = 21 → should return 60
        result = allocate_queries(20)
        assert result == 60

    def test_floor_applied_to_multiplication(self):
        # int() truncates toward zero (floor for positives)
        count = 3  # small tier, multiplier = 4.0
        raw = int(3 * TIER_MULTIPLIERS["small"])  # int(12.0) = 12
        expected = max(TIER_MIN_QUERIES["small"], raw)
        assert allocate_queries(count) == expected


# ---------------------------------------------------------------------------
# balance_allocations
# ---------------------------------------------------------------------------


class TestBalanceAllocations:
    def test_empty_list_returns_empty(self):
        result = balance_allocations([])
        assert result == []

    def test_zero_content_still_gets_min_allocation(self):
        # allocate_queries(0) returns TIER_MIN_QUERIES["tiny"] = 5 (never 0),
        # so balance_allocations still produces non-zero allocations for
        # subgraphs with content_count=0.  The total_raw==0 guard inside
        # balance_allocations is a defensive check that cannot be triggered
        # through the normal allocate_queries path.
        subgraphs = [
            make_minimal_subgraph(f"sg-{i}", content_count=0, tier="tiny") for i in range(3)
        ]
        result = balance_allocations(subgraphs)
        # Each subgraph still gets at least 1 query after scaling
        assert all(sg.query_allocation >= 1 for sg in result)

    def test_sum_approximates_target_total(self):
        subgraphs = [
            make_minimal_subgraph(f"sg-{i}", content_count=3, tier="small") for i in range(10)
        ]
        result = balance_allocations(subgraphs, target_total=100)
        total = sum(sg.query_allocation for sg in result)
        # Allow ±len(subgraphs) rounding variance
        assert abs(total - 100) <= len(subgraphs)

    def test_default_target_is_1500(self):
        subgraphs = [
            make_minimal_subgraph(f"sg-{i}", content_count=5, tier="medium") for i in range(20)
        ]
        result = balance_allocations(subgraphs)
        total = sum(sg.query_allocation for sg in result)
        # Allow ±len(subgraphs) rounding variance
        assert abs(total - TARGET_TOTAL) <= len(subgraphs)

    def test_original_subgraphs_not_modified(self):
        subgraphs = [
            make_minimal_subgraph("sg-001", content_count=5, tier="medium", query_allocation=0)
        ]
        original_allocation = subgraphs[0].query_allocation
        balance_allocations(subgraphs)
        assert subgraphs[0].query_allocation == original_allocation

    def test_returns_new_subgraph_instances(self):
        subgraphs = [make_minimal_subgraph("sg-001", content_count=5, tier="medium")]
        result = balance_allocations(subgraphs)
        assert result[0] is not subgraphs[0]

    def test_custom_target_total(self):
        subgraphs = [
            make_minimal_subgraph(f"sg-{i}", content_count=3, tier="small") for i in range(5)
        ]
        result = balance_allocations(subgraphs, target_total=500)
        total = sum(sg.query_allocation for sg in result)
        assert abs(total - 500) <= len(subgraphs)

    def test_each_allocation_at_least_one(self):
        # With non-zero content, each subgraph should get at least 1 query
        subgraphs = [
            make_minimal_subgraph(f"sg-{i}", content_count=1, tier="tiny") for i in range(3)
        ]
        result = balance_allocations(subgraphs, target_total=TARGET_TOTAL)
        assert all(sg.query_allocation >= 1 for sg in result)

    def test_preserves_order(self):
        subgraphs = [
            make_minimal_subgraph(f"sg-{i}", content_count=i + 1, tier="tiny") for i in range(5)
        ]
        result = balance_allocations(subgraphs)
        for original, updated in zip(subgraphs, result):
            assert updated.subgraph_id == original.subgraph_id

    def test_single_subgraph_gets_nonzero_allocation(self):
        subgraphs = [make_minimal_subgraph("sg-001", content_count=3, tier="small")]
        result = balance_allocations(subgraphs, target_total=100)
        assert result[0].query_allocation >= 1


# ---------------------------------------------------------------------------
# assign_allocations
# ---------------------------------------------------------------------------


class TestAssignAllocations:
    def test_assigns_raw_allocations(self):
        count = 5  # medium tier
        sg = make_minimal_subgraph("sg-001", content_count=count, tier="medium")
        result = assign_allocations([sg])
        expected = allocate_queries(count)
        assert result[0].query_allocation == expected

    def test_empty_list(self):
        assert assign_allocations([]) == []

    def test_multiple_subgraphs(self):
        subgraphs = [
            make_minimal_subgraph("sg-1", content_count=1, tier="tiny"),
            make_minimal_subgraph("sg-2", content_count=3, tier="small"),
            make_minimal_subgraph("sg-3", content_count=5, tier="medium"),
            make_minimal_subgraph("sg-4", content_count=8, tier="large"),
        ]
        result = assign_allocations(subgraphs)
        assert len(result) == 4
        for sg, res in zip(subgraphs, result):
            assert res.query_allocation == allocate_queries(sg.content_count)

    def test_original_not_modified(self):
        sg = make_minimal_subgraph("sg-001", content_count=5, tier="medium", query_allocation=0)
        assign_allocations([sg])
        assert sg.query_allocation == 0

    def test_returns_new_instances(self):
        sg = make_minimal_subgraph("sg-001", content_count=5, tier="medium")
        result = assign_allocations([sg])
        assert result[0] is not sg
