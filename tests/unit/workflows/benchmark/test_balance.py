"""Tests for balance.py — direction distribution analysis."""

from __future__ import annotations

from workflows.benchmark.balance import (
    DIRECTION_TARGETS,
    BalanceReport,
    analyze_distribution,
    compute_subgraph_allocation,
    compute_targets,
    identify_rebalance_candidates,
)
from workflows.benchmark.models import (
    BenchmarkQuery,
    ConfidenceScore,
    FactNode,
    Subgraph,
    SubgraphNode,
    SubgraphRelation,
)


def _make_query(
    query_id: str,
    direction: str = "TD",
    fact_nodes: list[FactNode] | None = None,
    relations_used: list[str] | None = None,
) -> BenchmarkQuery:
    if fact_nodes is None:
        fact_nodes = [FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a")]
    return BenchmarkQuery(
        query_id=query_id,
        query_text="Test question?",
        subgraph_id="sg_test",
        fact_nodes=fact_nodes,
        relations_used=relations_used or [],
        confidence=ConfidenceScore(
            question_clarity=8,
            fact_node_completeness=8,
            fact_node_precision=8,
            grade_certainty=8,
            overall=80,
        ),
        confidence_concerns="",
        reasoning="Test",
        direction=direction,
    )


class TestComputeTargets:
    def test_proportions_sum_to_total(self):
        total = 500
        targets = compute_targets(total)
        # Each target = int(total * pct), all should be positive
        for direction, pct in DIRECTION_TARGETS.items():
            expected = max(1, int(total * pct))
            assert targets[direction] == expected

    def test_zero_total_gives_ones(self):
        targets = compute_targets(0)
        for v in targets.values():
            assert v == 1

    def test_all_directions_present(self):
        targets = compute_targets(100)
        assert set(targets.keys()) == set(DIRECTION_TARGETS.keys())

    def test_small_total(self):
        targets = compute_targets(1)
        for v in targets.values():
            assert v >= 1


class TestAnalyzeDistribution:
    def test_empty_queries(self):
        report = analyze_distribution([])
        assert report.total_queries == 0
        assert report.is_balanced is True
        assert report.ambiguous_count == 0

    def test_perfectly_balanced(self):
        # 100 total with exact target percentages
        total = 100
        targets = compute_targets(total)
        queries = []
        for direction, count in targets.items():
            queries.extend(_make_query(f"{direction}_{i}", direction) for i in range(count))

        report = analyze_distribution(queries)
        assert report.total_queries == len(queries)
        # Should have no or minimal deficits
        for direction in DIRECTION_TARGETS:
            assert report.distribution.get(direction, 0) > 0

    def test_deficit_detected(self):
        # Only TD queries — BU, HZ, NT, CS all deficient
        queries = [_make_query(f"q{i}", "TD") for i in range(100)]
        report = analyze_distribution(queries)
        assert not report.is_balanced
        assert "BU" in report.deficits or "HZ" in report.deficits

    def test_ambiguous_counted(self):
        queries = [_make_query(f"q{i}", "AMBIGUOUS") for i in range(10)]
        report = analyze_distribution(queries)
        assert report.ambiguous_count == 10
        assert report.ambiguous_pct == 100.0

    def test_report_fields(self):
        queries = [_make_query("q1", "TD"), _make_query("q2", "HZ")]
        report = analyze_distribution(queries)
        assert isinstance(report, BalanceReport)
        assert report.total_queries == 2
        assert "TD" in report.distribution
        assert "HZ" in report.distribution


class TestIdentifyRebalanceCandidates:
    def _make_multi_section_query(
        self,
        query_id: str,
        direction: str,
        n_sections: int,
        n_facts: int = 2,
        has_relations: bool = False,
    ) -> BenchmarkQuery:
        fact_nodes = [
            FactNode(
                node_id=f"content_nodes/n{i}",
                grade=2,
                parent_section=f"sec/{i % n_sections}",
            )
            for i in range(n_facts)
        ]
        return BenchmarkQuery(
            query_id=query_id,
            query_text="Test?",
            subgraph_id="sg_test",
            fact_nodes=fact_nodes,
            relations_used=["REL1"] if has_relations else [],
            confidence=ConfidenceScore(
                question_clarity=8,
                fact_node_completeness=8,
                fact_node_precision=8,
                grade_certainty=8,
                overall=80,
            ),
            confidence_concerns="",
            reasoning="Test",
            direction=direction,
        )

    def test_bu_requires_3_sections(self):
        q_3sec = self._make_multi_section_query("q1", "TD", n_sections=3, n_facts=3)
        q_2sec = self._make_multi_section_query("q2", "TD", n_sections=2, n_facts=2)
        candidates = identify_rebalance_candidates([q_3sec, q_2sec], "BU")
        assert "sg_test" in candidates
        # Both from same subgraph, so only unique subgraphs returned
        assert len(candidates) == 1

    def test_hz_requires_relations(self):
        q_with_rel = self._make_multi_section_query("q1", "TD", n_sections=1, has_relations=True)
        q_no_rel = self._make_multi_section_query("q2", "TD", n_sections=1, has_relations=False)
        candidates_rel = identify_rebalance_candidates([q_with_rel], "HZ")
        candidates_no_rel = identify_rebalance_candidates([q_no_rel], "HZ")
        assert "sg_test" in candidates_rel
        assert "sg_test" not in candidates_no_rel

    def test_nt_requires_few_facts_one_section(self):
        # NT: n_facts <= 2 and 1 section
        q_nt = self._make_multi_section_query("q1", "TD", n_sections=1, n_facts=2)
        q_not_nt = self._make_multi_section_query("q2", "TD", n_sections=2, n_facts=2)
        candidates = identify_rebalance_candidates([q_nt, q_not_nt], "NT")
        assert "sg_test" in candidates

    def test_td_requires_many_facts_one_section(self):
        # TD: n_facts >= 3 and 1 section
        q_td = self._make_multi_section_query("q1", "TD", n_sections=1, n_facts=3)
        q_not_td = self._make_multi_section_query("q2", "TD", n_sections=1, n_facts=2)
        candidates_td = identify_rebalance_candidates([q_td], "TD")
        candidates_not = identify_rebalance_candidates([q_not_td], "TD")
        assert "sg_test" in candidates_td
        assert "sg_test" not in candidates_not

    def test_empty_queries(self):
        assert identify_rebalance_candidates([], "BU") == []


def _make_subgraph(
    n_sections: int = 3,
    has_relations: bool = True,
    n_nodes: int = 5,
) -> Subgraph:
    """Build a minimal Subgraph for testing compute_subgraph_allocation."""
    nodes = []
    for i in range(n_nodes):
        section_key = f"structure_nodes/sec{i % n_sections}"
        nodes.append(
            SubgraphNode(
                node_id=f"content_nodes/n{i}",
                text=f"Node {i}",
                parent_section_key=section_key,
                parent_section_title=f"Section {i % n_sections}",
            )
        )
    relations = []
    if has_relations and n_nodes >= 2:
        relations.append(
            SubgraphRelation(
                from_id="content_nodes/n0",
                to_id="content_nodes/n1",
                relation_type="TEST_REL",
            )
        )
    return Subgraph(
        subgraph_id="test-sg-id",
        anchor_key="structure_nodes/sec0",
        anchor_title="Test Anchor",
        breadcrumb="Root > Test",
        nodes=nodes,
        core_node_ids=[f"content_nodes/n{i}" for i in range(n_nodes)],
        enriched_node_ids=[],
        relations=relations,
        content_count=n_nodes,
        relation_count=len(relations),
        tier="large",
        query_allocation=10,
    )


class TestComputeSubgraphAllocation:
    def test_exact_sum_matches_target(self):
        sg = _make_subgraph(n_sections=3, has_relations=True)
        for target in (1, 5, 10, 13, 17, 100):
            alloc = compute_subgraph_allocation(sg, target)
            assert sum(alloc.values()) == target, f"sum mismatch for target={target}"

    def test_hz_excluded_without_relations(self):
        sg = _make_subgraph(n_sections=3, has_relations=False)
        alloc = compute_subgraph_allocation(sg, 10)
        assert "HZ" not in alloc

    def test_hz_included_with_relations(self):
        sg = _make_subgraph(n_sections=3, has_relations=True)
        alloc = compute_subgraph_allocation(sg, 10)
        assert "HZ" in alloc
        assert alloc["HZ"] > 0

    def test_bu_excluded_with_fewer_than_3_sections(self):
        sg = _make_subgraph(n_sections=2, has_relations=False)
        alloc = compute_subgraph_allocation(sg, 10)
        assert "BU" not in alloc

    def test_bu_included_with_3_or_more_sections(self):
        sg = _make_subgraph(n_sections=3, has_relations=False)
        alloc = compute_subgraph_allocation(sg, 10)
        assert "BU" in alloc

    def test_cs_excluded_with_single_section(self):
        sg = _make_subgraph(n_sections=1, has_relations=False)
        alloc = compute_subgraph_allocation(sg, 10)
        assert "CS" not in alloc

    def test_cs_included_with_2_or_more_sections(self):
        sg = _make_subgraph(n_sections=2, has_relations=False)
        alloc = compute_subgraph_allocation(sg, 10)
        assert "CS" in alloc

    def test_nt_and_td_always_present(self):
        # Even for a minimal single-section, no-relation subgraph
        sg = _make_subgraph(n_sections=1, has_relations=False)
        alloc = compute_subgraph_allocation(sg, 10)
        assert "NT" in alloc
        assert "TD" in alloc

    def test_all_zeros_excluded_target_one(self):
        # Target=1 must still distribute exactly 1 query
        sg = _make_subgraph(n_sections=1, has_relations=False)
        alloc = compute_subgraph_allocation(sg, 1)
        assert sum(alloc.values()) == 1
        assert all(v >= 0 for v in alloc.values())
