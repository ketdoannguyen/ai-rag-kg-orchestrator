"""Tests for triage.py — confidence-based triage pipeline."""

from __future__ import annotations

from workflows.benchmark.models import BenchmarkQuery, ConfidenceScore, FactNode
from workflows.benchmark.triage import (
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    classify_tier,
    triage_queries,
)


def _make_query(query_id: str, confidence: int, direction: str = "TD") -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        query_text="Test question?",
        subgraph_id="sg_test",
        fact_nodes=[FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a")],
        relations_used=[],
        confidence=ConfidenceScore(
            overall=confidence,
        ),
        reasoning="Test",
        direction=direction,
    )


class TestClassifyTier:
    def test_high_at_threshold(self):
        assert classify_tier(HIGH_THRESHOLD) == "high"

    def test_high_above_threshold(self):
        assert classify_tier(100) == "high"

    def test_high_just_above(self):
        assert classify_tier(HIGH_THRESHOLD + 1) == "high"

    def test_medium_at_threshold(self):
        assert classify_tier(MEDIUM_THRESHOLD) == "medium"

    def test_medium_just_below_high(self):
        assert classify_tier(HIGH_THRESHOLD - 1) == "medium"

    def test_medium_between_thresholds(self):
        assert classify_tier(80) == "medium"

    def test_low_just_below_medium(self):
        assert classify_tier(MEDIUM_THRESHOLD - 1) == "low"

    def test_low_at_zero(self):
        assert classify_tier(0) == "low"

    def test_low_boundary(self):
        assert classify_tier(74) == "low"


class TestTriageQueries:
    def test_tier_assignment_counts(self):
        queries = (
            [_make_query(f"h{i}", 90) for i in range(10)]
            + [_make_query(f"m{i}", 80) for i in range(10)]
            + [_make_query(f"l{i}", 60) for i in range(10)]
        )
        result = triage_queries(queries, seed=42)
        assert result.tier_counts["high"] == 10
        assert result.tier_counts["medium"] == 10
        assert result.tier_counts["low"] == 10

    def test_sample_rates(self):
        # 100 high → 10% = 10, 20 medium → 50% = 10, 10 low → 100% = 10
        queries = (
            [_make_query(f"h{i}", 90) for i in range(100)]
            + [_make_query(f"m{i}", 80) for i in range(20)]
            + [_make_query(f"l{i}", 60) for i in range(10)]
        )
        result = triage_queries(queries, seed=42)
        assert result.tier_counts["high"] == 100
        assert len(result.tier_details["high"]) == 10
        assert len(result.tier_details["medium"]) == 10
        assert len(result.tier_details["low"]) == 10

    def test_deterministic_with_seed(self):
        queries = [_make_query(f"q{i}", 90) for i in range(50)]
        r1 = triage_queries(queries, seed=42)
        r2 = triage_queries(queries, seed=42)
        assert r1.tier_details["high"] == r2.tier_details["high"]

    def test_different_seeds_differ(self):
        queries = [_make_query(f"q{i}", 90) for i in range(50)]
        r1 = triage_queries(queries, seed=1)
        r2 = triage_queries(queries, seed=2)
        # With enough queries, different seeds should produce different samples
        assert r1.tier_details["high"] != r2.tier_details["high"]

    def test_review_count_matches_details(self):
        queries = [_make_query(f"q{i}", 80) for i in range(10)]
        result = triage_queries(queries, seed=42)
        total_detail = sum(len(ids) for ids in result.tier_details.values())
        assert result.review_count == total_detail

    def test_low_confidence_fully_reviewed(self):
        queries = [_make_query(f"l{i}", 50) for i in range(10)]
        result = triage_queries(queries, seed=42)
        # 100% review rate for low tier
        assert len(result.tier_details["low"]) == 10

    def test_to_dict_keys(self):
        queries = [_make_query("q1", 80)]
        result = triage_queries(queries)
        d = result.to_dict()
        assert "tier_counts" in d
        assert "review_count" in d
        assert "tier_details" in d


class TestTriageEmpty:
    def test_empty_queries(self):
        result = triage_queries([])
        assert result.tier_counts == {"high": 0, "medium": 0, "low": 0}
        assert result.review_count == 0
        assert result.review_queries == []

    def test_single_query_always_reviewed(self):
        queries = [_make_query("q1", 90)]
        result = triage_queries(queries)
        # max(1, int(1 * 0.10)) = max(1, 0) = 1 → always reviewed
        assert result.review_count == 1
