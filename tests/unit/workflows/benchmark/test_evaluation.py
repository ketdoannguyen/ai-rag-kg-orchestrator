"""Unit tests for workflows/benchmark/evaluation.py.

Covers: ndcg_at_k, recall_at_k, precision_at_k, mrr,
build_relevance_map, evaluate_single, evaluate_batch.
"""

from __future__ import annotations

import pytest

from workflows.benchmark.evaluation import (
    build_relevance_map,
    evaluate_batch,
    evaluate_single,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from workflows.benchmark.models import (
    BenchmarkQuery,
    ConfidenceScore,
    EvaluationResult,
    FactNode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_confidence(**overrides) -> ConfidenceScore:
    defaults = {
        "overall": 75,
    }
    defaults.update(overrides)
    return ConfidenceScore(**defaults)


def make_fact_node(
    node_id: str = "content_nodes/abc123",
    grade: int = 2,
    parent_section: str = "s1",
) -> FactNode:
    return FactNode(node_id=node_id, grade=grade, parent_section=parent_section)


def make_benchmark_query(
    query_id: str = "q1",
    fact_nodes: list[FactNode] | None = None,
    **overrides,
) -> BenchmarkQuery:
    if fact_nodes is None:
        fact_nodes = [
            make_fact_node("content_nodes/a", grade=2, parent_section="s1"),
            make_fact_node("content_nodes/b", grade=1, parent_section="s1"),
            make_fact_node("content_nodes/c", grade=0, parent_section="s2"),
        ]
    defaults = {
        "query_id": query_id,
        "query_text": "Test query text",
        "subgraph_id": "sg-001",
        "fact_nodes": fact_nodes,
        "relations_used": [],
        "confidence": make_confidence(),
        "reasoning": "test",
    }
    defaults.update(overrides)
    return BenchmarkQuery(**defaults)


# ---------------------------------------------------------------------------
# Tests: ndcg_at_k
# ---------------------------------------------------------------------------


class TestNdcgAtK:
    def test_perfect_ranking(self):
        """All relevant docs retrieved in ideal order."""
        relevance = {"a": 2, "b": 1, "c": 0}
        retrieved = ["a", "b", "c"]
        result = ndcg_at_k(retrieved, relevance, k=3)
        assert result == pytest.approx(1.0)

    def test_reversed_ranking(self):
        """Relevant docs in reverse order — NDCG < 1."""
        relevance = {"a": 2, "b": 1}
        retrieved = ["b", "a"]  # suboptimal order
        result = ndcg_at_k(retrieved, relevance, k=2)
        assert 0.0 < result < 1.0

    def test_no_relevant_docs(self):
        """All grades are 0 → IDCG=0 → return 0."""
        relevance = {"a": 0, "b": 0}
        retrieved = ["a", "b"]
        assert ndcg_at_k(retrieved, relevance, k=2) == 0.0

    def test_empty_retrieved(self):
        relevance = {"a": 2}
        assert ndcg_at_k([], relevance, k=10) == 0.0

    def test_empty_relevance(self):
        assert ndcg_at_k(["a", "b"], {}, k=10) == 0.0

    def test_k_larger_than_retrieved(self):
        relevance = {"a": 2, "b": 1}
        retrieved = ["a"]
        result = ndcg_at_k(retrieved, relevance, k=10)
        assert 0.0 < result < 1.0

    def test_unknown_doc_treated_as_grade_0(self):
        relevance = {"a": 2}
        retrieved = ["x", "a"]  # x not in relevance → grade 0
        result = ndcg_at_k(retrieved, relevance, k=2)
        assert 0.0 < result < 1.0

    def test_single_relevant_at_rank_1(self):
        """One grade-2 doc at rank 1 → perfect NDCG."""
        relevance = {"a": 2}
        retrieved = ["a"]
        assert ndcg_at_k(retrieved, relevance, k=1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self):
        relevance = {"a": 2, "b": 1}
        retrieved = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)

    def test_partial_recall(self):
        relevance = {"a": 2, "b": 1}
        retrieved = ["a", "c"]
        assert recall_at_k(retrieved, relevance, k=2) == pytest.approx(0.5)

    def test_zero_recall(self):
        relevance = {"a": 2, "b": 1}
        retrieved = ["c", "d"]
        assert recall_at_k(retrieved, relevance, k=2) == pytest.approx(0.0)

    def test_no_relevant_docs(self):
        relevance = {"a": 0}
        retrieved = ["a"]
        assert recall_at_k(retrieved, relevance, k=1) == 0.0

    def test_empty_retrieved(self):
        relevance = {"a": 2}
        assert recall_at_k([], relevance, k=10) == 0.0

    def test_grade_0_not_counted_as_relevant(self):
        relevance = {"a": 0, "b": 2}
        retrieved = ["a"]  # only grade-0 found
        assert recall_at_k(retrieved, relevance, k=1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self):
        relevance = {"a": 2, "b": 1}
        retrieved = ["a", "b"]
        assert precision_at_k(retrieved, relevance, k=2) == pytest.approx(1.0)

    def test_half_precision(self):
        relevance = {"a": 2}
        retrieved = ["a", "c"]
        assert precision_at_k(retrieved, relevance, k=2) == pytest.approx(0.5)

    def test_zero_precision(self):
        relevance = {"a": 2}
        retrieved = ["c", "d"]
        assert precision_at_k(retrieved, relevance, k=2) == pytest.approx(0.0)

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a": 2}, k=0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a": 2}, k=10) == 0.0


# ---------------------------------------------------------------------------
# Tests: mrr
# ---------------------------------------------------------------------------


class TestMrr:
    def test_first_result_relevant(self):
        relevance = {"a": 2}
        assert mrr(["a", "b"], relevance) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        relevance = {"b": 1}
        assert mrr(["a", "b"], relevance) == pytest.approx(0.5)

    def test_no_relevant_result(self):
        relevance = {"x": 2}
        assert mrr(["a", "b"], relevance) == 0.0

    def test_grade_0_not_relevant(self):
        relevance = {"a": 0, "b": 2}
        assert mrr(["a", "b"], relevance) == pytest.approx(0.5)

    def test_empty_retrieved(self):
        assert mrr([], {"a": 2}) == 0.0


# ---------------------------------------------------------------------------
# Tests: build_relevance_map
# ---------------------------------------------------------------------------


class TestBuildRelevanceMap:
    def test_basic(self):
        query = make_benchmark_query(
            fact_nodes=[
                make_fact_node("content_nodes/a", grade=2),
                make_fact_node("content_nodes/b", grade=1),
                make_fact_node("content_nodes/c", grade=0),
            ]
        )
        rel = build_relevance_map(query)
        assert rel == {
            "content_nodes/a": 2,
            "content_nodes/b": 1,
            "content_nodes/c": 0,
        }


# ---------------------------------------------------------------------------
# Tests: evaluate_single
# ---------------------------------------------------------------------------


class TestEvaluateSingle:
    def test_returns_evaluation_result(self):
        query = make_benchmark_query(fact_nodes=[make_fact_node("content_nodes/a", grade=2)])
        result = evaluate_single(
            query, "B1", "Vector-Only", ["content_nodes/a", "content_nodes/x"]
        )
        assert isinstance(result, EvaluationResult)
        assert result.query_id == query.query_id
        assert result.method_id == "B1"
        assert result.ndcg_at_10 == pytest.approx(1.0)
        assert result.recall_at_10 == pytest.approx(1.0)
        assert result.mrr == pytest.approx(1.0)
        assert result.result_count == 2


# ---------------------------------------------------------------------------
# Tests: evaluate_batch
# ---------------------------------------------------------------------------


class TestEvaluateBatch:
    def test_batch_basic(self):
        q1 = make_benchmark_query(
            query_id="q1",
            fact_nodes=[make_fact_node("content_nodes/a", grade=2)],
        )
        q2 = make_benchmark_query(
            query_id="q2",
            fact_nodes=[make_fact_node("content_nodes/b", grade=1)],
        )
        results_map = {
            "q1": ["content_nodes/a"],
            "q2": ["content_nodes/x", "content_nodes/b"],
        }
        results = evaluate_batch([q1, q2], "B1", "Vector-Only", results_map)
        assert len(results) == 2
        assert results[0].query_id == "q1"
        assert results[1].query_id == "q2"

    def test_missing_query_skipped(self):
        q1 = make_benchmark_query(query_id="q1")
        results_map = {"q_other": ["a"]}
        results = evaluate_batch([q1], "B1", "Vector-Only", results_map)
        assert len(results) == 0
