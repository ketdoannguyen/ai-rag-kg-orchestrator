"""Tests for run_evaluation.py — 14-method evaluation runner."""

from __future__ import annotations

import pytest

from workflows.benchmark.models import BenchmarkQuery, ConfidenceScore, EvaluationResult, FactNode
from workflows.benchmark.run_evaluation import (
    METHOD_CONFIGS,
    aggregate_method_results,
    evaluate_all_methods_offline,
    evaluate_method_offline,
)


def _make_query(query_id: str, direction: str = "TD") -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        query_text="Test question?",
        subgraph_id="sg_test",
        fact_nodes=[
            FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a"),
            FactNode(node_id="content_nodes/n2", grade=0, parent_section="sec/b"),
        ],
        relations_used=[],
        confidence=ConfidenceScore(
            overall=80,
        ),
        reasoning="Test",
        direction=direction,
    )


def _make_eval(query_id: str, method_id: str, ndcg: float = 0.5) -> EvaluationResult:
    return EvaluationResult(
        query_id=query_id,
        method_id=method_id,
        method_name=f"Method {method_id}",
        retrieved_ids=["content_nodes/n1"],
        ndcg_at_10=ndcg,
        recall_at_10=ndcg,
        precision_at_10=ndcg,
        mrr=ndcg,
        result_count=1,
    )


class TestAggregateMethodResults:
    def test_empty_evaluations(self):
        result = aggregate_method_results("B1", "Vector-Only", [])
        assert result.method_id == "B1"
        assert result.avg_ndcg == 0.0
        assert result.avg_recall == 0.0
        assert result.query_count == 0

    def test_single_evaluation(self):
        evals = [_make_eval("q1", "B1", ndcg=0.8)]
        result = aggregate_method_results("B1", "Vector-Only", evals)
        assert result.avg_ndcg == pytest.approx(0.8)
        assert result.query_count == 1

    def test_average_computed_correctly(self):
        evals = [_make_eval("q1", "B1", ndcg=0.6), _make_eval("q2", "B1", ndcg=0.4)]
        result = aggregate_method_results("B1", "Vector-Only", evals)
        assert result.avg_ndcg == pytest.approx(0.5)
        assert result.query_count == 2

    def test_all_metrics_averaged(self):
        evals = [
            EvaluationResult(
                query_id="q1",
                method_id="B1",
                method_name="Vector-Only",
                retrieved_ids=[],
                ndcg_at_10=0.6,
                recall_at_10=0.7,
                precision_at_10=0.8,
                mrr=0.9,
                result_count=5,
            ),
            EvaluationResult(
                query_id="q2",
                method_id="B1",
                method_name="Vector-Only",
                retrieved_ids=[],
                ndcg_at_10=0.4,
                recall_at_10=0.3,
                precision_at_10=0.2,
                mrr=0.1,
                result_count=3,
            ),
        ]
        result = aggregate_method_results("B1", "Vector-Only", evals)
        assert result.avg_ndcg == pytest.approx(0.5)
        assert result.avg_recall == pytest.approx(0.5)
        assert result.avg_precision == pytest.approx(0.5)
        assert result.avg_mrr == pytest.approx(0.5)


class TestEvaluateMethodOffline:
    def test_basic_evaluation(self):
        queries = [_make_query("q1"), _make_query("q2")]
        results_map = {
            "q1": ["content_nodes/n1", "content_nodes/n2"],
            "q2": ["content_nodes/n1"],
        }
        mr = evaluate_method_offline("B1", "Vector-Only", queries, results_map)
        assert mr.query_count == 2
        assert 0.0 <= mr.avg_ndcg <= 1.0

    def test_missing_query_skipped(self):
        queries = [_make_query("q1"), _make_query("q2")]
        results_map = {"q1": ["content_nodes/n1"]}
        mr = evaluate_method_offline("B1", "Vector-Only", queries, results_map)
        assert mr.query_count == 1  # q2 missing from results_map

    def test_empty_results_map(self):
        queries = [_make_query("q1")]
        mr = evaluate_method_offline("B1", "Vector-Only", queries, {})
        assert mr.query_count == 0
        assert mr.avg_ndcg == 0.0

    def test_perfect_retrieval_ndcg_one(self):
        queries = [_make_query("q1")]
        # Grade-2 node retrieved first
        results_map = {"q1": ["content_nodes/n1", "content_nodes/n2"]}
        mr = evaluate_method_offline("B1", "Vector-Only", queries, results_map)
        assert mr.avg_ndcg == pytest.approx(1.0)


class TestEvaluateAllMethods:
    def test_all_14_configs_processed(self):
        queries = [_make_query("q1")]
        # Build results for all 14 methods
        all_results = {config["id"]: {"q1": ["content_nodes/n1"]} for config in METHOD_CONFIGS}
        summary = evaluate_all_methods_offline(queries, all_results)
        assert summary.total_methods == len(METHOD_CONFIGS)
        assert len(summary.method_results) == len(METHOD_CONFIGS)

    def test_method_ids_match_configs(self):
        queries = [_make_query("q1")]
        all_results = {config["id"]: {"q1": []} for config in METHOD_CONFIGS}
        summary = evaluate_all_methods_offline(queries, all_results)
        result_ids = {mr.method_id for mr in summary.method_results}
        config_ids = {c["id"] for c in METHOD_CONFIGS}
        assert result_ids == config_ids

    def test_to_dict_structure(self):
        queries = [_make_query("q1")]
        all_results = {config["id"]: {"q1": []} for config in METHOD_CONFIGS}
        summary = evaluate_all_methods_offline(queries, all_results)
        d = summary.to_dict()
        assert "total_methods" in d
        assert "methods" in d
        assert len(d["methods"]) == len(METHOD_CONFIGS)

    def test_empty_all_results(self):
        queries = [_make_query("q1")]
        summary = evaluate_all_methods_offline(queries, {})
        # All methods get 0 evaluations
        assert summary.total_evaluations == 0
        for mr in summary.method_results:
            assert mr.query_count == 0
