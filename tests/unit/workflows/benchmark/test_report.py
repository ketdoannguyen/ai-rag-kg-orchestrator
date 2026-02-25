"""Tests for report.py — evaluation report generator."""

from __future__ import annotations

from workflows.benchmark.models import BenchmarkQuery, ConfidenceScore, FactNode
from workflows.benchmark.report import (
    _sanity_checks,
    generate_report,
)
from workflows.benchmark.run_evaluation import EvaluationRunSummary, MethodRunResult
from workflows.benchmark.statistics import HypothesisResult, StatisticsReport


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


def _make_method_result(
    method_id: str,
    method_name: str,
    avg_ndcg: float = 0.5,
    query_count: int = 10,
) -> MethodRunResult:
    return MethodRunResult(
        method_id=method_id,
        method_name=method_name,
        evaluations=[],
        avg_ndcg=avg_ndcg,
        avg_recall=avg_ndcg,
        avg_precision=avg_ndcg,
        avg_mrr=avg_ndcg,
        query_count=query_count,
    )


def _make_summary(method_results: list[MethodRunResult] | None = None) -> EvaluationRunSummary:
    if method_results is None:
        method_results = [
            _make_method_result("B1", "Vector-Only", avg_ndcg=0.3),
            _make_method_result("E1a", "TopDown COSINE", avg_ndcg=0.5),
            _make_method_result("E3a", "BottomUp PARENT_COSINE", avg_ndcg=0.6),
        ]
    return EvaluationRunSummary(
        total_methods=len(method_results),
        total_queries=10,
        total_evaluations=len(method_results) * 10,
        method_results=method_results,
    )


def _make_stats() -> StatisticsReport:
    return StatisticsReport(
        friedman_statistic=12.5,
        friedman_p_value=0.002,
        friedman_significant=True,
        pairwise_results=[],
        hypothesis_results=[
            HypothesisResult(
                hypothesis_id="H1",
                description="Graph-enhanced > baseline",
                supported=True,
                evidence="B1 NDCG=0.30, E1a NDCG=0.50, delta=0.2" + "x" * 100,
            )
        ],
        per_direction={
            "TD": {"B1": 0.3, "E1a": 0.5},
        },
    )


class TestGenerateReport:
    def test_report_written_to_file(self, tmp_path):
        summary = _make_summary()
        stats = _make_stats()
        queries = [_make_query(f"q{i}", "TD") for i in range(10)]
        out = tmp_path / "report.md"
        result_path = generate_report(summary, stats, queries, str(out))
        assert out.exists()
        assert result_path == str(out)

    def test_report_contains_all_6_sections(self, tmp_path):
        summary = _make_summary()
        stats = _make_stats()
        queries = [_make_query(f"q{i}", "TD") for i in range(10)]
        out = tmp_path / "report.md"
        generate_report(summary, stats, queries, str(out))
        content = out.read_text(encoding="utf-8")
        assert "## 1. Dataset Statistics" in content
        assert "## 2. Method Performance" in content
        assert "## 3. Per-Direction Breakdown" in content
        assert "## 4. Statistical Significance" in content
        assert "## 5. Hypothesis Validation" in content
        assert "## 6. Sanity Checks" in content

    def test_creates_parent_dirs(self, tmp_path):
        summary = _make_summary()
        stats = _make_stats()
        queries = [_make_query("q1")]
        out = tmp_path / "sub" / "reports" / "report.md"
        generate_report(summary, stats, queries, str(out))
        assert out.exists()

    def test_direction_distribution_in_stats(self, tmp_path):
        summary = _make_summary()
        stats = _make_stats()
        queries = [_make_query(f"q{i}", "HZ") for i in range(5)]
        out = tmp_path / "report.md"
        generate_report(summary, stats, queries, str(out))
        content = out.read_text(encoding="utf-8")
        assert "HZ" in content


class TestSanityChecks:
    def test_b1_ndcg_greater_than_zero(self):
        summary = _make_summary(
            [
                _make_method_result("B1", "Vector-Only", avg_ndcg=0.3, query_count=10),
            ]
        )
        text = _sanity_checks(summary)
        assert "[PASS] B1 NDCG > 0" in text

    def test_b1_ndcg_zero_fails(self):
        summary = _make_summary(
            [
                _make_method_result("B1", "Vector-Only", avg_ndcg=0.0, query_count=10),
            ]
        )
        text = _sanity_checks(summary)
        assert "[FAIL] B1 NDCG > 0" in text

    def test_b1_less_than_best_e_method(self):
        summary = _make_summary(
            [
                _make_method_result("B1", "Vector-Only", avg_ndcg=0.3, query_count=10),
                _make_method_result("E1a", "TopDown", avg_ndcg=0.6, query_count=10),
            ]
        )
        text = _sanity_checks(summary)
        assert "[PASS] B1 < best E* method" in text

    def test_b1_exceeds_best_e_method_fails(self):
        summary = _make_summary(
            [
                _make_method_result("B1", "Vector-Only", avg_ndcg=0.8, query_count=10),
                _make_method_result("E1a", "TopDown", avg_ndcg=0.5, query_count=10),
            ]
        )
        text = _sanity_checks(summary)
        assert "[FAIL] B1 < best E* method" in text

    def test_all_methods_produce_results_pass(self):
        summary = _make_summary(
            [
                _make_method_result("B1", "Vector-Only", query_count=10),
                _make_method_result("E1a", "TopDown", query_count=5),
            ]
        )
        text = _sanity_checks(summary)
        assert "[PASS] All methods produce results" in text

    def test_zero_query_count_fails(self):
        summary = _make_summary(
            [
                _make_method_result("B1", "Vector-Only", query_count=0),
            ]
        )
        text = _sanity_checks(summary)
        assert "[FAIL] All methods produce results" in text
