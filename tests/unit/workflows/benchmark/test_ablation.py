"""Tests for ablation.py — E3 ablation study."""

from __future__ import annotations

import pytest

from workflows.benchmark.ablation import (
    E3_CONTENT_AWARE,
    E3_PARENT_LEVEL,
    AblationReport,
    analyze_precision_recall_tradeoff,
    build_ablation_entries,
    run_ablation_study,
)
from workflows.benchmark.ablation import (
    test_h7 as run_h7_test,
)
from workflows.benchmark.models import EvaluationResult
from workflows.benchmark.run_evaluation import MethodRunResult


def _make_method_result(
    method_id: str,
    avg_ndcg: float = 0.5,
    avg_recall: float = 0.5,
    avg_precision: float = 0.5,
    evaluations: list[EvaluationResult] | None = None,
) -> MethodRunResult:
    if evaluations is None:
        evaluations = []
    return MethodRunResult(
        method_id=method_id,
        method_name=f"Method {method_id}",
        evaluations=evaluations,
        avg_ndcg=avg_ndcg,
        avg_recall=avg_recall,
        avg_precision=avg_precision,
        avg_mrr=avg_ndcg,
        query_count=len(evaluations),
    )


def _make_eval(query_id: str, method_id: str, ndcg: float = 0.5) -> EvaluationResult:
    return EvaluationResult(
        query_id=query_id,
        method_id=method_id,
        method_name=f"Method {method_id}",
        retrieved_ids=[],
        ndcg_at_10=ndcg,
        recall_at_10=ndcg,
        precision_at_10=ndcg,
        mrr=ndcg,
        result_count=5,
    )


def _make_all_e3_results(e3d_ndcg: float = 0.7, e3b_ndcg: float = 0.5) -> list[MethodRunResult]:
    """Create method results for all E3 variants."""
    e3d_evals = [_make_eval(f"q{i}", "E3d", ndcg=e3d_ndcg) for i in range(5)]
    e3b_evals = [_make_eval(f"q{i}", "E3b", ndcg=e3b_ndcg) for i in range(5)]
    return [
        _make_method_result("E3a", evaluations=[]),
        _make_method_result("E3b", avg_ndcg=e3b_ndcg, evaluations=e3b_evals),
        _make_method_result("E3c", evaluations=[]),
        _make_method_result("E3d", avg_ndcg=e3d_ndcg, evaluations=e3d_evals),
        _make_method_result("E3e", evaluations=[]),
    ]


class TestBuildAblationEntries:
    def test_filters_to_e3_only(self):
        results = [
            _make_method_result("B1"),
            _make_method_result("E1a"),
            _make_method_result("E3a"),
            _make_method_result("E3b"),
        ]
        entries = build_ablation_entries(results)
        ids = [e.method_id for e in entries]
        assert "E3a" in ids
        assert "E3b" in ids
        assert "B1" not in ids
        assert "E1a" not in ids

    def test_sorted_by_method_id(self):
        results = [_make_method_result(mid) for mid in ["E3e", "E3a", "E3c", "E3b", "E3d"]]
        entries = build_ablation_entries(results)
        ids = [e.method_id for e in entries]
        assert ids == sorted(ids)

    def test_avg_result_count_computed(self):
        evals = [
            EvaluationResult(
                query_id=f"q{i}",
                method_id="E3a",
                method_name="E3a",
                retrieved_ids=[],
                ndcg_at_10=0.5,
                recall_at_10=0.5,
                precision_at_10=0.5,
                mrr=0.5,
                result_count=10,
            )
            for i in range(3)
        ]
        results = [_make_method_result("E3a", evaluations=evals)]
        entries = build_ablation_entries(results)
        assert entries[0].avg_result_count == pytest.approx(10.0)

    def test_empty_input(self):
        assert build_ablation_entries([]) == []

    def test_non_e3_only_input(self):
        results = [_make_method_result("B1"), _make_method_result("E1a")]
        assert build_ablation_entries(results) == []


class TestH7:
    def test_supported_when_e3d_better(self):
        # E3d NDCG clearly better than E3b (delta > 0.15)
        results = _make_all_e3_results(e3d_ndcg=0.9, e3b_ndcg=0.1)
        h7 = run_h7_test(results)
        assert h7["supported"] is True
        assert h7["cliff_delta"] > 0

    def test_not_supported_when_e3d_worse(self):
        results = _make_all_e3_results(e3d_ndcg=0.1, e3b_ndcg=0.9)
        h7 = run_h7_test(results)
        assert h7["supported"] is False

    def test_not_supported_when_methods_missing(self):
        results = [_make_method_result("E3a")]
        h7 = run_h7_test(results)
        assert h7["supported"] is False
        assert "not found" in h7["evidence"]

    def test_no_overlap_returns_false(self):
        # E3d and E3b have no common query IDs
        e3d_evals = [_make_eval("q1", "E3d", 0.8)]
        e3b_evals = [_make_eval("q2", "E3b", 0.4)]
        results = [
            _make_method_result("E3d", avg_ndcg=0.8, evaluations=e3d_evals),
            _make_method_result("E3b", avg_ndcg=0.4, evaluations=e3b_evals),
        ]
        h7 = run_h7_test(results)
        assert h7["supported"] is False

    def test_result_contains_required_keys(self):
        results = _make_all_e3_results()
        h7 = run_h7_test(results)
        assert "supported" in h7
        assert "evidence" in h7
        assert "cliff_delta" in h7
        assert "effect_size" in h7


class TestPrecisionRecallTradeoff:
    def test_correct_grouping(self):
        results = [
            _make_method_result("E3a"),  # parent_level
            _make_method_result("E3b"),  # parent_level
            _make_method_result("E3c"),  # parent_level
            _make_method_result("E3d"),  # content_aware
            _make_method_result("E3e"),  # content_aware
        ]
        tradeoff = analyze_precision_recall_tradeoff(results)
        assert set(tradeoff["content_aware"].keys()) == set(E3_CONTENT_AWARE)
        assert set(tradeoff["parent_level"].keys()) == set(E3_PARENT_LEVEL)

    def test_interpretation_present(self):
        results = [_make_method_result("E3a")]
        tradeoff = analyze_precision_recall_tradeoff(results)
        assert "interpretation" in tradeoff
        assert len(tradeoff["interpretation"]) > 10

    def test_non_e3_excluded(self):
        results = [_make_method_result("B1"), _make_method_result("E1a")]
        tradeoff = analyze_precision_recall_tradeoff(results)
        assert tradeoff["content_aware"] == {}
        assert tradeoff["parent_level"] == {}


class TestRunAblationStudy:
    def test_full_pipeline_returns_report(self):
        results = _make_all_e3_results(e3d_ndcg=0.7, e3b_ndcg=0.5)
        report = run_ablation_study(results)
        assert isinstance(report, AblationReport)
        assert len(report.entries) > 0
        assert "supported" in report.h7_result
        assert "content_aware" in report.precision_recall_tradeoff

    def test_to_dict_structure(self):
        results = _make_all_e3_results()
        report = run_ablation_study(results)
        d = report.to_dict()
        assert "e3_comparison" in d
        assert "h7" in d
        assert "precision_recall_tradeoff" in d

    def test_empty_method_results(self):
        report = run_ablation_study([])
        assert report.entries == []
        assert report.h7_result["supported"] is False
