"""Tests for qa_checks.py — per-query checks and aggregate validation."""

from __future__ import annotations

import pytest

from workflows.benchmark.models import (
    BenchmarkQuery,
    ConfidenceScore,
    FactNode,
)
from workflows.benchmark.qa_checks import (
    _cosine_similarity,
    check_confidence_completeness,
    check_grade0_presence,
    check_node_existence,
    run_qa_checks,
)


def _make_query(
    query_id: str = "q1",
    fact_nodes: list[FactNode] | None = None,
    overall_confidence: int = 85,
    direction: str | None = "TD",
    confidence_concerns: str = "",
) -> BenchmarkQuery:
    if fact_nodes is None:
        fact_nodes = [
            FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a"),
            FactNode(node_id="content_nodes/n2", grade=0, parent_section="sec/b"),
        ]

    return BenchmarkQuery(
        query_id=query_id,
        query_text="Test query?",
        subgraph_id="sg_test",
        fact_nodes=fact_nodes,
        relations_used=[],
        confidence=ConfidenceScore(
            question_clarity=8,
            fact_node_completeness=8,
            fact_node_precision=8,
            grade_certainty=8,
            overall=overall_confidence,
        ),
        confidence_concerns=confidence_concerns,
        reasoning="Test",
        direction=direction,
    )


class TestCheckNodeExistence:
    def test_all_valid(self):
        q = _make_query()
        valid_ids = {"content_nodes/n1", "content_nodes/n2"}
        result = check_node_existence(q, valid_ids)
        assert result.passed
        assert result.action == "PASS"

    def test_invalid_node(self):
        q = _make_query()
        valid_ids = {"content_nodes/n1"}  # n2 missing
        result = check_node_existence(q, valid_ids)
        assert not result.passed
        assert result.action == "REMOVE"


class TestCheckGrade0Presence:
    def test_has_grade0(self):
        q = _make_query()
        result = check_grade0_presence(q)
        assert result.passed

    def test_no_grade0(self):
        q = _make_query(
            fact_nodes=[
                FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a"),
            ]
        )
        result = check_grade0_presence(q)
        assert not result.passed
        assert result.action == "FLAG"


class TestCheckConfidenceCompleteness:
    def test_valid_confidence(self):
        q = _make_query(overall_confidence=85)
        result = check_confidence_completeness(q)
        assert result.passed

    # Note: confidence -1 would fail model validation, so we test boundary
    def test_zero_confidence_passes(self):
        q = _make_query(overall_confidence=0)
        result = check_confidence_completeness(q)
        assert result.passed


class TestRunQAChecks:
    def test_clean_query_passes(self):
        q = _make_query()
        clean, flagged, report = run_qa_checks([q])
        assert len(clean) == 1
        assert len(flagged) == 0
        assert report.total_removed == 0

    def test_invalid_node_removed(self):
        q = _make_query()
        valid_ids = {"content_nodes/n1"}  # n2 not in set
        clean, flagged, report = run_qa_checks([q], valid_node_ids=valid_ids)
        assert len(clean) == 0
        assert report.total_removed == 1

    def test_multiple_queries(self):
        q1 = _make_query("q1")  # Should pass
        q2 = _make_query(
            "q2",
            fact_nodes=[
                FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a"),
            ],
        )  # No grade-0 → flagged

        clean, flagged, report = run_qa_checks([q1, q2])
        assert len(clean) == 1
        assert len(flagged) == 1
        assert report.total_flagged == 1


class TestDedupAfterRemoval:
    """Regression tests: dedup must work even after earlier checks remove queries."""

    def test_dedup_runs_after_earlier_removal(self):
        """When check 1 removes a query, dedup should still run on survivors."""
        # q1 has an extra invalid node → removed by check 1
        q1 = _make_query(
            "q1",
            fact_nodes=[
                FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a"),
                FactNode(node_id="content_nodes/n2", grade=0, parent_section="sec/b"),
                FactNode(node_id="content_nodes/bad", grade=1, parent_section="sec/c"),
            ],
        )
        # q2 and q3 only have valid nodes → survive check 1, near-duplicates
        q2 = _make_query("q2")
        q3 = _make_query("q3")

        valid_ids = {"content_nodes/n1", "content_nodes/n2"}  # "bad" not valid
        # Embeddings aligned to original queries order [q1, q2, q3]
        embeddings = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
        clean, flagged, report = run_qa_checks(
            [q1, q2, q3],
            valid_node_ids=valid_ids,
            dedup_embeddings=embeddings,
            dedup_threshold=0.5,
        )
        assert "q1" in report.removed_query_ids
        # q2 and q3 are near-duplicates → one should be deduped
        assert "dedup" in report.check_stats
        assert report.check_stats["dedup"]["failed"] >= 1

    def test_dedup_removes_flagged_query_from_report(self):
        """Flagged queries removed by dedup should not appear in flagged output."""
        # q1: no grade-0 → flagged; q2: no grade-0 → flagged; near-duplicates
        q1 = _make_query(
            "q1",
            fact_nodes=[FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a")],
        )
        q2 = _make_query(
            "q2",
            fact_nodes=[FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a")],
        )
        embeddings = [[1.0, 0.0], [1.0, 0.0]]
        clean, flagged, report = run_qa_checks(
            [q1, q2],
            dedup_embeddings=embeddings,
            dedup_threshold=0.5,
        )
        # One deduped away → only 1 flagged
        assert len(flagged) == 1
        assert report.total_flagged == 1


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0
