"""Tests for iaa.py — Cohen's kappa and IAA computation."""

from __future__ import annotations

import pytest

from workflows.benchmark.iaa import (
    cohens_kappa,
    compute_iaa,
    select_iaa_sample,
)
from workflows.benchmark.models import (
    BenchmarkQuery,
    ConfidenceScore,
    FactNode,
)


def _make_query(query_id: str, direction: str = "TD") -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        query_text="Test question?",
        subgraph_id="sg_test",
        fact_nodes=[FactNode(node_id="content_nodes/n1", grade=2, parent_section="sec/a")],
        relations_used=[],
        confidence=ConfidenceScore(
            overall=80,
        ),
        reasoning="Test",
        direction=direction,
    )


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = ["yes", "no", "yes", "no"]
        b = ["yes", "no", "yes", "no"]
        kappa, p_o, p_e = cohens_kappa(a, b)
        assert kappa == 1.0
        assert p_o == 1.0

    def test_no_agreement_disjoint_labels(self):
        # Completely disjoint labels: p_e = 0, p_o = 0, so kappa = 0
        a = ["yes", "yes", "yes", "yes"]
        b = ["no", "no", "no", "no"]
        kappa, p_o, p_e = cohens_kappa(a, b)
        assert kappa == 0.0
        assert p_o == 0.0

    def test_worse_than_chance(self):
        # Same label set but systematic disagreement
        a = ["yes", "no", "yes", "no"]
        b = ["no", "yes", "no", "yes"]
        kappa, p_o, p_e = cohens_kappa(a, b)
        assert kappa < 0  # Worse than chance
        assert p_o == 0.0

    def test_moderate_agreement(self):
        a = ["yes", "no", "yes", "no", "yes"]
        b = ["yes", "no", "no", "no", "yes"]
        kappa, _, _ = cohens_kappa(a, b)
        assert 0.0 < kappa < 1.0

    def test_empty_lists(self):
        kappa, p_o, p_e = cohens_kappa([], [])
        assert kappa == 0.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            cohens_kappa(["a"], ["a", "b"])


class TestComputeIAA:
    def test_basic_agreement(self):
        ids = ["q1", "q2", "q3", "q4"]
        labels_a = {"q1": "approved", "q2": "approved", "q3": "rejected", "q4": "rejected"}
        labels_b = {"q1": "approved", "q2": "approved", "q3": "rejected", "q4": "approved"}

        report = compute_iaa(ids, labels_a, labels_b)
        assert report.sample_size == 4
        assert report.observed_agreement == 0.75
        assert len(report.disagreements) == 1
        assert report.disagreements[0]["query_id"] == "q4"

    def test_missing_labels_skipped(self):
        ids = ["q1", "q2", "q3"]
        labels_a = {"q1": "approved", "q2": "approved"}
        labels_b = {"q1": "approved", "q3": "rejected"}

        report = compute_iaa(ids, labels_a, labels_b)
        # Only q1 has both labels
        assert report.sample_size == 1


class TestSelectIAASample:
    def test_sample_size(self):
        queries = [_make_query(f"q{i}", "TD" if i < 5 else "HZ") for i in range(20)]
        sample = select_iaa_sample(queries, sample_size=10, seed=42)
        assert len(sample) == 10

    def test_deterministic(self):
        queries = [_make_query(f"q{i}", "TD" if i < 5 else "HZ") for i in range(20)]
        s1 = select_iaa_sample(queries, sample_size=5, seed=42)
        s2 = select_iaa_sample(queries, sample_size=5, seed=42)
        assert [q.query_id for q in s1] == [q.query_id for q in s2]

    def test_empty_input(self):
        sample = select_iaa_sample([], sample_size=10)
        assert sample == []
