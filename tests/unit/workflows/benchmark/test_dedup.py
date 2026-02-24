"""Unit tests for workflows/benchmark/dedup.py.

Covers: _cosine_similarity, _should_remove_first, dedup_queries.
"""

from __future__ import annotations

import pytest

from workflows.benchmark.dedup import _cosine_similarity, _should_remove_first, dedup_queries
from workflows.benchmark.models import BenchmarkQuery, ConfidenceScore, FactNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_confidence(overall: int = 70) -> ConfidenceScore:
    return ConfidenceScore(
        question_clarity=7,
        fact_node_completeness=7,
        fact_node_precision=7,
        grade_certainty=7,
        overall=overall,
    )


def make_fact_node(node_id: str = "content_nodes/n1") -> FactNode:
    return FactNode(node_id=node_id, grade=2, parent_section="s1")


def make_query(
    query_id: str = "q-001",
    confidence_overall: int = 70,
    num_fact_nodes: int = 1,
) -> BenchmarkQuery:
    fact_nodes = [make_fact_node(f"content_nodes/n{i}") for i in range(num_fact_nodes)]
    return BenchmarkQuery(
        query_id=query_id,
        query_text="What happened?",
        subgraph_id="sg-001",
        fact_nodes=fact_nodes,
        relations_used=[],
        confidence=make_confidence(overall=confidence_overall),
        confidence_concerns="",
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_identical_non_unit_vectors_return_one(self):
        v = [3.0, 4.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors_return_zero(self):
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_known_similarity(self):
        # [1, 1] and [1, 0]: cos = 1/sqrt(2) ≈ 0.7071
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / (2.0**0.5)
        assert _cosine_similarity(a, b) == pytest.approx(expected, rel=1e-5)

    def test_mismatched_lengths_raises(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            _cosine_similarity(a, b)

    def test_single_element_vectors(self):
        assert _cosine_similarity([5.0], [5.0]) == pytest.approx(1.0)
        assert _cosine_similarity([1.0], [-1.0]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# _should_remove_first
# ---------------------------------------------------------------------------


class TestShouldRemoveFirst:
    def test_lower_confidence_removes_first(self):
        q1 = make_query("q1", confidence_overall=60)
        q2 = make_query("q2", confidence_overall=80)
        assert _should_remove_first(q1, q2) is True

    def test_higher_confidence_keeps_first(self):
        q1 = make_query("q1", confidence_overall=90)
        q2 = make_query("q2", confidence_overall=70)
        assert _should_remove_first(q1, q2) is False

    def test_equal_confidence_fewer_facts_removes_first(self):
        q1 = make_query("q1", confidence_overall=70, num_fact_nodes=1)
        q2 = make_query("q2", confidence_overall=70, num_fact_nodes=3)
        assert _should_remove_first(q1, q2) is True

    def test_equal_confidence_more_facts_keeps_first(self):
        q1 = make_query("q1", confidence_overall=70, num_fact_nodes=3)
        q2 = make_query("q2", confidence_overall=70, num_fact_nodes=1)
        assert _should_remove_first(q1, q2) is False

    def test_equal_confidence_equal_facts_keeps_first(self):
        # Final tie-break: q2 wins (lower original index for q1), so do NOT remove q1
        q1 = make_query("q1", confidence_overall=70, num_fact_nodes=2)
        q2 = make_query("q2", confidence_overall=70, num_fact_nodes=2)
        assert _should_remove_first(q1, q2) is False

    def test_confidence_dominates_fact_count(self):
        # Even though q1 has more fact nodes, lower confidence means q1 is removed
        q1 = make_query("q1", confidence_overall=50, num_fact_nodes=5)
        q2 = make_query("q2", confidence_overall=80, num_fact_nodes=1)
        assert _should_remove_first(q1, q2) is True


# ---------------------------------------------------------------------------
# dedup_queries
# ---------------------------------------------------------------------------


class TestDedupQueries:
    def test_no_duplicates_all_kept(self):
        q1 = make_query("q1")
        q2 = make_query("q2")
        # Orthogonal embeddings: similarity = 0.0 (below threshold)
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        unique, removed = dedup_queries([q1, q2], embeddings)
        assert len(unique) == 2
        assert len(removed) == 0

    def test_identical_embeddings_one_removed(self):
        q1 = make_query("q1", confidence_overall=80)
        q2 = make_query("q2", confidence_overall=60)
        embedding = [1.0, 0.0, 0.0]
        unique, removed = dedup_queries([q1, q2], [embedding, embedding])
        assert len(unique) == 1
        assert len(removed) == 1
        # q2 has lower confidence → q2 is removed, q1 kept
        assert unique[0].query_id == "q1"
        kept_id, removed_id, sim = removed[0]
        assert kept_id == "q1"
        assert removed_id == "q2"
        assert sim == pytest.approx(1.0)

    def test_lower_confidence_removed_when_duplicate(self):
        q1 = make_query("q1", confidence_overall=50)
        q2 = make_query("q2", confidence_overall=90)
        embedding = [0.5, 0.5]
        unique, removed = dedup_queries([q1, q2], [embedding, embedding])
        assert len(unique) == 1
        # q1 has lower confidence → q1 removed
        assert unique[0].query_id == "q2"

    def test_mismatched_lengths_raises_value_error(self):
        q1 = make_query("q1")
        with pytest.raises(ValueError, match="equal length"):
            dedup_queries([q1], [[1.0], [0.0]])

    def test_empty_inputs(self):
        unique, removed = dedup_queries([], [])
        assert unique == []
        assert removed == []

    def test_single_query_never_removed(self):
        q = make_query("q1")
        unique, removed = dedup_queries([q], [[1.0, 0.0]])
        assert len(unique) == 1
        assert len(removed) == 0

    def test_below_threshold_not_deduplicated(self):
        q1 = make_query("q1")
        q2 = make_query("q2")
        # Similarity of 0.5, well below default threshold 0.92
        e1 = [1.0, 0.0]
        e2 = [0.5, 0.866]  # roughly 30 degrees apart
        unique, removed = dedup_queries([q1, q2], [e1, e2], threshold=0.92)
        assert len(unique) == 2
        assert len(removed) == 0

    def test_above_threshold_deduplicated(self):
        q1 = make_query("q1", confidence_overall=80)
        q2 = make_query("q2", confidence_overall=60)
        # Very similar embeddings (high cosine similarity)
        e1 = [1.0, 0.001]
        e2 = [1.0, 0.002]
        unique, removed = dedup_queries([q1, q2], [e1, e2], threshold=0.99)
        # similarity ≈ 0.9999... > 0.99
        assert len(unique) == 1
        assert len(removed) == 1

    def test_custom_threshold_exact_match(self):
        q1 = make_query("q1", confidence_overall=70)
        q2 = make_query("q2", confidence_overall=70)
        embedding = [1.0, 0.0]
        # Threshold exactly 1.0: similarity of 1.0 is NOT > 1.0
        unique, removed = dedup_queries([q1, q2], [embedding, embedding], threshold=1.0)
        # sim == threshold, NOT >, so no dedup
        assert len(unique) == 2
        assert len(removed) == 0

    def test_removed_pairs_contain_audit_info(self):
        q1 = make_query("q1", confidence_overall=80)
        q2 = make_query("q2", confidence_overall=60)
        embedding = [1.0, 0.0]
        _, removed = dedup_queries([q1, q2], [embedding, embedding])
        assert len(removed) == 1
        kept_id, removed_id, sim = removed[0]
        assert isinstance(kept_id, str)
        assert isinstance(removed_id, str)
        assert isinstance(sim, float)

    def test_three_queries_two_duplicates(self):
        """q1 and q2 are duplicates; q3 is unique."""
        q1 = make_query("q1", confidence_overall=80)
        q2 = make_query("q2", confidence_overall=60)
        q3 = make_query("q3", confidence_overall=75)
        # q1 and q2 are identical; q3 is orthogonal
        e1 = [1.0, 0.0]
        e2 = [1.0, 0.0]
        e3 = [0.0, 1.0]
        unique, removed = dedup_queries([q1, q2, q3], [e1, e2, e3])
        assert len(unique) == 2
        ids = {q.query_id for q in unique}
        assert "q1" in ids
        assert "q3" in ids

    def test_preserves_order_of_unique_queries(self):
        queries = [make_query(f"q{i}") for i in range(4)]
        # All orthogonal embeddings
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        unique, _ = dedup_queries(queries, embeddings)
        assert [q.query_id for q in unique] == ["q0", "q1", "q2", "q3"]
