"""Tests for adaptive top-K selection."""

from __future__ import annotations

from workflows.query_retrieval.relation_scorer import (
    MAX_K,
    MIN_K,
    QUERY_TYPE_BASE_K,
    compute_adaptive_k,
)


class TestAdaptiveTopK:
    """Tests for adaptive top-K selection."""

    def test_query_type_based_k(self):
        """Test that query type determines base K."""
        # No gap in scores, should use base K
        scores = [0.9, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78] * 3

        k_structural = compute_adaptive_k(scores, "structural")
        k_lesson = compute_adaptive_k(scores, "lesson_synthesis")

        assert k_structural == 3  # structural has lowest K
        assert k_lesson == 12  # lesson_synthesis has highest K

    def test_gap_based_cutoff(self):
        """Test gap detection cuts at correct position."""
        # Clear gap at position 4
        scores = [0.90, 0.88, 0.85, 0.82, 0.50, 0.48, 0.45]

        k = compute_adaptive_k(scores, "causal", gap_threshold=0.1)

        # Gap is 0.82 - 0.50 = 0.32 > 0.1, so cut at position 4
        assert k == 4

    def test_gap_respects_base_k(self):
        """Test that K doesn't exceed base_k even without gap."""
        # No gap, but structural has base_k=3
        scores = [0.9, 0.88, 0.86, 0.84, 0.82, 0.80]

        k = compute_adaptive_k(scores, "structural")
        assert k == 3  # Limited by base_k

    def test_gap_can_reduce_below_base_k(self):
        """Test that gap can reduce K below base_k but respects min_k."""
        # Gap at position 3 (algorithm starts checking from index 2)
        scores = [0.90, 0.88, 0.85, 0.40, 0.38, 0.35]

        k = compute_adaptive_k(scores, "causal")  # base_k = 8
        # Gap at index 2-3 (0.85 - 0.40 = 0.45), so k = 3
        assert k == MIN_K  # Gap suggests 3, which equals min_k

    def test_min_k_enforced(self):
        """Test minimum K is enforced."""
        # Very early gap
        scores = [0.9, 0.3, 0.2, 0.1]

        k = compute_adaptive_k(scores, "structural")
        assert k >= MIN_K

    def test_max_k_enforced(self):
        """Test maximum K is enforced."""
        # No gap, many high scores
        scores = [0.9] * 100

        k = compute_adaptive_k(scores, "lesson_synthesis")
        assert k <= MAX_K

    def test_empty_scores(self):
        """Test with empty scores returns min_k."""
        k = compute_adaptive_k([], "causal")
        assert k == MIN_K

    def test_single_score(self):
        """Test with single score."""
        k = compute_adaptive_k([0.9], "causal")
        assert k == MIN_K

    def test_two_scores(self):
        """Test with two scores."""
        k = compute_adaptive_k([0.9, 0.8], "causal")
        assert k == MIN_K

    def test_unknown_query_type(self):
        """Test unknown query type uses default."""
        scores = [0.9] * 20

        k = compute_adaptive_k(scores, "unknown_type")
        # Should use default K from dict (7)
        assert k == 7

    def test_multiple_gaps_uses_first(self):
        """Test that first significant gap is used."""
        # Two gaps: at position 3 and 5
        scores = [0.95, 0.92, 0.90, 0.50, 0.48, 0.20, 0.18]

        k = compute_adaptive_k(scores, "causal", gap_threshold=0.1)
        # Should stop at first gap (position 3)
        assert k == 3

    def test_gap_above_threshold(self):
        """Test gap above threshold triggers cutoff."""
        scores = [0.9, 0.85, 0.80, 0.60]  # Gap = 0.20 > 0.10

        k = compute_adaptive_k(scores, "factual", gap_threshold=0.1)
        # Gap at position 2 (0.80 - 0.60 = 0.20), so k = 3
        assert k == 3

    def test_all_query_types_have_base_k(self):
        """Test all expected query types are in QUERY_TYPE_BASE_K."""
        expected_types = [
            "factual",
            "causal",
            "comparison",
            "figure",
            "concept",
            "lesson_synthesis",
            "structural",
            "other",
        ]
        for qt in expected_types:
            assert qt in QUERY_TYPE_BASE_K

    def test_base_k_values_reasonable(self):
        """Test base K values are within bounds."""
        for query_type, base_k in QUERY_TYPE_BASE_K.items():
            assert MIN_K <= base_k <= MAX_K, f"{query_type} has invalid base_k={base_k}"

    def test_no_gap_uses_base_k(self):
        """Test when no gap found, uses base_k."""
        # Uniform scores with no significant gap
        scores = [0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83]

        k = compute_adaptive_k(scores, "factual")  # base_k = 5
        assert k == 5
