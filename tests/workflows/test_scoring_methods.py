"""Unit tests for pluggable scoring methods.

Tests cover:
- RRF utility functions (compute_ranks, rrf_fusion_2way, rrf_fusion_3way)
- Scoring method enums
- Backward compatibility (default values)
"""

from __future__ import annotations

import numpy as np
import pytest

from workflows.query_retrieval.strategies.base import (
    compute_ranks_descending,
    compute_ranks_with_ties,
    rrf_fusion_2way,
    rrf_fusion_3way,
)
from workflows.query_retrieval.strategies.scoring import (
    RRF_K,
    E1ScoringMethod,
    E2ScoringMethod,
    E3ScoringMethod,
)

# =============================================================================
# Test RRF Constants
# =============================================================================


class TestRRFConstants:
    """Test RRF constants and enums."""

    def test_rrf_k_is_60(self) -> None:
        """RRF_K should be 60 (literature standard)."""
        assert RRF_K == 60

    def test_e1_scoring_methods(self) -> None:
        """E1 should have 2 scoring methods."""
        assert len(E1ScoringMethod) == 2
        assert E1ScoringMethod.COSINE.value == "cosine"
        assert E1ScoringMethod.RRF_CONTENT_SEED.value == "rrf_content_seed"

    def test_e2_scoring_methods(self) -> None:
        """E2 should have 4 scoring methods."""
        assert len(E2ScoringMethod) == 4
        assert E2ScoringMethod.COSINE.value == "cosine"
        assert E2ScoringMethod.RRF_COSINE_SEED.value == "rrf_cosine_seed"
        assert E2ScoringMethod.RRF_COSINE_RELATION.value == "rrf_cosine_relation"
        assert E2ScoringMethod.RRF_COSINE_SEED_RELATION.value == "rrf_cosine_seed_relation"

    def test_e3_scoring_methods(self) -> None:
        """E3 should have 5 scoring methods."""
        assert len(E3ScoringMethod) == 5
        assert E3ScoringMethod.PARENT_COSINE.value == "parent_cosine"
        assert E3ScoringMethod.AVG_LOG.value == "avg_log"
        assert E3ScoringMethod.RRF_PARENT_AVGLOG.value == "rrf_parent_avglog"
        assert E3ScoringMethod.FILTERED_AVG_LOG.value == "filtered_avg_log"
        assert E3ScoringMethod.RRF_PARENT_FILTERED.value == "rrf_parent_filtered"


# =============================================================================
# Test Rank Functions
# =============================================================================


class TestRankFunctions:
    """Test rank computation functions."""

    def test_compute_ranks_descending_basic(self) -> None:
        """Basic descending rank computation."""
        scores = np.array([0.9, 0.5, 0.7, 0.3], dtype=np.float32)
        ranks = compute_ranks_descending(scores)
        # 0.9 -> rank 1, 0.7 -> rank 2, 0.5 -> rank 3, 0.3 -> rank 4
        expected = np.array([1, 3, 2, 4], dtype=np.float32)
        np.testing.assert_array_equal(ranks, expected)

    def test_compute_ranks_descending_empty(self) -> None:
        """Empty array should return empty ranks."""
        scores = np.array([], dtype=np.float32)
        ranks = compute_ranks_descending(scores)
        assert len(ranks) == 0

    def test_compute_ranks_with_ties_basic(self) -> None:
        """Tied scores should get average rank."""
        scores = np.array([0.8, 0.6, 0.8, 0.6], dtype=np.float32)
        ranks = compute_ranks_with_ties(scores)
        # 0.8s share ranks 1-2 -> avg 1.5
        # 0.6s share ranks 3-4 -> avg 3.5
        expected = np.array([1.5, 3.5, 1.5, 3.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(ranks, expected)

    def test_compute_ranks_with_ties_no_ties(self) -> None:
        """Without ties, should behave like descending."""
        scores = np.array([0.9, 0.5, 0.7], dtype=np.float32)
        ranks = compute_ranks_with_ties(scores)
        expected = np.array([1, 3, 2], dtype=np.float32)
        np.testing.assert_array_equal(ranks, expected)

    def test_compute_ranks_with_ties_all_same(self) -> None:
        """All same scores should all get average rank."""
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        ranks = compute_ranks_with_ties(scores)
        # Ranks 1, 2, 3 -> avg = 2.0
        expected = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(ranks, expected)


# =============================================================================
# Test RRF Fusion
# =============================================================================


class TestRRFFusion:
    """Test RRF fusion functions."""

    def test_rrf_fusion_2way_basic(self) -> None:
        """Test basic 2-way RRF computation."""
        scores_a = np.array([0.9, 0.5, 0.3], dtype=np.float32)
        scores_b = np.array([0.3, 0.9, 0.5], dtype=np.float32)

        rrf = rrf_fusion_2way(scores_a, scores_b, k=60)

        # Node 0: rank 1 in A, rank 3 in B -> 1/61 + 1/63
        # Node 1: rank 2 in A, rank 1 in B -> 1/62 + 1/61
        # Node 2: rank 3 in A, rank 2 in B -> 1/63 + 1/62
        node1_score = 1 / 62 + 1 / 61
        node0_score = 1 / 61 + 1 / 63

        # Node 1 should rank higher (better combined)
        assert rrf[1] > rrf[0]
        np.testing.assert_almost_equal(rrf[1], node1_score, decimal=6)
        np.testing.assert_almost_equal(rrf[0], node0_score, decimal=6)

    def test_rrf_fusion_2way_empty(self) -> None:
        """Empty arrays should return empty result."""
        rrf = rrf_fusion_2way(np.array([]), np.array([]), k=60)
        assert len(rrf) == 0

    def test_rrf_fusion_2way_single(self) -> None:
        """Single item should return max possible 2-signal RRF."""
        rrf = rrf_fusion_2way(np.array([0.5]), np.array([0.5]), k=60)
        expected = 2.0 / (60 + 1)
        np.testing.assert_almost_equal(rrf[0], expected, decimal=6)

    def test_rrf_fusion_3way_basic(self) -> None:
        """Test basic 3-way RRF computation."""
        scores_a = np.array([0.9, 0.5], dtype=np.float32)
        scores_b = np.array([0.5, 0.9], dtype=np.float32)
        scores_c = np.array([0.7, 0.7], dtype=np.float32)

        rrf = rrf_fusion_3way(scores_a, scores_b, scores_c, k=60)

        # Both items should have non-zero RRF scores
        assert rrf[0] > 0
        assert rrf[1] > 0

    def test_rrf_fusion_3way_empty(self) -> None:
        """Empty arrays should return empty result."""
        rrf = rrf_fusion_3way(np.array([]), np.array([]), np.array([]), k=60)
        assert len(rrf) == 0

    def test_rrf_fusion_3way_single(self) -> None:
        """Single item should return max possible 3-signal RRF."""
        rrf = rrf_fusion_3way(np.array([0.5]), np.array([0.5]), np.array([0.5]), k=60)
        expected = 3.0 / (60 + 1)
        np.testing.assert_almost_equal(rrf[0], expected, decimal=6)

    def test_rrf_with_tie_handling(self) -> None:
        """Test RRF with tie handling enabled."""
        scores_a = np.array([0.8, 0.8, 0.6], dtype=np.float32)
        scores_b = np.array([0.5, 0.5, 0.9], dtype=np.float32)

        rrf = rrf_fusion_2way(scores_a, scores_b, k=60, use_tie_handling=True)

        # First two items have same score in A, should have same rank
        # This affects their RRF scores
        assert len(rrf) == 3
        # Node 2 has rank 1 in B (0.9 highest), should have good RRF
        assert rrf[2] > 0


# =============================================================================
# Test Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Test that defaults maintain backward compatibility."""

    def test_e1_default_is_cosine(self) -> None:
        """E1 default should be COSINE (current behavior)."""
        from inspect import signature

        from workflows.query_retrieval.strategies.top_down import TopDownRetrieval

        sig = signature(TopDownRetrieval.execute)
        param = sig.parameters.get("scoring_method")
        assert param is not None
        assert param.default == E1ScoringMethod.COSINE

    def test_e2_default_is_rrf_cosine_seed(self) -> None:
        """E2 default should be RRF_COSINE_SEED (current behavior)."""
        from inspect import signature

        from workflows.query_retrieval.strategies.horizontal import HorizontalRetrieval

        sig = signature(HorizontalRetrieval.execute)
        param = sig.parameters.get("scoring_method")
        assert param is not None
        assert param.default == E2ScoringMethod.RRF_COSINE_SEED

    def test_e3_default_is_avg_log(self) -> None:
        """E3 default should be AVG_LOG (current behavior)."""
        from inspect import signature

        from workflows.query_retrieval.strategies.bottom_up import BottomUpRetrieval

        sig = signature(BottomUpRetrieval.execute)
        param = sig.parameters.get("scoring_method")
        assert param is not None
        assert param.default == E3ScoringMethod.AVG_LOG


# =============================================================================
# Test Enum String Conversion
# =============================================================================


class TestEnumConversion:
    """Test enum from string conversion (for CLI args)."""

    def test_e1_from_string(self) -> None:
        """E1 enum should be constructible from string."""
        assert E1ScoringMethod("cosine") == E1ScoringMethod.COSINE
        assert E1ScoringMethod("rrf_content_seed") == E1ScoringMethod.RRF_CONTENT_SEED

    def test_e2_from_string(self) -> None:
        """E2 enum should be constructible from string."""
        assert E2ScoringMethod("cosine") == E2ScoringMethod.COSINE
        assert E2ScoringMethod("rrf_cosine_seed") == E2ScoringMethod.RRF_COSINE_SEED
        assert E2ScoringMethod("rrf_cosine_relation") == E2ScoringMethod.RRF_COSINE_RELATION
        assert (
            E2ScoringMethod("rrf_cosine_seed_relation") == E2ScoringMethod.RRF_COSINE_SEED_RELATION
        )

    def test_e3_from_string(self) -> None:
        """E3 enum should be constructible from string."""
        assert E3ScoringMethod("parent_cosine") == E3ScoringMethod.PARENT_COSINE
        assert E3ScoringMethod("avg_log") == E3ScoringMethod.AVG_LOG
        assert E3ScoringMethod("rrf_parent_avglog") == E3ScoringMethod.RRF_PARENT_AVGLOG
        assert E3ScoringMethod("filtered_avg_log") == E3ScoringMethod.FILTERED_AVG_LOG
        assert E3ScoringMethod("rrf_parent_filtered") == E3ScoringMethod.RRF_PARENT_FILTERED

    def test_invalid_string_raises(self) -> None:
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError):
            E1ScoringMethod("invalid_method")
