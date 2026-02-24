"""Unit tests for E3 Bottom-Up filtered scoring methods and expand_to_content_nodes.

Covers:
- FILTERED_AVG_LOG scoring (median threshold filtering)
- RRF_PARENT_FILTERED scoring (median threshold filtering)
- expand_to_content_nodes (section → children expansion)
"""

from __future__ import annotations

import statistics
from unittest.mock import MagicMock

import numpy as np
import pytest

from workflows.query_retrieval.strategies.bottom_up import BottomUpRetrieval

# =============================================================================
# Test Helpers
# =============================================================================

EMB_DIM = 3072


def _make_embedding(seed: int = 0) -> list[float]:
    """Create a deterministic embedding."""
    return np.random.default_rng(seed).random(EMB_DIM).tolist()


def _make_parents_dict(
    count: int = 5,
    children_per_parent: int = 3,
    base_score: float = 0.8,
) -> dict[str, dict]:
    """Create a mock parents_dict for scoring tests."""
    parents = {}
    for i in range(count):
        pid = f"structure_nodes/parent_{i}"
        scores = [max(0.1, base_score - i * 0.1 - j * 0.02) for j in range(children_per_parent)]
        parents[pid] = {
            "key": f"parent_{i}",
            "title": f"Parent {i}",
            "summary": f"Summary {i}",
            "level": 5,
            "node_type": "section",
            "embedding": _make_embedding(i),
            "children_scores": scores,
        }
    return parents


def _make_mock_db() -> MagicMock:
    """Create a mock database."""
    db = MagicMock()
    db.aql.execute.return_value = iter([])
    return db


# =============================================================================
# FILTERED_AVG_LOG tests
# =============================================================================


class TestFilteredAvgLog:
    def test_single_parent_returned_unfiltered(self):
        """With <=1 parent, filtering is skipped (median meaningless)."""
        parents = _make_parents_dict(count=1)
        retrieval = BottomUpRetrieval(_make_mock_db())
        result = retrieval._score_filtered_avg_log(parents)
        assert len(result) == 1

    def test_empty_parents_returns_empty(self):
        """With no parents, result should be empty."""
        retrieval = BottomUpRetrieval(_make_mock_db())
        result = retrieval._score_filtered_avg_log({})
        assert result == []

    def test_filters_below_median(self):
        """Should keep ~half of parents (those >= median score)."""
        parents = _make_parents_dict(count=6)
        retrieval = BottomUpRetrieval(_make_mock_db())
        result = retrieval._score_filtered_avg_log(parents)
        # Compute expected tau
        full_scores = retrieval._score_parents_avg_log(parents)
        tau = statistics.median([p["aggregated_score"] for p in full_scores])
        for p in result:
            assert p["aggregated_score"] >= tau
        # Should filter ~half
        assert len(result) <= len(parents)
        assert len(result) >= 1

    def test_scoring_method_label_set(self):
        """Output items should have scoring_method = 'filtered_avg_log'."""
        parents = _make_parents_dict(count=4)
        retrieval = BottomUpRetrieval(_make_mock_db())
        result = retrieval._score_filtered_avg_log(parents)
        for p in result:
            assert p["scoring_method"] == "filtered_avg_log"

    def test_tau_is_median_of_aggregated_scores(self):
        """Tau should be median of aggregated_score, not raw cosines."""
        parents = _make_parents_dict(count=5)
        retrieval = BottomUpRetrieval(_make_mock_db())
        full = retrieval._score_parents_avg_log(parents)
        tau = statistics.median([p["aggregated_score"] for p in full])
        filtered = retrieval._score_filtered_avg_log(parents)
        # All filtered should be >= tau
        for p in filtered:
            assert p["aggregated_score"] >= tau
        # At least one from full should be < tau (was filtered out), if count > 1
        if len(full) > 1:
            full_above = [p for p in full if p["aggregated_score"] >= tau]
            assert len(filtered) == len(full_above)


# =============================================================================
# RRF_PARENT_FILTERED tests
# =============================================================================


class TestRrfParentFiltered:
    def test_single_parent_returned_unfiltered(self):
        """With <=1 parent, filtering is skipped."""
        parents = _make_parents_dict(count=1)
        query_emb = _make_embedding(99)
        retrieval = BottomUpRetrieval(_make_mock_db())
        result = retrieval._score_rrf_parent_filtered(query_emb, parents)
        assert len(result) == 1

    def test_filters_below_median_rrf(self):
        """Should keep ~half of parents by RRF score median."""
        parents = _make_parents_dict(count=6)
        query_emb = _make_embedding(99)
        retrieval = BottomUpRetrieval(_make_mock_db())
        result = retrieval._score_rrf_parent_filtered(query_emb, parents)
        assert len(result) >= 1
        assert len(result) <= len(parents)

    def test_scoring_method_label_set(self):
        """Output items should have scoring_method = 'rrf_parent_filtered'."""
        parents = _make_parents_dict(count=4)
        query_emb = _make_embedding(99)
        retrieval = BottomUpRetrieval(_make_mock_db())
        result = retrieval._score_rrf_parent_filtered(query_emb, parents)
        for p in result:
            assert p["scoring_method"] == "rrf_parent_filtered"


# =============================================================================
# expand_to_content_nodes tests
# =============================================================================


class TestExpandToContentNodes:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        """Empty ranked_sections should return empty list."""
        retrieval = BottomUpRetrieval(_make_mock_db())
        result = await retrieval.expand_to_content_nodes([])
        assert result == []

    @pytest.mark.asyncio
    async def test_expands_sections_to_children(self):
        """Should expand each section to its content children."""
        db = _make_mock_db()
        children = [
            {
                "content_id": "content_nodes/c1",
                "content_key": "c1",
                "text": "T1",
                "node_type": "paragraph",
                "position": 0,
            },
            {
                "content_id": "content_nodes/c2",
                "content_key": "c2",
                "text": "T2",
                "node_type": "paragraph",
                "position": 1,
            },
        ]
        db.aql.execute.return_value = iter(children)
        retrieval = BottomUpRetrieval(db)
        sections = [{"parent_id": "structure_nodes/p1"}]
        result = await retrieval.expand_to_content_nodes(sections)
        assert len(result) == 2
        assert result[0]["content_id"] == "content_nodes/c1"

    @pytest.mark.asyncio
    async def test_deduplicates_across_sections(self):
        """Content nodes shared by multiple sections should appear once."""
        db = _make_mock_db()
        shared_child = {
            "content_id": "content_nodes/c1",
            "content_key": "c1",
            "text": "T",
            "node_type": "paragraph",
            "position": 0,
        }
        unique_child = {
            "content_id": "content_nodes/c2",
            "content_key": "c2",
            "text": "T2",
            "node_type": "paragraph",
            "position": 1,
        }
        # First call returns both, second call returns shared
        call_count = 0

        def mock_execute(_query, bind_vars=None):  # noqa: ARG001
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return iter([shared_child, unique_child])
            call_count += 1
            return iter([shared_child])

        db.aql.execute.side_effect = mock_execute
        retrieval = BottomUpRetrieval(db)
        sections = [
            {"parent_id": "structure_nodes/p1"},
            {"parent_id": "structure_nodes/p2"},
        ]
        result = await retrieval.expand_to_content_nodes(sections)
        content_ids = [r["content_id"] for r in result]
        assert len(content_ids) == len(set(content_ids))  # no duplicates
        assert len(result) == 2  # c1 + c2 (c1 not repeated)
