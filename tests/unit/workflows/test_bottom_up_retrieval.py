"""Unit tests for E3 Bottom-Up Retrieval strategy.

Tests cover all 4 pipeline stages:
1. Over-Retrieval (_over_retrieve_content)
2. Parent Discovery (_discover_parents)
3. Parent Scoring (_score_parents_avg_log)
4. Select & Return (_select_and_return)
Plus full execute() pipeline tests.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.bottom_up import (
    DEFAULT_TOP_K,
    MAX_CONTENT_POOL,
    OVER_RETRIEVAL_MULTIPLIER,
    BottomUpMetrics,
    BottomUpRetrieval,
)

# =============================================================================
# Fixtures
# =============================================================================

RNG = np.random.default_rng(42)
EMB_DIM = 3072


def _make_embedding(seed: int = 0) -> list[float]:
    """Create a deterministic embedding."""
    return np.random.default_rng(seed).random(EMB_DIM).tolist()


def _make_content_nodes(count: int = 10, start_score: float = 0.9) -> list[dict]:
    """Create mock content nodes with scores (sorted by score DESC)."""
    return [
        {
            "_id": f"content_nodes/content_{i}",
            "_key": f"content_{i}",
            "node_type": "paragraph",
            "text": f"Content text {i}",
            "score": max(0.1, start_score - i * 0.05),
        }
        for i in range(count)
    ]


def _make_parent_discovery_results(
    content_count: int = 10,
    parent_count: int = 3,
) -> list[dict]:
    """Create mock parent discovery AQL results.

    Args:
        content_count: Total content nodes
        parent_count: Number of unique parents

    Returns:
        List of dicts mapping content to parent
    """
    results = []
    for i in range(content_count):
        parent_idx = i % parent_count
        results.append(
            {
                "content_id": f"content_nodes/content_{i}",
                "content_score": 0.9 - i * 0.05,
                "parent_id": f"structure_nodes/parent_{parent_idx}",
                "parent_key": f"parent_{parent_idx}",
                "parent_title": f"Parent Section {parent_idx}",
                "parent_summary": f"This is the summary for parent section {parent_idx}.",
                "parent_level": 3,
                "parent_node_type": "structure",
            }
        )
    return results


@pytest.fixture
def mock_db():
    """Mock ArangoDB StandardDatabase."""
    db = MagicMock()
    db.aql = MagicMock()
    return db


@pytest.fixture
def bottom_up(mock_db):
    """Create BottomUpRetrieval instance with mocked DB."""
    return BottomUpRetrieval(db=mock_db, graph_name="knowledge_graph")


@pytest.fixture
def query_embedding():
    """Deterministic query embedding."""
    return _make_embedding(999)


# =============================================================================
# Test: BottomUpMetrics
# =============================================================================


class TestBottomUpMetrics:
    def test_default_values(self):
        """All metrics start at 0."""
        m = BottomUpMetrics()
        assert m.content_nodes_retrieved == 0
        assert m.unique_parents_found == 0
        assert m.parents_returned == 0
        assert m.latency_total_ms == 0.0

    def test_to_dict(self):
        """to_dict() returns serializable dict with all fields."""
        m = BottomUpMetrics(
            content_nodes_retrieved=30,
            unique_parents_found=5,
            parents_returned=5,
            latency_total_ms=123.456,
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["content_nodes_retrieved"] == 30
        assert d["unique_parents_found"] == 5
        assert d["parents_returned"] == 5
        assert d["latency_total_ms"] == 123.5  # rounded

    def test_to_dict_has_all_keys(self):
        """to_dict() includes all metric fields."""
        m = BottomUpMetrics()
        d = m.to_dict()
        expected_keys = {
            "content_nodes_retrieved",
            "unique_parents_found",
            "parents_returned",
            "avg_children_per_parent",
            "max_children_per_parent",
            "min_children_per_parent",
            "parent_levels_distribution",
            "parents_with_summary",
            "parents_with_title_fallback",
            "latency_over_retrieve_ms",
            "latency_parent_discovery_ms",
            "latency_scoring_ms",
            "latency_total_ms",
        }
        assert set(d.keys()) == expected_keys


# =============================================================================
# Test: _over_retrieve_content (Stage 1)
# =============================================================================


class TestOverRetrieval:
    @pytest.mark.asyncio
    async def test_returns_correct_count(self, bottom_up, mock_db, query_embedding):
        """Should return K*M content nodes."""
        content = _make_content_nodes(30)
        mock_db.aql.execute.return_value = iter(content)

        result = await bottom_up._over_retrieve_content(
            query_embedding=query_embedding,
            top_k=10,
            multiplier=3,
        )

        assert len(result) == 30

    @pytest.mark.asyncio
    async def test_sorted_by_score(self, bottom_up, mock_db, query_embedding):
        """Results should be sorted by score descending (from AQL)."""
        content = _make_content_nodes(10)
        mock_db.aql.execute.return_value = iter(content)

        result = await bottom_up._over_retrieve_content(
            query_embedding=query_embedding,
            top_k=5,
            multiplier=2,
        )

        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_caps_at_max(self, bottom_up, mock_db, query_embedding):
        """Should cap at MAX_CONTENT_POOL."""
        # Request more than MAX_CONTENT_POOL
        large_content = _make_content_nodes(MAX_CONTENT_POOL + 50)
        mock_db.aql.execute.return_value = iter(large_content)

        result = await bottom_up._over_retrieve_content(
            query_embedding=query_embedding,
            top_k=100,
            multiplier=5,  # 100 * 5 = 500 > MAX_CONTENT_POOL
        )

        # Verify limit parameter passed to AQL was capped
        call_args = mock_db.aql.execute.call_args
        bind_vars = call_args.kwargs.get("bind_vars") or call_args[1].get("bind_vars")
        assert bind_vars["limit"] <= MAX_CONTENT_POOL

    @pytest.mark.asyncio
    async def test_metrics_updated(self, bottom_up, mock_db, query_embedding):
        """content_nodes_retrieved and latency populated."""
        mock_db.aql.execute.return_value = iter(_make_content_nodes(20))

        await bottom_up._over_retrieve_content(
            query_embedding=query_embedding,
            top_k=10,
            multiplier=2,
        )

        assert bottom_up.metrics.content_nodes_retrieved == 20
        assert bottom_up.metrics.latency_over_retrieve_ms > 0

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty(self, bottom_up, mock_db, query_embedding):
        """Empty content_nodes returns []."""
        mock_db.aql.execute.return_value = iter([])

        result = await bottom_up._over_retrieve_content(
            query_embedding=query_embedding,
            top_k=10,
            multiplier=3,
        )

        assert result == []
        assert bottom_up.metrics.content_nodes_retrieved == 0


# =============================================================================
# Test: _discover_parents (Stage 2)
# =============================================================================


class TestParentDiscovery:
    @pytest.mark.asyncio
    async def test_groups_by_parent(self, bottom_up, mock_db):
        """Children should be grouped under correct parent."""
        content = _make_content_nodes(6)
        parent_results = _make_parent_discovery_results(content_count=6, parent_count=2)
        mock_db.aql.execute.return_value = iter(parent_results)

        result = await bottom_up._discover_parents(content)

        assert len(result) == 2  # 2 unique parents
        assert "structure_nodes/parent_0" in result
        assert "structure_nodes/parent_1" in result

    @pytest.mark.asyncio
    async def test_preserves_scores(self, bottom_up, mock_db):
        """Children scores should be preserved in group."""
        content = _make_content_nodes(4)
        parent_results = _make_parent_discovery_results(content_count=4, parent_count=2)
        mock_db.aql.execute.return_value = iter(parent_results)

        result = await bottom_up._discover_parents(content)

        # Parent 0 has content 0, 2 (indices 0, 2 with scores 0.9, 0.8)
        parent_0 = result["structure_nodes/parent_0"]
        assert len(parent_0["children_scores"]) == 2

        # Parent 1 has content 1, 3 (indices 1, 3 with scores 0.85, 0.75)
        parent_1 = result["structure_nodes/parent_1"]
        assert len(parent_1["children_scores"]) == 2

    @pytest.mark.asyncio
    async def test_handles_empty(self, bottom_up, mock_db):
        """Should return {} for empty input."""
        result = await bottom_up._discover_parents([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_metrics_updated(self, bottom_up, mock_db):
        """unique_parents_found and latency populated."""
        content = _make_content_nodes(6)
        parent_results = _make_parent_discovery_results(content_count=6, parent_count=3)
        mock_db.aql.execute.return_value = iter(parent_results)

        await bottom_up._discover_parents(content)

        assert bottom_up.metrics.unique_parents_found == 3
        assert bottom_up.metrics.latency_parent_discovery_ms > 0

    @pytest.mark.asyncio
    async def test_parent_info_preserved(self, bottom_up, mock_db):
        """Parent info (title, summary, level, node_type) preserved."""
        content = _make_content_nodes(2)
        parent_results = _make_parent_discovery_results(content_count=2, parent_count=1)
        mock_db.aql.execute.return_value = iter(parent_results)

        result = await bottom_up._discover_parents(content)

        parent = result["structure_nodes/parent_0"]
        assert parent["key"] == "parent_0"
        assert parent["title"] == "Parent Section 0"
        assert "summary" in parent and parent["summary"] is not None
        assert parent["level"] == 3
        assert parent["node_type"] == "structure"


# =============================================================================
# Test: _group_by_parent (helper)
# =============================================================================


class TestGroupByParent:
    def test_groups_correctly(self, bottom_up):
        """Groups content by parent_id."""
        raw = [
            {
                "content_id": "c1",
                "content_score": 0.9,
                "parent_id": "p1",
                "parent_key": "k1",
                "parent_title": "T1",
                "parent_summary": "S1",
                "parent_level": 2,
                "parent_node_type": "structure",
            },
            {
                "content_id": "c2",
                "content_score": 0.8,
                "parent_id": "p1",  # Same parent
                "parent_key": "k1",
                "parent_title": "T1",
                "parent_summary": "S1",
                "parent_level": 2,
                "parent_node_type": "structure",
            },
            {
                "content_id": "c3",
                "content_score": 0.7,
                "parent_id": "p2",  # Different parent
                "parent_key": "k2",
                "parent_title": "T2",
                "parent_summary": "S2",
                "parent_level": 3,
                "parent_node_type": "structure",
            },
        ]

        result = bottom_up._group_by_parent(raw)

        assert len(result) == 2
        assert result["p1"]["children_scores"] == [0.9, 0.8]
        assert result["p2"]["children_scores"] == [0.7]


# =============================================================================
# Test: _score_parents_avg_log (Stage 3)
# =============================================================================


class TestParentScoring:
    def test_formula_correct(self, bottom_up):
        """AVG * ln(1+count) computed correctly."""
        parents_dict = {
            "p1": {
                "key": "p1",
                "title": "Parent 1",
                "summary": "Summary 1",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.85, 0.80, 0.75, 0.70, 0.65],
            }
        }

        results = bottom_up._score_parents_avg_log(parents_dict)

        # Expected: AVG=0.75, count=5, score = 0.75 * ln(6) ≈ 1.344
        expected = 0.75 * math.log(6)
        assert abs(results[0]["aggregated_score"] - expected) < 0.01

    def test_ranking_order(self, bottom_up):
        """High AVG + high count should rank first."""
        parents_dict = {
            "p1": {  # Should rank 1st: good AVG, good count
                "key": "p1",
                "title": "P1",
                "summary": "S1",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.85, 0.80, 0.75, 0.70, 0.65],  # AVG=0.75, count=5
            },
            "p2": {  # Should rank 2nd: high AVG, low count
                "key": "p2",
                "title": "P2",
                "summary": "S2",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.90, 0.88],  # AVG=0.89, count=2
            },
            "p3": {  # Should rank 3rd: high AVG, single child
                "key": "p3",
                "title": "P3",
                "summary": "S3",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.92],  # AVG=0.92, count=1
            },
        }

        results = bottom_up._score_parents_avg_log(parents_dict)

        # P1: 0.75 * ln(6) ≈ 1.344
        # P2: 0.89 * ln(3) ≈ 0.978
        # P3: 0.92 * ln(2) ≈ 0.638
        assert results[0]["key"] == "p1"
        assert results[1]["key"] == "p2"
        assert results[2]["key"] == "p3"

    def test_single_child_penalty(self, bottom_up):
        """Single child should get ln(2) ≈ 0.69 penalty."""
        parents_dict = {
            "p1": {
                "key": "p1",
                "title": "P1",
                "summary": "S1",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.92],
            }
        }

        results = bottom_up._score_parents_avg_log(parents_dict)

        # Expected: 0.92 * ln(2) ≈ 0.638
        expected = 0.92 * math.log(2)
        assert abs(results[0]["aggregated_score"] - expected) < 0.01

    def test_empty_dict_returns_empty(self, bottom_up):
        """Empty parents_dict returns []."""
        results = bottom_up._score_parents_avg_log({})
        assert results == []

    def test_metrics_updated(self, bottom_up):
        """Metrics populated: avg/max/min children, level distribution."""
        parents_dict = {
            "p1": {
                "key": "p1",
                "title": "P1",
                "summary": "S1",
                "level": 2,
                "node_type": "structure",
                "children_scores": [0.9, 0.8, 0.7],
            },
            "p2": {
                "key": "p2",
                "title": "P2",
                "summary": "S2",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.85],
            },
        }

        bottom_up._score_parents_avg_log(parents_dict)

        assert bottom_up.metrics.avg_children_per_parent == 2.0  # (3+1)/2
        assert bottom_up.metrics.max_children_per_parent == 3
        assert bottom_up.metrics.min_children_per_parent == 1
        assert bottom_up.metrics.parent_levels_distribution == {2: 1, 3: 1}

    def test_output_has_all_fields(self, bottom_up):
        """Output includes aggregated_score, avg_cosine, children_count."""
        parents_dict = {
            "p1": {
                "key": "p1",
                "title": "P1",
                "summary": "S1",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.9, 0.8],
            }
        }

        results = bottom_up._score_parents_avg_log(parents_dict)

        assert "aggregated_score" in results[0]
        assert "avg_cosine" in results[0]
        assert "children_count" in results[0]
        assert results[0]["children_count"] == 2
        assert abs(results[0]["avg_cosine"] - 0.85) < 0.01


# =============================================================================
# Test: _select_and_return (Stage 4)
# =============================================================================


class TestSelectAndReturn:
    def test_returns_top_k(self, bottom_up):
        """Should return exactly K parents."""
        scored_parents = [
            {
                "key": f"p{i}",
                "title": f"T{i}",
                "summary": f"S{i}",
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0 - i * 0.1,
            }
            for i in range(10)
        ]

        results = bottom_up._select_and_return(scored_parents, top_k=5)

        assert len(results) == 5

    def test_fewer_than_k(self, bottom_up):
        """Returns all if fewer than K available."""
        scored_parents = [
            {
                "key": "p0",
                "title": "T0",
                "summary": "S0",
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0,
            }
        ]

        results = bottom_up._select_and_return(scored_parents, top_k=5)

        assert len(results) == 1

    def test_uses_summary(self, bottom_up):
        """Should use summary when available."""
        scored_parents = [
            {
                "key": "p1",
                "title": "Title",
                "summary": "Summary text",
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0,
            }
        ]

        results = bottom_up._select_and_return(scored_parents, top_k=1)

        assert results[0].text == "Summary text"

    def test_fallback_to_title(self, bottom_up):
        """Should fallback to title when summary is None."""
        scored_parents = [
            {
                "key": "p1",
                "title": "Title",
                "summary": None,
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0,
            }
        ]

        results = bottom_up._select_and_return(scored_parents, top_k=1)

        assert results[0].text == "Title"

    def test_fallback_to_title_empty_summary(self, bottom_up):
        """Should fallback to title when summary is empty string."""
        scored_parents = [
            {
                "key": "p1",
                "title": "Title",
                "summary": "   ",  # Whitespace only
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0,
            }
        ]

        results = bottom_up._select_and_return(scored_parents, top_k=1)

        assert results[0].text == "Title"

    def test_node_type_structure(self, bottom_up):
        """All returned nodes should have node_type='structure'."""
        scored_parents = [
            {
                "key": "p1",
                "title": "T1",
                "summary": "S1",
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0,
            }
        ]

        results = bottom_up._select_and_return(scored_parents, top_k=1)

        assert results[0].node_type == "structure"

    def test_found_via_relation_none(self, bottom_up):
        """E3 doesn't use relations, so found_via_relation should be None."""
        scored_parents = [
            {
                "key": "p1",
                "title": "T1",
                "summary": "S1",
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0,
            }
        ]

        results = bottom_up._select_and_return(scored_parents, top_k=1)

        assert results[0].found_via_relation is None

    def test_metrics_updated(self, bottom_up):
        """Metrics track summary vs title usage."""
        scored_parents = [
            {
                "key": "p1",
                "title": "T1",
                "summary": "S1",
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0,
            },
            {
                "key": "p2",
                "title": "T2",
                "summary": None,  # Will use title
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 0.9,
            },
        ]

        bottom_up._select_and_return(scored_parents, top_k=2)

        assert bottom_up.metrics.parents_returned == 2
        assert bottom_up.metrics.parents_with_summary == 1
        assert bottom_up.metrics.parents_with_title_fallback == 1

    def test_output_is_content_with_context(self, bottom_up):
        """Output is list of ContentWithContext."""
        scored_parents = [
            {
                "key": "p1",
                "title": "T1",
                "summary": "S1",
                "level": 3,
                "node_type": "structure",
                "aggregated_score": 1.0,
            }
        ]

        results = bottom_up._select_and_return(scored_parents, top_k=1)

        assert all(isinstance(r, ContentWithContext) for r in results)
        assert results[0].node_id == "p1"
        assert results[0].score == 1.0


# =============================================================================
# Test: execute() full pipeline
# =============================================================================


class TestExecutePipeline:
    @pytest.mark.asyncio
    async def test_empty_content_returns_empty(self, bottom_up, mock_db, query_embedding):
        """No content nodes -> returns [] immediately."""
        mock_db.aql.execute.return_value = iter([])

        result = await bottom_up.execute(
            query_embedding=query_embedding,
            top_k=10,
        )

        assert result == []
        assert bottom_up.metrics.content_nodes_retrieved == 0
        assert bottom_up.metrics.latency_total_ms > 0

    @pytest.mark.asyncio
    async def test_no_parents_returns_empty(self, bottom_up, mock_db, query_embedding):
        """Content but no parents -> returns []."""
        content = _make_content_nodes(5)
        # First call: over-retrieval returns content
        # Second call: parent discovery returns empty (no parents)
        mock_db.aql.execute.side_effect = [iter(content), iter([])]

        result = await bottom_up.execute(
            query_embedding=query_embedding,
            top_k=10,
        )

        assert result == []
        assert bottom_up.metrics.unique_parents_found == 0

    @pytest.mark.asyncio
    async def test_full_pipeline(self, bottom_up, mock_db, query_embedding):
        """Full pipeline: over-retrieve -> parents -> score -> results."""
        content = _make_content_nodes(10)
        parent_results = _make_parent_discovery_results(content_count=10, parent_count=3)

        mock_db.aql.execute.side_effect = [iter(content), iter(parent_results)]

        result = await bottom_up.execute(
            query_embedding=query_embedding,
            top_k=5,
        )

        assert len(result) == 3  # 3 unique parents, all returned since < top_k
        assert all(isinstance(r, ContentWithContext) for r in result)
        assert all(r.node_type == "structure" for r in result)

    @pytest.mark.asyncio
    async def test_respects_top_k(self, bottom_up, mock_db, query_embedding):
        """Returns at most top_k parents."""
        content = _make_content_nodes(20)
        parent_results = _make_parent_discovery_results(content_count=20, parent_count=10)

        mock_db.aql.execute.side_effect = [iter(content), iter(parent_results)]

        result = await bottom_up.execute(
            query_embedding=query_embedding,
            top_k=3,
        )

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_metrics_all_populated(self, bottom_up, mock_db, query_embedding):
        """All BottomUpMetrics fields populated after execute()."""
        content = _make_content_nodes(10)
        parent_results = _make_parent_discovery_results(content_count=10, parent_count=3)

        mock_db.aql.execute.side_effect = [iter(content), iter(parent_results)]

        await bottom_up.execute(
            query_embedding=query_embedding,
            top_k=5,
        )

        m = bottom_up.metrics
        assert m.content_nodes_retrieved == 10
        assert m.unique_parents_found == 3
        assert m.parents_returned == 3
        assert m.latency_total_ms > 0
        assert m.latency_over_retrieve_ms > 0
        assert m.latency_parent_discovery_ms > 0
        assert m.latency_scoring_ms >= 0

    @pytest.mark.asyncio
    async def test_metrics_to_dict(self, bottom_up, mock_db, query_embedding):
        """metrics.to_dict() returns serializable dict."""
        content = _make_content_nodes(5)
        parent_results = _make_parent_discovery_results(content_count=5, parent_count=2)

        mock_db.aql.execute.side_effect = [iter(content), iter(parent_results)]

        await bottom_up.execute(
            query_embedding=query_embedding,
            top_k=10,
        )

        d = bottom_up.metrics.to_dict()
        assert isinstance(d, dict)
        assert "content_nodes_retrieved" in d
        assert "unique_parents_found" in d
        assert "latency_total_ms" in d

    @pytest.mark.asyncio
    async def test_default_parameters(self, bottom_up, mock_db, query_embedding):
        """Default top_k and multiplier are applied."""
        content = _make_content_nodes(DEFAULT_TOP_K * OVER_RETRIEVAL_MULTIPLIER)
        parent_results = _make_parent_discovery_results(
            content_count=len(content), parent_count=DEFAULT_TOP_K
        )

        mock_db.aql.execute.side_effect = [iter(content), iter(parent_results)]

        result = await bottom_up.execute(query_embedding=query_embedding)

        assert len(result) <= DEFAULT_TOP_K

    @pytest.mark.asyncio
    async def test_output_different_from_content_type(self, bottom_up, mock_db, query_embedding):
        """E3 returns structure nodes, not content nodes."""
        content = _make_content_nodes(10)
        parent_results = _make_parent_discovery_results(content_count=10, parent_count=3)

        mock_db.aql.execute.side_effect = [iter(content), iter(parent_results)]

        result = await bottom_up.execute(
            query_embedding=query_embedding,
            top_k=5,
        )

        # All results should be structure type, not paragraph
        for r in result:
            assert r.node_type == "structure"
            assert r.node_type != "paragraph"


# =============================================================================
# Test: Scoring formula verification
# =============================================================================


class TestScoringFormulaVerification:
    """Detailed tests for the AVG * ln(1+count) formula."""

    def test_exact_formula_example_a(self, bottom_up):
        """Example A from plan: 5 children, scores [0.85, 0.80, 0.75, 0.70, 0.65]."""
        parents_dict = {
            "p_a": {
                "key": "p_a",
                "title": "PA",
                "summary": "SA",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.85, 0.80, 0.75, 0.70, 0.65],
            }
        }

        results = bottom_up._score_parents_avg_log(parents_dict)

        # AVG = 0.75, count = 5, score = 0.75 * ln(6) ≈ 1.344
        expected = 0.75 * math.log(6)
        assert abs(results[0]["aggregated_score"] - expected) < 0.001
        assert abs(results[0]["avg_cosine"] - 0.75) < 0.001
        assert results[0]["children_count"] == 5

    def test_exact_formula_example_b(self, bottom_up):
        """Example B from plan: 2 children, scores [0.90, 0.88]."""
        parents_dict = {
            "p_b": {
                "key": "p_b",
                "title": "PB",
                "summary": "SB",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.90, 0.88],
            }
        }

        results = bottom_up._score_parents_avg_log(parents_dict)

        # AVG = 0.89, count = 2, score = 0.89 * ln(3) ≈ 0.978
        expected = 0.89 * math.log(3)
        assert abs(results[0]["aggregated_score"] - expected) < 0.001
        assert abs(results[0]["avg_cosine"] - 0.89) < 0.001
        assert results[0]["children_count"] == 2

    def test_exact_formula_example_c(self, bottom_up):
        """Example C from plan: 1 child, score [0.92]."""
        parents_dict = {
            "p_c": {
                "key": "p_c",
                "title": "PC",
                "summary": "SC",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.92],
            }
        }

        results = bottom_up._score_parents_avg_log(parents_dict)

        # AVG = 0.92, count = 1, score = 0.92 * ln(2) ≈ 0.638
        expected = 0.92 * math.log(2)
        assert abs(results[0]["aggregated_score"] - expected) < 0.001
        assert abs(results[0]["avg_cosine"] - 0.92) < 0.001
        assert results[0]["children_count"] == 1

    def test_ranking_matches_plan(self, bottom_up):
        """Ranking A > B > C as specified in plan."""
        parents_dict = {
            "p_a": {
                "key": "p_a",
                "title": "PA",
                "summary": "SA",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.85, 0.80, 0.75, 0.70, 0.65],
            },
            "p_b": {
                "key": "p_b",
                "title": "PB",
                "summary": "SB",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.90, 0.88],
            },
            "p_c": {
                "key": "p_c",
                "title": "PC",
                "summary": "SC",
                "level": 3,
                "node_type": "structure",
                "children_scores": [0.92],
            },
        }

        results = bottom_up._score_parents_avg_log(parents_dict)

        # Expected ranking: A (1.34) > B (0.98) > C (0.64)
        assert results[0]["key"] == "p_a"
        assert results[1]["key"] == "p_b"
        assert results[2]["key"] == "p_c"

        # Verify approximate scores
        assert 1.3 < results[0]["aggregated_score"] < 1.4  # A ≈ 1.34
        assert 0.9 < results[1]["aggregated_score"] < 1.0  # B ≈ 0.98
        assert 0.6 < results[2]["aggregated_score"] < 0.7  # C ≈ 0.64
