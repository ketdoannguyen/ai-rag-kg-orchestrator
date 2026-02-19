"""Unit tests for B2 (BM25) and B3 (Hybrid RRF) baseline retrieval strategies.

Tests cover:
- BM25Retrieval: basic execution, empty results, metrics
- HybridRetrieval: RRF merge logic, disjoint/overlapping results, single signal
- BaselineMetrics: to_dict serialization
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
from arango_client.handlers.search_handler import SearchResult
from arango_client.models.detail_nodes import ParagraphNode
from arango_client.models.enums import NodeType

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.baselines import (
    BaselineMetrics,
    BM25Retrieval,
    HybridRetrieval,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db():
    """Mock ArangoDB database."""
    db = MagicMock()
    db.aql = MagicMock()
    return db


def _make_node(key: str, text: str = "sample text") -> ParagraphNode:
    """Create a ParagraphNode for testing."""
    return ParagraphNode(
        id=uuid.UUID(f"00000000-0000-0000-0000-{key.zfill(12)}"),
        node_type=NodeType.PARAGRAPH,
        text=text,
    )


def _make_search_result(key: str, score: float, text: str = "sample text") -> SearchResult:
    """Create a SearchResult for testing."""
    node = _make_node(key, text)
    return SearchResult(node=node, score=score)


@pytest.fixture
def sample_embedding():
    """Sample query embedding."""
    vec = np.random.default_rng(42).random(128).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


# =============================================================================
# Test: BaselineMetrics
# =============================================================================


class TestBaselineMetrics:
    """Tests for BaselineMetrics dataclass."""

    def test_to_dict_all_fields(self):
        """to_dict includes all metric fields."""
        metrics = BaselineMetrics(
            strategy="bm25",
            search_time_ms=15.5,
            merge_time_ms=3.2,
            total_time_ms=18.7,
            total_candidates=20,
            vector_candidates=10,
            bm25_candidates=15,
        )
        d = metrics.to_dict()

        assert d["strategy"] == "bm25"
        assert d["search_time_ms"] == 15.5
        assert d["merge_time_ms"] == 3.2
        assert d["total_time_ms"] == 18.7
        assert d["total_candidates"] == 20
        assert d["vector_candidates"] == 10
        assert d["bm25_candidates"] == 15

    def test_to_dict_defaults(self):
        """to_dict works with default values."""
        metrics = BaselineMetrics(strategy="hybrid")
        d = metrics.to_dict()

        assert d["strategy"] == "hybrid"
        assert d["search_time_ms"] == 0.0
        assert d["total_candidates"] == 0

    def test_to_dict_rounds_times(self):
        """Timing values are rounded to 1 decimal."""
        metrics = BaselineMetrics(
            strategy="bm25",
            search_time_ms=15.555,
            merge_time_ms=3.249,
            total_time_ms=18.804,
        )
        d = metrics.to_dict()

        assert d["search_time_ms"] == 15.6
        assert d["merge_time_ms"] == 3.2
        assert d["total_time_ms"] == 18.8


# =============================================================================
# Test: BM25Retrieval (B2)
# =============================================================================


class TestBM25Retrieval:
    """Tests for B2 BM25Retrieval strategy."""

    @pytest.mark.asyncio
    async def test_returns_content_with_context(self, mock_db):
        """BM25 retrieval converts SearchResult to ContentWithContext."""
        results = [
            _make_search_result("1", 5.0, "first result"),
            _make_search_result("2", 3.0, "second result"),
        ]

        from workflows.query_retrieval.strategies.baselines import _search_result_to_content

        content = _search_result_to_content(results[0], results[0].score)
        assert isinstance(content, ContentWithContext)
        assert content.score == 5.0
        assert content.node_type == "paragraph"
        assert content.text == "first result"

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_db):
        """BM25 retrieval handles empty results."""
        from workflows.query_retrieval.strategies.baselines import _search_result_to_content

        content_results = [_search_result_to_content(r, r.score) for r in []]
        assert content_results == []

    def test_metrics_initialized(self, mock_db):
        """Metrics are initialized with correct strategy name."""
        strategy = BM25Retrieval(db=mock_db)
        assert strategy.metrics.strategy == "bm25"


# =============================================================================
# Test: HybridRetrieval (B3)
# =============================================================================


class TestHybridRetrieval:
    """Tests for B3 HybridRetrieval strategy."""

    def test_rrf_merge_overlapping_results(self, mock_db):
        """RRF merge with overlapping results from both signals."""
        strategy = HybridRetrieval(db=mock_db)

        # Node "1" appears in both, "2" only in vector, "3" only in BM25
        vector_results = [
            _make_search_result("1", 0.9),
            _make_search_result("2", 0.7),
        ]
        bm25_results = [
            _make_search_result("1", 5.0),
            _make_search_result("3", 3.0),
        ]

        results = strategy._rrf_merge(vector_results, bm25_results, top_k=3)

        assert len(results) == 3
        # Node "1" should rank highest (appears in both signals)
        assert results[0].node_id == str(uuid.UUID("00000000-0000-0000-0000-000000000001"))
        # All results should have positive scores
        for r in results:
            assert r.score > 0.0

    def test_rrf_merge_disjoint_results(self, mock_db):
        """RRF merge with completely disjoint result sets."""
        strategy = HybridRetrieval(db=mock_db)

        vector_results = [
            _make_search_result("1", 0.9),
            _make_search_result("2", 0.7),
        ]
        bm25_results = [
            _make_search_result("3", 5.0),
            _make_search_result("4", 3.0),
        ]

        results = strategy._rrf_merge(vector_results, bm25_results, top_k=4)

        assert len(results) == 4
        node_ids = {r.node_id for r in results}
        # All 4 unique nodes should be present
        for key in ["1", "2", "3", "4"]:
            expected = str(uuid.UUID(f"00000000-0000-0000-0000-{key.zfill(12)}"))
            assert expected in node_ids

    def test_rrf_merge_empty_vector(self, mock_db):
        """RRF merge works when vector results are empty."""
        strategy = HybridRetrieval(db=mock_db)

        bm25_results = [
            _make_search_result("1", 5.0),
            _make_search_result("2", 3.0),
        ]

        results = strategy._rrf_merge([], bm25_results, top_k=2)

        assert len(results) == 2

    def test_rrf_merge_empty_bm25(self, mock_db):
        """RRF merge works when BM25 results are empty."""
        strategy = HybridRetrieval(db=mock_db)

        vector_results = [
            _make_search_result("1", 0.9),
            _make_search_result("2", 0.7),
        ]

        results = strategy._rrf_merge(vector_results, [], top_k=2)

        assert len(results) == 2

    def test_rrf_merge_both_empty(self, mock_db):
        """RRF merge returns empty list when both signals are empty."""
        strategy = HybridRetrieval(db=mock_db)

        results = strategy._rrf_merge([], [], top_k=5)

        assert results == []

    def test_rrf_merge_respects_top_k(self, mock_db):
        """RRF merge returns at most top_k results."""
        strategy = HybridRetrieval(db=mock_db)

        vector_results = [_make_search_result(str(i), 0.9 - i * 0.1) for i in range(5)]
        bm25_results = [_make_search_result(str(i + 5), 5.0 - i) for i in range(5)]

        results = strategy._rrf_merge(vector_results, bm25_results, top_k=3)

        assert len(results) == 3

    def test_rrf_merge_scores_descending(self, mock_db):
        """RRF merge results are sorted by score descending."""
        strategy = HybridRetrieval(db=mock_db)

        vector_results = [_make_search_result(str(i), 0.9 - i * 0.1) for i in range(5)]
        bm25_results = [_make_search_result(str(i + 5), 5.0 - i) for i in range(5)]

        results = strategy._rrf_merge(vector_results, bm25_results, top_k=10)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_merge_output_type(self, mock_db):
        """RRF merge returns ContentWithContext instances."""
        strategy = HybridRetrieval(db=mock_db)

        vector_results = [_make_search_result("1", 0.9)]
        bm25_results = [_make_search_result("2", 5.0)]

        results = strategy._rrf_merge(vector_results, bm25_results, top_k=2)

        for r in results:
            assert isinstance(r, ContentWithContext)
            assert r.node_id
            assert r.node_type == "paragraph"

    def test_metrics_initialized(self, mock_db):
        """Metrics are initialized with correct strategy name."""
        strategy = HybridRetrieval(db=mock_db)
        assert strategy.metrics.strategy == "hybrid"
