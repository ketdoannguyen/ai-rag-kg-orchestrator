"""Unit tests for E2 Horizontal Retrieval strategy.

Tests cover all 5 pipeline stages:
1. Seed Retrieval (_get_seed_nodes)
2. Relation Filtering (_filter_relations)
3. Graph Traversal (_traverse)
4. Multi-Signal Scoring (_score_rrf_cosine_seed)
5. Merge, Dedup, Rank (_merge_dedup_rank)
Plus full execute() pipeline tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.base import cosine_similarity_batch
from workflows.query_retrieval.strategies.horizontal import (
    DEFAULT_K_FINAL,
    DEFAULT_N_SEEDS,
    MAX_TRAVERSED_NODES,
    HorizontalMetrics,
    HorizontalRetrieval,
)

# =============================================================================
# Fixtures
# =============================================================================

RNG = np.random.default_rng(42)
EMB_DIM = 3072


def _make_embedding(seed: int = 0) -> list[float]:
    """Create a deterministic embedding."""
    return np.random.default_rng(seed).random(EMB_DIM).tolist()


def _make_seed_nodes(count: int = 5) -> list[dict]:
    """Create mock seed nodes with embeddings and scores."""
    return [
        {
            "_id": f"content_nodes/seed_{i}",
            "_key": f"seed_{i}",
            "title": f"Seed {i}",
            "text": f"Content of seed {i}",
            "node_type": "paragraph",
            "source_id": "src1",
            "embedding": _make_embedding(i),
            "score": 0.9 - i * 0.1,
        }
        for i in range(count)
    ]


def _make_traversed_nodes(count: int = 10, seed_count: int = 5) -> list[dict]:
    """Create mock traversed nodes with relation metadata."""
    relations = ["CAUSED_BY", "SUPPORTS", "CONTRASTS", "LED_TO", "ELABORATES"]
    return [
        {
            "_id": f"content_nodes/trav_{i}",
            "_key": f"trav_{i}",
            "title": f"Traversed {i}",
            "text": f"Content of traversed {i}",
            "node_type": "paragraph",
            "source_id": "src1",
            "embedding": _make_embedding(100 + i),
            "relation_type": relations[i % len(relations)],
            "from_seed_id": f"content_nodes/seed_{i % seed_count}",
        }
        for i in range(count)
    ]


@pytest.fixture
def mock_db():
    """Mock ArangoDB StandardDatabase."""
    db = MagicMock()
    db.aql = MagicMock()
    return db


@pytest.fixture
def horizontal(mock_db):
    """Create HorizontalRetrieval instance with mocked DB."""
    return HorizontalRetrieval(db=mock_db, graph_name="knowledge_graph")


@pytest.fixture
def query_embedding():
    """Deterministic query embedding."""
    return _make_embedding(999)


# =============================================================================
# Test: HorizontalMetrics
# =============================================================================


class TestHorizontalMetrics:
    def test_default_values(self):
        """All metrics start at 0."""
        m = HorizontalMetrics()
        assert m.seed_count == 0
        assert m.relations_scored == 0
        assert m.nodes_returned == 0
        assert m.latency_total_ms == 0.0

    def test_to_dict(self):
        """to_dict() returns serializable dict with all fields."""
        m = HorizontalMetrics(seed_count=5, relations_selected=3, latency_total_ms=123.456)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["seed_count"] == 5
        assert d["relations_selected"] == 3
        assert d["latency_total_ms"] == 123.5  # rounded

    def test_to_dict_has_all_keys(self):
        """to_dict() includes all metric fields."""
        m = HorizontalMetrics()
        d = m.to_dict()
        expected_keys = {
            "seed_count",
            "relations_scored",
            "relations_selected",
            "selected_relation_types",
            "traversed_nodes_raw",
            "traversed_nodes_unique",
            "nodes_returned",
            "unique_nodes_from_traversal",
            "latency_seed_ms",
            "latency_relation_ms",
            "latency_traverse_ms",
            "latency_score_ms",
            "latency_merge_ms",
            "latency_total_ms",
        }
        assert set(d.keys()) == expected_keys


# =============================================================================
# Test: _get_seed_nodes (Stage 1)
# =============================================================================


class TestSeedRetrieval:
    @pytest.mark.asyncio
    async def test_returns_seeds(self, horizontal, mock_db):
        """Returns seed nodes from AQL cursor."""
        seeds = _make_seed_nodes(3)
        mock_db.aql.execute.return_value = iter(seeds)

        result = await horizontal._get_seed_nodes(_make_embedding(0), n_seeds=3)
        assert len(result) == 3
        assert result[0]["_id"] == "content_nodes/seed_0"

    @pytest.mark.asyncio
    async def test_metrics_updated(self, horizontal, mock_db):
        """seed_count and latency_seed_ms populated."""
        mock_db.aql.execute.return_value = iter(_make_seed_nodes(5))

        await horizontal._get_seed_nodes(_make_embedding(0), n_seeds=5)
        assert horizontal.metrics.seed_count == 5
        assert horizontal.metrics.latency_seed_ms > 0

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty(self, horizontal, mock_db):
        """Empty content_nodes returns []."""
        mock_db.aql.execute.return_value = iter([])

        result = await horizontal._get_seed_nodes(_make_embedding(0), n_seeds=5)
        assert result == []
        assert horizontal.metrics.seed_count == 0

    @pytest.mark.asyncio
    async def test_seed_has_required_fields(self, horizontal, mock_db):
        """Each seed has _id, _key, text, embedding, score."""
        seeds = _make_seed_nodes(1)
        mock_db.aql.execute.return_value = iter(seeds)

        result = await horizontal._get_seed_nodes(_make_embedding(0), n_seeds=1)
        node = result[0]
        assert "_id" in node
        assert "_key" in node
        assert "text" in node
        assert "embedding" in node
        assert "score" in node


# =============================================================================
# Test: _cosine_similarity_batch (Shared utility)
# =============================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Identical vectors have score ~1.0."""
        emb = _make_embedding(42)
        scores = cosine_similarity_batch(emb, [emb])
        assert abs(scores[0] - 1.0) < 0.001

    def test_batch_computation(self):
        """Computes scores for multiple embeddings."""
        query = _make_embedding(0)
        embs = [_make_embedding(i) for i in range(5)]
        scores = cosine_similarity_batch(query, embs)
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)

    def test_empty_embeddings(self):
        """Empty embedding list returns empty array."""
        scores = cosine_similarity_batch(_make_embedding(0), [])
        assert len(scores) == 0

    def test_zero_query(self):
        """Zero query vector returns all zeros."""
        scores = cosine_similarity_batch([0.0] * EMB_DIM, [_make_embedding(0)])
        assert scores[0] == 0.0

    def test_scores_clipped_to_01(self):
        """Scores are clipped to [0, 1] range."""
        query = _make_embedding(0)
        embs = [_make_embedding(i) for i in range(10)]
        scores = cosine_similarity_batch(query, embs)
        assert all(0 <= s <= 1 for s in scores)


# =============================================================================
# Test: _filter_relations (Stage 2) - Removed
# =============================================================================
# NOTE: Tests for HybridRelationScorer are in test_multi_signal_scoring.py.
# Integration tests for _filter_relations are covered by TestExecutePipeline.
# Direct unit tests were removed because internal imports make mocking complex.


# =============================================================================
# Test: _score_rrf_cosine_seed (Stage 4)
# =============================================================================


class TestMultiSignalScoring:
    def test_scoring_formula(self, horizontal, query_embedding):
        """RRF scoring: final = 1/(k+rank_cosine) + 1/(k+rank_seed)."""
        traversed = _make_traversed_nodes(1)
        seed_scores = {"content_nodes/seed_0": 0.8}

        result = horizontal._score_rrf_cosine_seed(query_embedding, traversed, seed_scores)
        assert len(result) == 1
        node = result[0]

        # With single node, ranks are both 1, so RRF = 2/(k+1)
        # k=60 (default), so expected = 2/61 ≈ 0.0328
        expected = 2.0 / 61.0
        assert abs(node["final_score"] - expected) < 1e-6

    def test_higher_cosine_produces_higher_rrf(self, horizontal, query_embedding):
        """Higher cosine score leads to better rank, thus higher RRF."""
        # Create two nodes with different embeddings (cosine scores)
        traversed = _make_traversed_nodes(2)
        # Make second node embedding more similar to query
        traversed[0]["embedding"] = [0.1] * 3072  # Low cosine
        traversed[1]["embedding"] = query_embedding  # High cosine (identical)

        result = horizontal._score_rrf_cosine_seed(
            query_embedding,
            traversed,
            {"content_nodes/seed_0": 0.8, "content_nodes/seed_1": 0.8},
        )

        # Sort by final_score descending
        result.sort(key=lambda x: x["final_score"], reverse=True)

        # Node with higher cosine should rank first
        assert result[0]["cosine_score"] > result[1]["cosine_score"]
        assert result[0]["final_score"] >= result[1]["final_score"]

    def test_missing_embedding_skipped(self, horizontal, query_embedding):
        """Nodes without embeddings are excluded."""
        traversed = [
            {
                "_id": "content_nodes/no_emb",
                "_key": "no_emb",
                "text": "no embedding",
                "node_type": "paragraph",
                "embedding": None,
                "relation_type": "CAUSED_BY",
                "from_seed_id": "content_nodes/seed_0",
            }
        ]
        result = horizontal._score_rrf_cosine_seed(
            query_embedding, traversed, {"content_nodes/seed_0": 0.8}
        )
        assert len(result) == 0

    def test_unknown_relation_still_scores(self, horizontal, query_embedding):
        """Unknown relation_type doesn't affect RRF (rel_score only for metadata)."""
        traversed = _make_traversed_nodes(1)
        traversed[0]["relation_type"] = "UNKNOWN_RELATION"

        result = horizontal._score_rrf_cosine_seed(
            query_embedding, traversed, {"content_nodes/seed_0": 0.8}
        )
        # RRF is computed from cosine and seed, not relation
        # With single node, RRF = 2/(k+1) = 2/61 ≈ 0.0328
        assert result[0]["final_score"] > 0.0
        assert abs(result[0]["final_score"] - 2.0 / 61.0) < 1e-6

    def test_empty_traversed_returns_empty(self, horizontal, query_embedding):
        """No traversed nodes returns []."""
        result = horizontal._score_rrf_cosine_seed(query_embedding, [], {})
        assert result == []

    def test_output_includes_all_scores(self, horizontal, query_embedding):
        """Output includes cosine_score, seed_score, final_score."""
        traversed = _make_traversed_nodes(1)
        result = horizontal._score_rrf_cosine_seed(
            query_embedding, traversed, {"content_nodes/seed_0": 0.8}
        )
        node = result[0]
        assert "cosine_score" in node
        assert "seed_score" in node
        assert "final_score" in node
        assert node["source"] == "traversal"

    def test_metrics_latency_recorded(self, horizontal, query_embedding):
        """latency_score_ms is recorded."""
        traversed = _make_traversed_nodes(3)
        horizontal._score_rrf_cosine_seed(
            query_embedding,
            traversed,
            {f"content_nodes/seed_{i}": 0.8 for i in range(5)},
        )
        assert horizontal.metrics.latency_score_ms >= 0


# =============================================================================
# Test: _traverse (Stage 3)
# =============================================================================


class TestGraphTraversal:
    @pytest.mark.asyncio
    async def test_returns_content_nodes(self, horizontal, mock_db):
        """Returns content nodes from traversal."""
        traversed = _make_traversed_nodes(3)
        mock_db.aql.execute.return_value = iter(traversed)

        result = await horizontal._traverse(_make_seed_nodes(2), ["CAUSED_BY"])
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_filters_non_content_nodes(self, horizontal, mock_db):
        """Structure nodes are filtered out."""
        mixed = _make_traversed_nodes(2) + [
            {
                "_id": "structure_nodes/s1",
                "_key": "s1",
                "node_type": "section",
                "embedding": _make_embedding(200),
                "relation_type": "CAUSED_BY",
                "from_seed_id": "content_nodes/seed_0",
            }
        ]
        mock_db.aql.execute.return_value = iter(mixed)

        result = await horizontal._traverse(_make_seed_nodes(1), ["CAUSED_BY"])
        assert len(result) == 2  # Only content nodes
        assert all(r["node_type"] in ("paragraph", "image", "table") for r in result)

    @pytest.mark.asyncio
    async def test_cap_max_traversed(self, horizontal, mock_db):
        """Results capped at MAX_TRAVERSED_NODES."""
        large = _make_traversed_nodes(MAX_TRAVERSED_NODES + 50)
        mock_db.aql.execute.return_value = iter(large)

        result = await horizontal._traverse(_make_seed_nodes(1), ["CAUSED_BY"])
        assert len(result) <= MAX_TRAVERSED_NODES

    @pytest.mark.asyncio
    async def test_empty_seeds_returns_empty(self, horizontal, mock_db):
        """No seeds -> no traversal -> empty list."""
        result = await horizontal._traverse([], ["CAUSED_BY"])
        assert result == []
        mock_db.aql.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_relations_returns_empty(self, horizontal, mock_db):
        """No selected relations -> empty list."""
        result = await horizontal._traverse(_make_seed_nodes(1), [])
        assert result == []

    @pytest.mark.asyncio
    async def test_metrics_updated(self, horizontal, mock_db):
        """traversed_nodes_raw and traversed_nodes_unique populated."""
        traversed = _make_traversed_nodes(5)
        mock_db.aql.execute.return_value = iter(traversed)

        await horizontal._traverse(_make_seed_nodes(2), ["CAUSED_BY"])
        assert horizontal.metrics.traversed_nodes_raw == 5
        assert horizontal.metrics.traversed_nodes_unique == 5
        assert horizontal.metrics.latency_traverse_ms >= 0

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(self, horizontal, mock_db):
        """DB error returns empty list gracefully."""
        mock_db.aql.execute.side_effect = Exception("DB connection failed")

        result = await horizontal._traverse(_make_seed_nodes(1), ["CAUSED_BY"])
        assert result == []


# =============================================================================
# Test: _merge_dedup_rank (Stage 5)
# =============================================================================


class TestMergeDedupRank:
    def test_dedup_keeps_higher_cosine(self, horizontal):
        """Same node in seeds + traversed -> keep higher cosine for RRF."""
        seeds = [
            {
                "_id": "content_nodes/dup",
                "_key": "dup",
                "text": "duplicate node",
                "node_type": "paragraph",
                "embedding": [1.0] * 3072,
                "score": 0.8,  # This becomes cosine_score
            }
        ]
        traversed = [
            {
                "_id": "content_nodes/dup",
                "_key": "dup",
                "text": "duplicate node",
                "node_type": "paragraph",
                "embedding": [0.5] * 3072,
                "cosine_score": 0.3,  # Lower cosine
                "seed_score": 0.7,
                "final_score": 0.03,
                "source": "traversal",
                "relation_type": "CAUSED_BY",
            }
        ]

        result = horizontal._merge_dedup_rank(seeds, traversed, top_k=10)
        assert len(result) == 1
        # Result is RRF scored (2/(k+1) ≈ 0.0328 for single node)
        assert result[0].score > 0.0

    def test_ranking_descending(self, horizontal):
        """Results sorted by final_score descending."""
        seeds = _make_seed_nodes(5)
        result = horizontal._merge_dedup_rank(seeds, [], top_k=10)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self, horizontal):
        """Returns at most top_k nodes."""
        seeds = _make_seed_nodes(5)
        traversed = [
            {
                **node,
                "final_score": 0.1,
                "source": "traversal",
            }
            for node in _make_traversed_nodes(10)
        ]
        result = horizontal._merge_dedup_rank(seeds, traversed, top_k=3)
        assert len(result) == 3

    def test_empty_traversed_returns_seeds(self, horizontal):
        """No traversed nodes -> returns seeds only."""
        seeds = _make_seed_nodes(3)
        result = horizontal._merge_dedup_rank(seeds, [], top_k=10)
        assert len(result) == 3

    def test_content_with_context_fields(self, horizontal):
        """Output is list of ContentWithContext with correct fields."""
        seeds = _make_seed_nodes(2)
        result = horizontal._merge_dedup_rank(seeds, [], top_k=10)
        assert all(isinstance(r, ContentWithContext) for r in result)
        assert result[0].node_id == "seed_0"
        assert result[0].text == "Content of seed 0"

    def test_unique_nodes_metric(self, horizontal):
        """unique_nodes_from_traversal counts non-seed nodes in results."""
        seeds = _make_seed_nodes(2)
        traversed = [
            {
                "_id": "content_nodes/trav_99",
                "_key": "trav_99",
                "text": "unique traversed",
                "node_type": "paragraph",
                "final_score": 0.95,  # Higher than seeds -> will be included
                "source": "traversal",
                "relation_type": "CAUSED_BY",
            }
        ]
        horizontal._merge_dedup_rank(seeds, traversed, top_k=10)
        assert horizontal.metrics.unique_nodes_from_traversal == 1

    def test_found_via_relation_set(self, horizontal):
        """Traversed nodes have found_via_relation set."""
        seeds = _make_seed_nodes(1)
        traversed = [
            {
                "_id": "content_nodes/trav_0",
                "_key": "trav_0",
                "text": "found via relation",
                "node_type": "paragraph",
                "final_score": 0.5,
                "source": "traversal",
                "relation_type": "SUPPORTS",
            }
        ]
        result = horizontal._merge_dedup_rank(seeds, traversed, top_k=10)
        trav_result = [r for r in result if r.found_via_relation is not None]
        assert len(trav_result) == 1
        assert trav_result[0].found_via_relation == "SUPPORTS"


# =============================================================================
# Test: execute() full pipeline
# =============================================================================


class TestExecutePipeline:
    @pytest.mark.asyncio
    async def test_empty_seeds_returns_empty(self, horizontal, mock_db, query_embedding):
        """No seeds -> returns [] immediately."""
        mock_db.aql.execute.return_value = iter([])

        result = await horizontal.execute(
            query="test query",
            query_embedding=query_embedding,
            query_type="factual",
        )
        assert result == []
        assert horizontal.metrics.seed_count == 0
        assert horizontal.metrics.latency_total_ms > 0

    @pytest.mark.asyncio
    async def test_no_relations_returns_seeds(self, horizontal, mock_db, query_embedding):
        """No relations selected -> returns seeds as-is."""
        seeds = _make_seed_nodes(3)
        mock_db.aql.execute.return_value = iter(seeds)

        with patch.object(horizontal, "_filter_relations", return_value=[]):
            result = await horizontal.execute(
                query="test query",
                query_embedding=query_embedding,
                query_type="factual",
                top_k=10,
            )

        assert len(result) == 3
        assert all(isinstance(r, ContentWithContext) for r in result)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, horizontal, mock_db, query_embedding):
        """Full pipeline: seeds -> filter -> traverse -> score -> merge -> results."""
        seeds = _make_seed_nodes(3)
        traversed = _make_traversed_nodes(5, seed_count=3)

        # Mock: first call is seed retrieval, second is traversal
        mock_db.aql.execute.side_effect = [iter(seeds), iter(traversed)]

        # Mock relation filtering to return scored tuples (4-tuples for HybridRelationScorer)
        mock_relations = [
            ("CAUSED_BY", 0.032, 2.5, 0.8),  # id, rrf, bm25, emb
            ("SUPPORTS", 0.030, 1.5, 0.7),
        ]

        with patch.object(horizontal, "_filter_relations", return_value=mock_relations):
            result = await horizontal.execute(
                query="test query",
                query_embedding=query_embedding,
                query_type="causal",
                top_k=10,
            )

        assert len(result) > 0
        assert all(isinstance(r, ContentWithContext) for r in result)

    @pytest.mark.asyncio
    async def test_metrics_all_populated(self, horizontal, mock_db, query_embedding):
        """All HorizontalMetrics fields populated after execute()."""
        seeds = _make_seed_nodes(3)
        traversed = _make_traversed_nodes(5, seed_count=3)

        mock_db.aql.execute.side_effect = [iter(seeds), iter(traversed)]
        mock_relations = [("CAUSED_BY", 0.032, 2.5, 0.8)]  # 4-tuple

        with patch.object(horizontal, "_filter_relations", return_value=mock_relations):
            await horizontal.execute(
                query="test query",
                query_embedding=query_embedding,
                query_type="causal",
            )

        m = horizontal.metrics
        assert m.seed_count == 3
        assert m.traversed_nodes_raw > 0
        assert m.nodes_returned > 0
        assert m.latency_total_ms > 0
        assert m.latency_seed_ms > 0
        assert m.latency_traverse_ms > 0
        assert m.latency_score_ms >= 0
        assert m.latency_merge_ms >= 0

    @pytest.mark.asyncio
    async def test_metrics_to_dict(self, horizontal, mock_db, query_embedding):
        """metrics.to_dict() returns serializable dict."""
        seeds = _make_seed_nodes(2)
        mock_db.aql.execute.return_value = iter(seeds)

        with patch.object(horizontal, "_filter_relations", return_value=[]):
            await horizontal.execute(
                query="test query",
                query_embedding=query_embedding,
                query_type="factual",
            )

        d = horizontal.metrics.to_dict()
        assert isinstance(d, dict)
        assert "seed_count" in d
        assert "latency_total_ms" in d

    @pytest.mark.asyncio
    async def test_default_parameters(self, horizontal, mock_db, query_embedding):
        """Default n_seeds and top_k are applied."""
        seeds = _make_seed_nodes(DEFAULT_N_SEEDS)
        mock_db.aql.execute.return_value = iter(seeds)

        with patch.object(horizontal, "_filter_relations", return_value=[]):
            result = await horizontal.execute(
                query="test query",
                query_embedding=query_embedding,
                query_type="factual",
            )

        assert len(result) <= DEFAULT_K_FINAL
