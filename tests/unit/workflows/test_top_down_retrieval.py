"""Unit tests for E1 Top-Down Retrieval strategy.

Tests cover all 4 pipeline stages:
- Content-rich pool building (pre-filter)
- Structure cosine search
- Content harvest (descend)
- Re-rank & return
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from workflows.query_retrieval.models import ContentWithContext
from workflows.query_retrieval.strategies.top_down import (
    MAX_COLLECTED_NODES,
    TopDownRetrieval,
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


@pytest.fixture
def strategy(mock_db):
    """TopDownRetrieval instance with mock DB."""
    return TopDownRetrieval(db=mock_db)


@pytest.fixture
def sample_embedding():
    """Sample query embedding (normalized unit vector)."""
    vec = np.random.default_rng(42).random(128).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _make_pool_node(node_id, key, embedding=None, level=4, hc_count=3):
    """Helper to create a pool node dict."""
    return {
        "_id": f"structure_nodes/{node_id}",
        "_key": node_id,
        "embedding": embedding,
        "level": level,
        "summary": f"Summary for {key}",
        "has_content_count": hc_count,
    }


def _make_content_node(node_id, text="content text", embedding=None, depth=0):
    """Helper to create a harvested content node dict."""
    return {
        "_id": f"content_nodes/{node_id}",
        "_key": node_id,
        "text": text,
        "embedding": embedding,
        "node_type": "paragraph",
        "source": "structure_nodes/s1",
        "depth": depth,
    }


# =============================================================================
# Test: Content-Rich Pool (Task 2)
# =============================================================================


class TestBuildContentRichPool:
    """Tests for _build_content_rich_pool()."""

    @pytest.mark.asyncio
    async def test_returns_pool_from_aql(self, strategy, mock_db):
        """Pool is built from AQL query results."""
        pool_data = [
            _make_pool_node("s1", "section-1", level=4, hc_count=4),
            _make_pool_node("s2", "section-2", level=5, hc_count=5),
        ]
        mock_db.aql.execute.return_value = iter(pool_data)

        pool = await strategy._build_content_rich_pool()

        assert len(pool) == 2
        assert pool[0]["_key"] == "s1"
        mock_db.aql.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_source_id_passed_to_aql(self, strategy, mock_db):
        """source_id is passed as bind_var to AQL query."""
        mock_db.aql.execute.return_value = iter([])

        await strategy._build_content_rich_pool(source_id="src-123")

        call_args = mock_db.aql.execute.call_args
        assert call_args[1]["bind_vars"]["source_id"] == "src-123"

    @pytest.mark.asyncio
    async def test_cache_hit(self, strategy, mock_db):
        """Second call returns cached result, no DB query."""
        pool_data = [_make_pool_node("s1", "section-1")]
        mock_db.aql.execute.return_value = iter(pool_data)

        result1 = await strategy._build_content_rich_pool()
        result2 = await strategy._build_content_rich_pool()

        assert result1 == result2
        assert mock_db.aql.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_per_source_id(self, strategy, mock_db):
        """Different source_ids have separate cache entries."""
        mock_db.aql.execute.side_effect = [
            iter([_make_pool_node("s1", "a")]),
            iter([_make_pool_node("s2", "b")]),
        ]

        r1 = await strategy._build_content_rich_pool(source_id="src-1")
        r2 = await strategy._build_content_rich_pool(source_id="src-2")

        assert r1[0]["_key"] == "s1"
        assert r2[0]["_key"] == "s2"
        assert mock_db.aql.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_pool(self, strategy, mock_db):
        """Returns empty list for 0-node pool."""
        mock_db.aql.execute.return_value = iter([])

        pool = await strategy._build_content_rich_pool()

        assert pool == []

    def test_invalidate_cache(self, strategy):
        """invalidate_pool_cache clears all cached pools."""
        strategy._pool_cache["__global__"] = [{"_id": "test"}]
        strategy._pool_cache["src-1"] = [{"_id": "test2"}]

        strategy.invalidate_pool_cache()

        assert strategy._pool_cache == {}


# =============================================================================
# Test: Structure Cosine Search (Task 3)
# =============================================================================


class TestStructureCosineSearch:
    """Tests for _structure_cosine_search()."""

    def test_returns_top_k(self, strategy, sample_embedding):
        """Returns exactly top_k results sorted by score desc."""
        rng = np.random.default_rng(42)
        pool = [
            _make_pool_node(f"s{i}", f"sec-{i}", embedding=rng.random(128).tolist())
            for i in range(10)
        ]

        results = strategy._structure_cosine_search(sample_embedding, pool, top_k=3)

        assert len(results) == 3
        # Verify descending order
        scores = [r["structure_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_only_pool_nodes_returned(self, strategy, sample_embedding):
        """Only nodes from the pool are returned."""
        pool = [
            _make_pool_node("s1", "sec-1", embedding=sample_embedding),
        ]

        results = strategy._structure_cosine_search(sample_embedding, pool, top_k=5)

        assert len(results) == 1
        assert results[0]["_key"] == "s1"

    def test_skips_nodes_without_embedding(self, strategy, sample_embedding):
        """Nodes with null/empty embedding are excluded."""
        pool = [
            _make_pool_node("s1", "with-emb", embedding=sample_embedding),
            _make_pool_node("s2", "no-emb", embedding=None),
            _make_pool_node("s3", "empty-emb", embedding=[]),
        ]

        results = strategy._structure_cosine_search(sample_embedding, pool, top_k=5)

        node_keys = [r["_key"] for r in results]
        assert "s1" in node_keys
        assert "s2" not in node_keys
        assert "s3" not in node_keys

    def test_empty_pool_returns_empty(self, strategy, sample_embedding):
        """Empty pool returns empty results."""
        results = strategy._structure_cosine_search(sample_embedding, [], top_k=5)
        assert results == []

    def test_scores_in_valid_range(self, strategy, sample_embedding):
        """All scores are in [0, 1] range."""
        rng = np.random.default_rng(99)
        pool = [
            _make_pool_node(f"s{i}", f"sec-{i}", embedding=rng.random(128).tolist())
            for i in range(5)
        ]

        results = strategy._structure_cosine_search(sample_embedding, pool, top_k=5)

        for r in results:
            assert 0.0 <= r["structure_score"] <= 1.0

    def test_identical_embedding_scores_1(self, strategy, sample_embedding):
        """Identical embedding to query should score ~1.0."""
        pool = [_make_pool_node("s1", "identical", embedding=sample_embedding)]

        results = strategy._structure_cosine_search(sample_embedding, pool, top_k=1)

        assert len(results) == 1
        assert results[0]["structure_score"] == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# Test: Content Harvest (Task 4)
# =============================================================================


class TestContentHarvest:
    """Tests for _content_harvest()."""

    @pytest.mark.asyncio
    async def test_returns_harvested_nodes(self, strategy, mock_db):
        """Returns content nodes from AQL harvest query."""
        harvest_data = [
            _make_content_node("c1", depth=0),
            _make_content_node("c2", depth=1),
        ]
        mock_db.aql.execute.return_value = iter(harvest_data)

        result = await strategy._content_harvest(["structure_nodes/s1"])

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_deduplication(self, strategy, mock_db):
        """Duplicate nodes are deduplicated, keeping minimum depth."""
        harvest_data = [
            _make_content_node("c1", depth=0),
            _make_content_node("c1", depth=1),  # Duplicate, higher depth
            _make_content_node("c1", depth=2),  # Duplicate, even higher
        ]
        mock_db.aql.execute.return_value = iter(harvest_data)

        result = await strategy._content_harvest(["structure_nodes/s1"])

        assert len(result) == 1
        assert result[0]["depth"] == 0  # Kept minimum depth

    @pytest.mark.asyncio
    async def test_collection_cap(self, strategy, mock_db):
        """More than MAX_COLLECTED_NODES nodes are capped."""
        harvest_data = [
            _make_content_node(f"c{i}", depth=i % 3) for i in range(MAX_COLLECTED_NODES + 50)
        ]
        mock_db.aql.execute.return_value = iter(harvest_data)

        result = await strategy._content_harvest(["structure_nodes/s1"])

        assert len(result) <= MAX_COLLECTED_NODES

    @pytest.mark.asyncio
    async def test_cap_prefers_shallow(self, strategy, mock_db):
        """When capped, shallow (depth=0) nodes are preferred."""
        # Create nodes: 60 at depth=0, 60 at depth=2
        harvest_data = [_make_content_node(f"shallow-{i}", depth=0) for i in range(60)] + [
            _make_content_node(f"deep-{i}", depth=2) for i in range(60)
        ]
        mock_db.aql.execute.return_value = iter(harvest_data)

        result = await strategy._content_harvest(["structure_nodes/s1"])

        assert len(result) == MAX_COLLECTED_NODES
        # All shallow nodes should be included
        shallow_count = sum(1 for r in result if r["depth"] == 0)
        assert shallow_count == 60

    @pytest.mark.asyncio
    async def test_empty_input(self, strategy, mock_db):
        """Empty structure_node_ids returns empty result."""
        result = await strategy._content_harvest([])

        assert result == []
        mock_db.aql.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_bind_vars_passed(self, strategy, mock_db):
        """Structure node IDs and max depth passed as bind vars."""
        mock_db.aql.execute.return_value = iter([])

        await strategy._content_harvest(["structure_nodes/s1", "structure_nodes/s2"])

        call_args = mock_db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["structure_node_ids"] == [
            "structure_nodes/s1",
            "structure_nodes/s2",
        ]
        assert bind_vars["max_descend_depth"] == 2


# =============================================================================
# Test: Re-rank (Task 5)
# =============================================================================


class TestRerank:
    """Tests for _rerank_cosine()."""

    def test_correct_ordering(self, strategy):
        """Nodes ordered by cosine score descending."""
        # Create embeddings with known similarity order
        query_emb = [1.0, 0.0, 0.0]
        nodes = [
            _make_content_node("c1", embedding=[0.5, 0.5, 0.0]),  # Lower sim
            _make_content_node("c2", embedding=[1.0, 0.0, 0.0]),  # Highest sim
            _make_content_node("c3", embedding=[0.7, 0.3, 0.0]),  # Medium sim
        ]

        results = strategy._rerank_cosine(query_emb, nodes, top_k=3)

        assert len(results) == 3
        assert results[0].node_id == "c2"  # Highest score first
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self, strategy):
        """Only top_k nodes returned from larger set."""
        query_emb = [1.0, 0.0, 0.0]
        nodes = [
            _make_content_node(f"c{i}", embedding=[float(i) / 10, 0.5, 0.5]) for i in range(1, 11)
        ]

        results = strategy._rerank_cosine(query_emb, nodes, top_k=3)

        assert len(results) == 3

    def test_handles_missing_embeddings(self, strategy):
        """Nodes without embeddings excluded from primary ranking."""
        query_emb = [1.0, 0.0, 0.0]
        nodes = [
            _make_content_node("c1", embedding=[1.0, 0.0, 0.0]),
            _make_content_node("c2", embedding=None),
        ]

        results = strategy._rerank_cosine(query_emb, nodes, top_k=5)

        # c1 should be first (has embedding), c2 second (no embedding, score=0)
        assert results[0].node_id == "c1"
        assert results[0].score > 0

    def test_empty_input(self, strategy):
        """Empty content_nodes returns empty result."""
        results = strategy._rerank_cosine([1.0, 0.0], [], top_k=5)
        assert results == []

    def test_converts_to_content_with_context(self, strategy):
        """Output is list[ContentWithContext] with correct fields."""
        query_emb = [1.0, 0.0, 0.0]
        nodes = [_make_content_node("c1", text="hello", embedding=[1.0, 0.0, 0.0])]

        results = strategy._rerank_cosine(query_emb, nodes, top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], ContentWithContext)
        assert results[0].node_id == "c1"
        assert results[0].text == "hello"
        assert results[0].node_type == "paragraph"
        assert results[0].found_via_relation is None

    def test_score_range(self, strategy):
        """All scores in [0, 1] range."""
        rng = np.random.default_rng(42)
        query_emb = rng.random(128).tolist()
        nodes = [
            _make_content_node(f"c{i}", embedding=rng.random(128).tolist()) for i in range(10)
        ]

        results = strategy._rerank_cosine(query_emb, nodes, top_k=10)

        for r in results:
            assert 0.0 <= r.score <= 1.0


# =============================================================================
# Test: Full Pipeline (Task 6)
# =============================================================================


class TestTopDownExecute:
    """Tests for the full execute() pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, strategy, mock_db, sample_embedding):
        """Full pipeline: pool → search → harvest → rerank."""
        # Pool data
        pool_data = [
            _make_pool_node("s1", "sec-1", embedding=sample_embedding, hc_count=3),
        ]
        # Harvest data
        harvest_data = [
            _make_content_node("c1", text="result text", embedding=sample_embedding),
        ]

        mock_db.aql.execute.side_effect = [
            iter(pool_data),
            iter(harvest_data),
        ]

        results = await strategy.execute(query_embedding=sample_embedding, top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], ContentWithContext)
        assert results[0].text == "result text"

    @pytest.mark.asyncio
    async def test_metrics_populated(self, strategy, mock_db, sample_embedding):
        """Metrics are populated after execution."""
        pool_data = [
            _make_pool_node("s1", "sec-1", embedding=sample_embedding),
        ]
        harvest_data = [
            _make_content_node("c1", embedding=sample_embedding),
        ]
        mock_db.aql.execute.side_effect = [
            iter(pool_data),
            iter(harvest_data),
        ]

        await strategy.execute(query_embedding=sample_embedding, top_k=5)

        m = strategy.metrics
        assert m.pool_size == 1
        assert m.structure_nodes_matched == 1
        assert m.content_nodes_harvested == 1
        assert m.content_nodes_returned == 1
        assert m.latency_total_ms > 0

    @pytest.mark.asyncio
    async def test_empty_pool_returns_empty(self, strategy, mock_db, sample_embedding):
        """Empty pool results in empty output."""
        mock_db.aql.execute.return_value = iter([])

        results = await strategy.execute(query_embedding=sample_embedding, top_k=5)

        assert results == []
        assert strategy.metrics.pool_size == 0

    @pytest.mark.asyncio
    async def test_metrics_to_dict(self, strategy, mock_db, sample_embedding):
        """Metrics.to_dict() returns serializable dict."""
        mock_db.aql.execute.return_value = iter([])

        await strategy.execute(query_embedding=sample_embedding, top_k=5)

        d = strategy.metrics.to_dict()
        assert isinstance(d, dict)
        assert "pool_size" in d
        assert "latency_total_ms" in d
